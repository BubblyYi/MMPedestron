# @Time    : 31/01/2024 14:57
# @Author  : BubblyYi
# @FileName: mmpedestron.py
# @Software: PyCharm
from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS

from .co_detr import CoDETR


@DETECTORS.register_module()
class MMPedestron(CoDETR):

    def extract_feat(self, img, extra_mod_img=None):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img, extra_mod_img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        extra_mod_img = kwargs.get('extra_mod_img', None)

        batch_input_shape = tuple(img[0].size()[-2:])

        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        if not self.with_attn_mask:  # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, extra_mod_img)

        losses = dict()

        def upd_loss(losses, idx, weight=1):
            new_losses = dict()
            for k, v in losses.items():
                new_k = '{}{}'.format(k, idx)
                if isinstance(v, list) or isinstance(v, tuple):
                    new_losses[new_k] = [i * weight for i in v]
                else:
                    new_losses[new_k] = v * weight
            return new_losses

        # DETR encoder and decoder forward
        if self.with_query_head:
            bbox_losses, x = self.query_head.forward_train(
                x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
            losses.update(bbox_losses)

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg[self.head_idx].get(
                'rpn_proposal', self.test_cfg[self.head_idx].rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        positive_coords = []
        for i in range(len(self.roi_head)):
            roi_losses = self.roi_head[i].forward_train(
                x, img_metas, proposal_list, gt_bboxes, gt_labels,
                gt_bboxes_ignore, gt_masks, **kwargs)
            if self.with_pos_coord:
                positive_coords.append(roi_losses.pop('pos_coords'))
            else:
                if 'pos_coords' in roi_losses.keys():
                    tmp = roi_losses.pop('pos_coords')
            roi_losses = upd_loss(roi_losses, idx=i)
            losses.update(roi_losses)

        for i in range(len(self.bbox_head)):
            bbox_losses = self.bbox_head[i].forward_train(
                x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
            if self.with_pos_coord:
                pos_coords = bbox_losses.pop('pos_coords')
                positive_coords.append(pos_coords)
            else:
                if 'pos_coords' in bbox_losses.keys():
                    tmp = bbox_losses.pop('pos_coords')
            bbox_losses = upd_loss(bbox_losses, idx=i + len(self.roi_head))
            losses.update(bbox_losses)

        if self.with_pos_coord and len(positive_coords) > 0:
            for i in range(len(positive_coords)):
                bbox_losses = self.query_head.forward_train_aux(
                    x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore,
                    positive_coords[i], i)
                bbox_losses = upd_loss(bbox_losses, idx=i)
                losses.update(bbox_losses)

        return losses

    def simple_test_roi_head(self,
                             img,
                             extra_mod_img,
                             img_metas,
                             proposals=None,
                             rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask:  # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, extra_mod_img)
        if self.with_query_head:
            results = self.query_head.forward(x, img_metas)
            x = results[-1]
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head[self.eval_index].simple_test(x,
                                                          proposal_list,
                                                          img_metas,
                                                          rescale=rescale)

    def simple_test_query_head(self,
                               img,
                               extra_mod_img,
                               img_metas,
                               proposals=None,
                               rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        index = 0
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask:  # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, extra_mod_img)
        results_list = self.query_head.simple_test(x,
                                                   img_metas,
                                                   rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.query_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def simple_test_bbox_head(self,
                              img,
                              extra_mod_img,
                              img_metas,
                              proposals=None,
                              rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """

        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask:  # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, extra_mod_img)
        if self.with_query_head:
            results = self.query_head.forward(x, img_metas)
            x = results[-1]
        results_list = self.bbox_head[self.eval_index].simple_test(
            x, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels,
                        self.bbox_head[self.eval_index].num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def simple_test(self,
                    img,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation."""
        assert self.eval_module in ['detr', 'one-stage', 'two-stage']

        extra_mod_img = kwargs.get('extra_mod_img', None)[0]

        if self.with_bbox and self.eval_module == 'one-stage':
            return self.simple_test_bbox_head(img, extra_mod_img, img_metas,
                                              proposals, rescale)
        if self.with_roi_head and self.eval_module == 'two-stage':
            return self.simple_test_roi_head(img, extra_mod_img, img_metas,
                                             proposals, rescale)
        return self.simple_test_query_head(img, extra_mod_img, img_metas,
                                           proposals, rescale)
