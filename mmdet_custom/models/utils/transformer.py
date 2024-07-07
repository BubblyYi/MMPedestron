# @Time    : 01/02/2024 16:39
# @Author  : BubblyYi
# @FileName: transformer.py
# @Software: PyCharm
from fairscale.nn.checkpoint import checkpoint_wrapper
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DetrTransformerEncoder_CP(TransformerLayerSequence):
    """TransformerEncoder of DETR.

    Args:
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(
            self, *args, post_norm_cfg=dict(type='LN'),
            with_cp=-1, **kwargs):
        super(DetrTransformerEncoder_CP, self).__init__(*args, **kwargs)
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(
                post_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        else:
            assert not self.pre_norm, f'Use prenorm in ' \
                                      f'{self.__class__.__name__},' \
                                      f'Please specify post_norm_cfg'
            self.post_norm = None
        self.with_cp = with_cp
        if self.with_cp > 0:
            for i in range(self.with_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
