# @Time    : 21/09/2022 23:06
# @Author  : BubblyYi
# @FileName: gc_optimizer.py
# @Software: PyCharm
import gc

import torch
from mmcv.runner import HOOKS, Hook
from torch.nn.utils import clip_grad


@HOOKS.register_module()
class GC_OptimizerHook(Hook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()
        del runner.outputs['loss']
        gc.collect()
        torch.cuda.empty_cache()
