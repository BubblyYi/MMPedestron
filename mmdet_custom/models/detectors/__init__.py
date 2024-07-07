# @Time    : 31/01/2024 14:58
# @Author  : BubblyYi
# @FileName: __init__.py
# @Software: PyCharm
from .co_detr import CoDETR
from .dual_faster_rcnn import DualFasterRCNN
from .mmpedestron import MMPedestron
from .multi_faster_rcnn import MulitFastRCNN

__all__ = [
    'CoDETR', 'MMPedestron', 'DualFasterRCNN', 'MulitFastRCNN'
]
