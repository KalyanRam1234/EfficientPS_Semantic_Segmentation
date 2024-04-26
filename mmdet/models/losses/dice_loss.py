import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weight_reduce_loss
import torch
import torch.nn.functional as F

def dice_loss(pred, target, weight=None, reduction='mean', avg_factor=None):
    """
    Dice loss for semantic segmentation.
    
    Args:
        pred (torch.Tensor): The raw predicted logits.
        target (torch.Tensor): The ground truth segmentation masks (integer values).
        weight (torch.Tensor, optional): Optional tensor of weights for each pixel.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default is 'mean'.
        avg_factor (int, optional): Average factor for computing the mean.
    
    Returns:
        torch.Tensor: The computed Dice loss.
    """
    print(pred.shape)
    # Apply softmax to the predictions
    # pred_softmax = F.softmax(pred, dim=1)
    max_probs_indices = torch.argmax(pred, dim=1)
    predval = torch.eye(pred.shape[1],requires_grad=True)[max_probs_indices]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predval=predval.to(device)
    
    # Convert integer labels to one-hot encoding
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).float()   
    print(target_one_hot.shape)
    print("Softmax vals")
    print(predval.shape)
    # print(predval)
    # Compute Dice score
    smooth = 1e-7
    # print(torch.sum(predval*target_one_hot))
    intersection = torch.sum(predval * target_one_hot)
    cardinality = torch.sum(predval) + torch.sum(target_one_hot)
    dice = (2. * intersection + smooth) / (cardinality + smooth)

    # Compute Dice loss
    loss = 1. - dice

    # Apply weights and do reduction
    if weight is not None:
        loss = loss * weight

    if reduction == 'mean':
        if avg_factor is None:
            avg_factor = max(torch.sum(weight).item(), 1.0)
        loss = loss.sum() / avg_factor
    elif reduction == 'sum':
        loss = loss.sum()

    return loss


@LOSSES.register_module
class DiceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 loss_weight=1.0):
        super(DiceLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.cls_criterion=dice_loss

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
