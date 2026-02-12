# losses/damon_loss.py
import torch.nn.functional as F

def contact_loss(pred, gt):
    return F.cross_entropy(
        pred.view(-1, 2),
        gt.view(-1),
        weight=pred.new_tensor([0.3, 0.7])  # balance
    )

def mesh_loss(pred, gt):
    return ((pred - gt) ** 2).mean()
