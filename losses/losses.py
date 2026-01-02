"""
变化检测损失函数（1通道二值输出版本）
- BCE + Dice组合损失
- 边界感知损失
- 深度监督支持
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


class DiceLoss(nn.Module):
    """Dice损失（二值版本）"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) logits
            target: (B, H, W) 二值标签 (0/1)
        """
        pred_prob = torch.sigmoid(pred).squeeze(1)  # (B, H, W)
        target_float = target.float()  # (B, H, W)

        intersection = (pred_prob * target_float).sum()
        union = pred_prob.sum() + target_float.sum()
        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice


class BoundaryLoss(nn.Module):
    """边界感知损失"""
    def __init__(self, boundary_weight=2.0):
        super().__init__()
        self.boundary_weight = boundary_weight

        # Sobel算子
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

    def extract_boundary(self, mask):
        """提取边界"""
        device = mask.device

        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)

        mask_float = mask.unsqueeze(1).float()  # (B, 1, H, W)

        edge_x = F.conv2d(mask_float, sobel_x, padding=1)
        edge_y = F.conv2d(mask_float, sobel_y, padding=1)
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)

        boundary = (edges > 0.1).float()
        return boundary

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) logits
            target: (B, H, W) 二值标签
        """
        boundary = self.extract_boundary(target)  # (B, 1, H, W)

        # BCE损失（逐像素）
        target_float = target.unsqueeze(1).float()  # (B, 1, H, W)
        bce = F.binary_cross_entropy_with_logits(pred, target_float, reduction='none')

        # 边界权重map
        weight_map = 1.0 + self.boundary_weight * boundary

        # 加权BCE
        weighted_bce = bce * weight_map
        return weighted_bce.mean()


class ChangeLoss(nn.Module):
    """
    变化检测组合损失（1通道二值版本）

    支持:
    - BCE + Dice
    - 边界感知
    - 类别权重（pos_weight处理不平衡）
    - 深度监督
    """

    def __init__(self):
        super().__init__()

        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss(boundary_weight=2.0)

        # 损失权重
        self.w_bce = cfg.LOSS_WEIGHT_BCE
        self.w_dice = cfg.LOSS_WEIGHT_DICE
        self.w_boundary = cfg.LOSS_WEIGHT_BOUNDARY

        # 类别权重（处理不平衡：变化像素少）
        if cfg.USE_CLASS_WEIGHTS:
            # pos_weight用于BCEWithLogitsLoss，表示正样本的权重
            # 如果CLASS_WEIGHTS = [1.0, 10.0]，则pos_weight = 10.0
            self.pos_weight = torch.tensor([cfg.CLASS_WEIGHTS[1]])
        else:
            self.pos_weight = None

        # 深度监督权重
        if cfg.DEEP_SUPERVISION:
            self.ds_weights = cfg.DEEP_SUPERVISION_WEIGHTS
        else:
            self.ds_weights = None

    def compute_loss(self, pred, target):
        """计算单个预测的损失"""
        device = pred.device

        # BCE损失
        target_float = target.unsqueeze(1).float()  # (B, 1, H, W)

        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(device)
            loss_bce = F.binary_cross_entropy_with_logits(
                pred, target_float, pos_weight=pos_weight
            )
        else:
            loss_bce = F.binary_cross_entropy_with_logits(pred, target_float)

        # Dice损失
        loss_dice = self.dice_loss(pred, target)

        # 边界损失
        if self.w_boundary > 0:
            loss_boundary = self.boundary_loss(pred, target)
        else:
            loss_boundary = torch.tensor(0.0, device=device)

        # 组合
        total = self.w_bce * loss_bce + self.w_dice * loss_dice + self.w_boundary * loss_boundary

        return total, {
            'bce': loss_bce.item(),
            'dice': loss_dice.item(),
            'boundary': loss_boundary.item()
        }

    def forward(self, outputs, targets):
        """
        Args:
            outputs: {
                'pred': (B, 1, H, W) 主预测logits
                'aux_preds': [(B, 1, H, W), ...] 辅助预测（深度监督）
            }
            targets: (B, H, W) 二值标签

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失详情
        """
        pred = outputs['pred']
        label = targets

        # 主损失
        main_loss, loss_dict = self.compute_loss(pred, label)

        total_loss = main_loss
        loss_dict['main'] = main_loss.item()

        # 深度监督损失
        if self.training and 'aux_preds' in outputs and self.ds_weights is not None:
            aux_losses = []
            for i, aux_pred in enumerate(outputs['aux_preds']):
                aux_loss, _ = self.compute_loss(aux_pred, label)
                aux_losses.append(aux_loss * self.ds_weights[i])
                loss_dict[f'aux_{i}'] = aux_loss.item()

            ds_loss = sum(aux_losses)
            total_loss = total_loss + ds_loss
            loss_dict['deep_supervision'] = ds_loss.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


class SCDLoss(nn.Module):
    """
    Snow Cover Days regression loss.

    outputs['pred']: (B, 1, H, W) predicted SCD in days (float)
    targets: (B, 1, H, W) or (B, H, W) ground-truth SCD in days (float)
    """

    def __init__(self):
        super().__init__()
        loss_type = str(getattr(cfg, "LOSS_TYPE", "smoothl1")).lower()
        if loss_type in ("l1", "mae"):
            self.loss_fn = nn.L1Loss()
        elif loss_type in ("mse", "l2"):
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.SmoothL1Loss(beta=1.0)

    def forward(self, outputs, targets):
        pred = outputs["pred"] if isinstance(outputs, dict) else outputs
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        loss = self.loss_fn(pred.float(), targets)
        return loss, {"total": float(loss.detach().cpu().item())}


def build_criterion():
    """构建损失函数（根据 cfg.TASK 自动选择）"""
    if getattr(cfg, "TASK", "cd") == "scd":
        return SCDLoss()
    return ChangeLoss()


if __name__ == '__main__':
    """测试损失函数"""
    print("=" * 60)
    print("Testing Change Detection Loss (Binary Version)")
    print("=" * 60)

    # 虚拟数据
    B, H, W = 2, 256, 256

    # 模型输出（1通道）
    outputs = {
        'pred': torch.randn(B, 1, H, W),
        'aux_preds': [
            torch.randn(B, 1, H, W),
            torch.randn(B, 1, H, W),
            torch.randn(B, 1, H, W),
            torch.randn(B, 1, H, W)
        ]
    }

    # 二值标签
    targets = torch.randint(0, 2, (B, H, W))

    # 创建损失函数
    criterion = ChangeLoss()
    criterion.train()

    # 计算损失
    total_loss, loss_dict = criterion(outputs, targets)

    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"\nLoss Details:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("Loss function test passed!")
    print("=" * 60)
