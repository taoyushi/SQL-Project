# utils/classifier_loss_multi_gpu.py
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input_tensor, target_tensor):
        assert input_tensor.shape[0] == target_tensor.shape[0]
        
        # 计算交叉熵损失
        ce_loss = nn.functional.cross_entropy(input_tensor, target_tensor, reduction='none')
        
        # 计算pt
        pt = torch.exp(-ce_loss)
        
        # 计算alpha权重
        alpha_t = self.alpha * target_tensor + (1 - self.alpha) * (1 - target_tensor)
        
        # 计算focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

class ClassifierLossMultiGPU:
    def __init__(self, alpha=0.75, gamma=2.0):
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

    def compute_batch_loss_with_masks(self, batch_logits, batch_labels, batch_masks):
        """
        计算带掩码的批处理损失
        batch_logits: (batch_size, max_items, 2)
        batch_labels: list of tensors
        batch_masks: (batch_size, max_items)
        """
        total_loss = 0.0
        valid_samples = 0
        
        batch_size = batch_logits.size(0)
        for batch_id in range(batch_size):
            mask = batch_masks[batch_id]
            if mask.sum() == 0 or batch_id >= len(batch_labels) or batch_labels[batch_id] is None:
                continue
                
            num_items = int(mask.sum().item())
            logits = batch_logits[batch_id, :num_items, :]
            labels = batch_labels[batch_id]
            
            if logits.size(0) != labels.size(0):
                # 进一步处理大小不匹配
                min_size = min(logits.size(0), labels.size(0))
                logits = logits[:min_size]
                labels = labels[:min_size]
            
            if logits.size(0) > 0:
                loss = self.focal_loss(logits, labels)
                total_loss += loss
                valid_samples += 1
        
        return total_loss / max(valid_samples, 1)

    def compute_loss(self, model_outputs, batch_table_labels, batch_column_labels):
        """
        计算总损失
        """
        # 检查输出格式
        if "batch_table_logits" in model_outputs:
            # 新的多GPU兼容格式
            table_loss = self.compute_batch_loss_with_masks(
                model_outputs["batch_table_logits"],
                batch_table_labels,
                model_outputs["batch_table_masks"]
            )
            
            column_loss = self.compute_batch_loss_with_masks(
                model_outputs["batch_column_logits"],
                batch_column_labels,
                model_outputs["batch_column_masks"]
            )
        else:
            # 原始格式 - 回退到传统处理
            table_loss = self.compute_batch_loss_traditional(
                model_outputs["batch_table_name_cls_logits"],
                batch_table_labels
            )
            
            column_loss = self.compute_batch_loss_traditional(
                model_outputs["batch_column_info_cls_logits"],
                batch_column_labels
            )
        
        return table_loss + column_loss

    def compute_batch_loss_traditional(self, batch_logits_list, batch_labels):
        """
        传统的批处理损失计算（用于回退）
        """
        total_loss = 0.0
        valid_samples = 0
        
        for batch_id, logits in enumerate(batch_logits_list):
            if logits is None or batch_id >= len(batch_labels) or batch_labels[batch_id] is None:
                continue
                
            labels = batch_labels[batch_id]
            if logits.size(0) != labels.size(0):
                min_size = min(logits.size(0), labels.size(0))
                logits = logits[:min_size]
                labels = labels[:min_size]
            
            if logits.size(0) > 0:
                loss = self.focal_loss(logits, labels)
                total_loss += loss
                valid_samples += 1
        
        return total_loss / max(valid_samples, 1)