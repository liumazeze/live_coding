import torch
import torch.nn as nn
import torch.nn.functional as F


def custom_CrossEntropy(logits, targets, mask=None):
    """
    计算带有mask的自定义损失 (不使用 CrossEntropyLoss)
    :param logits: [batch_size, num_classes], 未归一化的logits
    :param targets: [batch_size], 真实类别索引
    :param mask: [batch_size], 值为0的位置表示该样本不计算损失
    :return: 标量损失值
    """
    # 计算 log-softmax
    log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, num_classes]

    # 选取目标类别的 log 预测概率
    targets_one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).float()
    loss = -torch.sum(log_probs * targets_one_hot, dim=-1)  # [batch_size]

    if mask is not None:
        # 仅保留 mask 不为 0 的损失值
        loss = loss * mask

        # 归一化，只对有效样本取均值，避免 loss 受 mask 影响变小
        return loss.sum() / mask.sum().clamp(min=1)

    return loss.sum() / len(loss)


if __name__ == "__main__":
    # 假设有 3 个类别，batch size 为 2
    # 预测值 (batch_size, num_classes)
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
    # 真实标签 (batch_size)
    labels = torch.tensor([0, 1])

    # nn.CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    print(criterion(logits, labels))  # tensor(0.3185)

    # 自定义 CrossEntropyLoss
    print(custom_CrossEntropy(logits, labels))  # tensor(0.3185)

    # 带有mask
    mask = torch.tensor([1, 0])
    print(custom_CrossEntropy(logits, labels, mask))
