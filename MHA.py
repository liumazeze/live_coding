import torch
import torch.nn as nn
import torch.nn.functional as F


def prepare_4d_causal_padding_mask(input_ids, pad_token_id):
    """
    根据 input_ids 生成相应的的 causal mask 和 padding mask
    """
    batch_size, seq_len = input_ids.shape

    # Step 1: 生成 Causal Mask (seq_len, seq_len)，保证因果性
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool)
    )  # 下三角矩阵

    # Step 2: 生成 Padding Mask (batch_size, seq_len)
    padding_mask = (
        input_ids != pad_token_id
    )  # (batch_size, seq_len)，填充 token 位置为 0

    # Step 3: 让 Padding Mask 变成 (batch_size, seq_len, seq_len)
    padding_mask = padding_mask.unsqueeze(1)  # 变成 (batch_size, 1, seq_len)
    padding_mask = padding_mask & padding_mask.transpose(
        1, 2
    )  # (batch_size, seq_len, seq_len)

    # Step 4: 结合因果 Mask 和 Padding Mask，确保填充 token 不能参与 attention
    final_mask = causal_mask.unsqueeze(0) & padding_mask

    return final_mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_states, num_heads):
        super().__init__()
        self.hidden_states = hidden_states
        self.num_heads = num_heads
        self.d_k = hidden_states // num_heads

        self.W_q = nn.Linear(hidden_states, hidden_states)
        self.W_k = nn.Linear(hidden_states, hidden_states)
        self.W_v = nn.Linear(hidden_states, hidden_states)
        self.W_o = nn.Linear(hidden_states, hidden_states)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # 线性投影并分头
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        print(f"scores: \n{scores}\n")

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        # 合并头并输出
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_states)
        return self.W_o(out)


if __name__ == "__main__":
    # 示例数据
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4],  # 句子1
            [6, 5, 0, 0],  # 句子2（右填充，pad_token_id=0）
        ]
    )
    pad_token_id = 0

    BATCH_SIZE = 2
    SEQ_LEN = 4
    HIDDEN_STATES = 32
    NUM_HEADS = 4

    mha = MultiHeadAttention(HIDDEN_STATES, NUM_HEADS)
    mask = prepare_4d_causal_padding_mask(
        input_ids, pad_token_id
    ).int()  # 转换为 int 方便查看

    print(f"mask: \n{mask}\n")

    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_STATES)
    output = mha.forward(x, mask)
    print(f"output shape: {output.shape}")
