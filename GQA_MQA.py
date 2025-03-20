import torch
import torch.nn as nn


class GroupQueryAttention(nn.Module):
    """
    num_key_value_heads 为 1 时，就是 MQA (Multi-Query Attention)
    """

    def __init__(
        self, hidden_size, num_attention_heads, num_key_value_heads, dropout=0.0
    ):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        assert num_attention_heads % num_key_value_heads == 0

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.dropout = dropout
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()

        query_states = (
            self.q_proj(hidden_states)
            .view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        # The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        key_states = torch.repeat_interleave(
            input=key_states, dim=1, repeats=self.num_key_value_groups
        )
        value_states = torch.repeat_interleave(
            input=value_states, dim=1, repeats=self.num_key_value_groups
        )

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(-1, -2))
            * self.head_dim**-0.5
        )
        if attention_mask is not None:
            # TODO 待完善，暂时忽略 mask
            pass

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1).contiguous()

        attn_output = self.o_proj(attn_output)

        return attn_output


if __name__ == "__main__":
    BATCH_SIZE = 6
    SEQ_LEN = 8
    HIDDEN_SIZE = 128
    NUM_ATTENTION_HEADS = 8
    NUM_KEY_VALUE_HEADS = 4

    x = torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    gqa = GroupQueryAttention(HIDDEN_SIZE, NUM_ATTENTION_HEADS, NUM_KEY_VALUE_HEADS)
    gqa.forward(x, None)

    print(f"x.shape is : {x.shape}")
