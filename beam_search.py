import math


def beam_search(data, beam_size=2):
    """
    有一个序列 data(seq_len, v)
    seq_len: 输出长度
    vocab_size: 词典大小
    beam_size: 束搜索的范围
    """
    # 初始化：空序列，初始得分为0（使用log概率）
    sequences = [(0.0, [])]

    for step, probs in enumerate(data):
        candidates = []
        # 扩展当前所有候选
        for prev_score, seq in sequences:
            # 遍历当前步骤所有可能的token
            for token_id, token_prob in enumerate(probs):
                # 计算新得分（log概率相加）
                new_score = prev_score + math.log(token_prob)
                # 创建新序列
                new_seq = seq + [token_id]
                candidates.append((new_score, new_seq))

        # 按得分排序，保留前beam_size个
        candidates.sort(reverse=True, key=lambda x: x[0])
        sequences = candidates[:beam_size]

        # 打印每步结果（调试用）
        print(f"Step {step + 1}候选：")
        for score, seq in sequences:
            print(f"  得分: {score:.3f}, 序列: {seq}")

    return sequences


if __name__ == "__main__":
    data = [
        [0.7, 0.2, 0.1, 0.6, 0.9],
        [0.6, 0.3, 0.1, 0.05, 0.3],
        [0.1, 0.2, 0.7, 0.05, 0.05],
        [0.05, 0.05, 0.05, 0.7, 0.05],
    ]

    beam_size = 2

    final_sequences = beam_search(data, beam_size)

    print("\n最终结果：")
    for score, seq in final_sequences:
        print(f"得分: {score:.3f}, 序列: {seq}")
