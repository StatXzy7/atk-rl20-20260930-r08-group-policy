# 微型 GRPO 风格策略优化

这个仓库实现一个不依赖现成 GRPO 训练器、不下载权重的微型自回归策略更新实验。

## 内容

- `grpo_mini/grpo.py`：组优势、PPO clipped surrogate、全合法词表显式条件 KL、两种归约。
- `grpo_mini/model.py`：小型因果 Transformer；PAD logit 为 `-inf`，所有合法词元概率为正。
- `grpo_mini/sampling.py`：从冻结 old policy 采样可变长 completion，EOS 后补 PAD。
- `scripts/micro_update.py`：固定随机种子的一次真实 SGD 更新、长度审计和有限差分梯度核验。
- `tests/test_grpo.py`：边界条件、手算例子和自回归模型集成测试。
- `DERIVATION.md`：目标推导、长度归约等价性分析、条件 KL 与全轨迹 KL 区别。
- `RESULTS.md`：实际测试和更新数值。

## 运行

```bash
pytest -q
python -m scripts.micro_update
```

## 目标要点

- 每个组 `G>=2`；优势使用组内均值和总体标准差，标准差为零时严格置零。
- old policy、reference policy、奖励和优势不接收梯度。
- 只对有效 response token 计概率比率、clip 项和 KL；prompt 与 PAD 被掩码。
- `reduction="sequence"` 先做序列内平均，再让每条 completion 等权。
- `reduction="token"` 对所有有效 response token 等权，长序列自动获得更大权重。
- KL 在每个 old-policy 采样前缀上对全部合法词元显式求和；它不是整条新策略轨迹 KL。
