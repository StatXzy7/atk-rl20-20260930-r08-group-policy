# GRPO 风格变长 Completion 目标审计

这个最小实现不调用现成 GRPO 训练器，也不下载权重。它用一个词表大小为 5、隐藏维度为 4 的自回归 Elman RNN 固定 old policy 和 reference policy，采样变长 completion，随后用 PyTorch 完成一次显式更新并核验梯度。

## 文件

- `grpo_lab/grpo.py`：组优势、clipped surrogate、显式 categorical KL、两种归约。
- `grpo_lab/model.py`：微型自回归 RNN、冻结/克隆、old policy 变长采样。
- `run_experiment.py`：确定性采样和一次 SGD 更新，输出完整审计 JSON。
- `experiment_results.json`：本次实际运行结果。
- `tests/test_grpo.py`：14 个单元测试和双精度 `gradcheck`。

## 目标定义

对 prompt `p` 的组 `g`，固定采样 `G_p >= 2` 条 completion。序列标量奖励为 `r_i`，使用组内均值和总体标准差：

```text
μ_g = (1/G_g) Σ_i r_i
σ_g = sqrt((1/G_g) Σ_i (r_i - μ_g)^2)
A_i = (r_i - μ_g) / σ_g
```

当 `σ_g == 0` 时严格设置整组 `A_i = 0`，不使用 epsilon。奖励和优势都在 `torch.no_grad()` 下计算并 detach，不反向传播。

设 `s_{i,t}` 是 completion 的第 `t` 个 response token，`h_{i,t}` 是它前面的已采样 prefix。概率比率为：

```text
ρ_{i,t}(θ) = πθ(s_{i,t} | x_i, s_i,<t) / πold(s_{i,t} | x_i, s_i,<t)
ρ̄_{i,t} = clip(ρ_{i,t}, 1 - ε, 1 + ε)
L_policy^{token}(i,t) = min(A_i ρ_{i,t}, A_i ρ̄_{i,t})
```

序列优势是标量，并广播到该序列所有有效 response token。prompt token、EOS 之后的 padding token 都由 `response_mask` 排除；自回归对齐方式是 `logits[:, :-1]` 预测 `input_ids[:, 1:]`。

两种明确不同的归约为：

```text
J_seq  = (1/N) Σ_i (1/L_i) Σ_t L_policy^{token}(i,t)
J_tok  = (1/Σ_i L_i) Σ_i Σ_t L_policy^{token}(i,t)
```

这里 `N = Σ_p G_p`。因此 `sequence_mean` 中每条 completion/序列等权；不同 prompt 若 `G_p` 不同，会按其采样序列数量自然占比。`token_mean` 中每个 response token 等权。若还要求 prompt 等权，需要再乘显式的 per-prompt 权重；本实现没有偷偷加入这种权重。

最终最大化目标和最小化损失分别是：

```text
J = J_policy - β J_KL
loss = -J
```

`β=0` 时 `kl_penalty` 严格为零，但 policy 项仍保留。

## 为什么长度变化使两种归约不等价

`J_seq` 先把每条序列内部的 token 项平均为序列项，再对序列平均。于是长度复制不会让一条序列在全局目标中获得更多票数。`J_tok` 直接把 token 项汇总后除以总 token 数，长序列天然贡献更多项。

手算例子：序列 A 长度为 2，两个 token 的项都是 `1.1`；序列 B 长度为 1，项为 `1.0`。

```text
sequence_mean = ((1.1 + 1.1)/2 + 1.0)/2 = 1.05
token_mean    = (1.1 + 1.1 + 1.0)/3     = 16/15 ≈ 1.0666667
```

如果把 B 复制成同样的 token，使其长度也为 2，则两者都是 `1.05`；如果把 A 再复制到长度 4，token 平均会变成 `(4·1.1 + 1.0)/5 = 1.08`，而序列平均仍为 `1.05`。这不是求和顺序造成的浮点差异，而是优化的统计单位不同：一个是“随机抽一条序列”的期望，另一个是“随机抽一个 response token”的长度偏置期望。变长 completion 还会影响梯度规模、KL 权重以及长短序列之间的相对优化压力。

## 显式 KL 正则项

在每个实际访问到的 response prefix 上，对小词表的所有合法动作显式计算：

```text
D_KL(h) = Σ_{v ∈ V_legal(h)} πθ(v | h) [log πθ(v | h) - log πref(v | h)]
```

然后只在有效 response prefix 上应用与 policy 项相同的 `sequence_mean` 或 `token_mean`。这不是 `log πref/πθ` 近似，也不是只在采样 token 上的单样本估计。

合法动作通过 `legal_vocab_mask` 支持；PAD 在实验中不是合法采样动作。非法 logits 被屏蔽，所有参与归一化和 KL 的合法概率都必须严格为正，代码会检查：

- 每个分布至少有一个合法动作；
- 合法 logits 和合法 log-probabilities 有限；
- 所有合法概率 `> 0`；
- 每个被采样 response token 的比率有限且为正。

### 采样前缀 KL 与整条新轨迹 KL 的区别

当前正则项使用的前缀来自冻结的 old policy：

```text
h ~ πold(· | prompt)
D_current(h) = KL(πθ(· | h) || πref(· | h))
```

它回答的是：在已经由 old policy 访问到的状态上，新策略下一步分布离 reference 多远。它是一次更新中可直接计算的经验状态正则项。

整条新策略轨迹的 KL 还包含状态分布差异。若新策略改变了 token 概率，未来 prefix 的访问频率也改变：

```text
KL( πθ(sequence|prompt) || πref(sequence|prompt) )
= E_{s ~ πθ}[ Σ_t log πθ(s_t|h_t) - log πref(s_t|h_t) ]
```

这个期望里的 `h_t` 由 `πθ` 自己生成；本实现不会从更新后的 `πθ` 重新采样轨迹，因此没有声称自己精确计算了整条新轨迹 KL。当 `πold = πref` 时，首次更新在 `θ=θold` 处的前缀状态分布与新轨迹状态分布相同，但一旦 θ 偏离或继续多步更新，两者一般不再相同。严格轨迹 KL 通常需要从当前策略重新采样、使用路径空间估计，或做额外分解。

## 微型模型与采样

`TinyAutoregressiveRNN` 为：

```text
h_t = tanh(W_x E[x_t] + W_h h_{t-1} + b_x)
logits_t = W_o h_t + b_o
```

采样时：

- old policy 冻结并调用 `eval()`；
- reference policy 从 old policy 深拷贝后冻结；
- 使用固定 `torch.Generator` 采样；
- EOS 作为一个有效 response token 计入该次采样，EOS 后只写 PAD；
- 每条 completion 至少有一个 response token；
- 更新后的 policy 从同一个 perturbed 初始点分别用两种归约更新，便于公平比较梯度。

`clip_epsilon=1.0` 仅作为测试中的“无截断”精确手算边界；实验和默认目标使用 PPO 式 `0.2`。

## 实际更新结果

运行命令：

```bash
python run_experiment.py > experiment_results.json
python -m unittest discover -s tests -v
```

本次实际实验配置：词表 5、隐藏维度 4、两个 prompt，组大小 `[2, 3]`，最大新 token 数 6，`β=0.04`，`ε=0.2`，SGD 学习率 0.2。

采样长度为 `[2, 3, 3, 5, 6]`，奖励为：

```text
[0.75, -0.75, -0.75, 0.0, -0.25]
```

组优势为：

```text
[1.0, -1.0, -1.3363062, 1.0690451, 0.2672613]
```

19 个有效 response token 中，实验同时触发了 clip 的两侧分支：正优势上侧 6 个 token，负优势下侧 1 个 token。两种归约的一次更新审计结果：

| 归约 | policy objective | β·KL | loss |
| --- | ---: | ---: | ---: |
| `sequence_mean` | -0.4980615973 | 0.0143228564 | 0.5123844743 |
| `token_mean` | -0.3418626785 | 0.0143832481 | 0.3562459349 |

部分参数梯度 L2 范数也不同，例如 `output_layer.weight`：

```text
sequence_mean: 0.2384574
token_mean:    0.2150176
```

完整逐参数梯度、最大绝对梯度、更新前后参数范数变化和原始 batch 均保存在 `experiment_results.json`。

## 覆盖情况

测试显式覆盖：

- 零奖励方差时优势严格为零，端到端 `β=0` 梯度为零；
- `G=1` 拒绝，`G=[2,3]` 支持；
- 极不均匀长度和两种归约的手算相对权重；
- 正优势上侧 clip、负优势下侧 clip；
- 完全相同新旧/reference 策略时 ratio 为 1、无 clip、KL 为 0；
- `β=0` 移除 KL 项但保留 policy 梯度；
- prompt 和 padding 不进入目标；
- 所有合法概率严格为正，显式 KL 非负；
- 两种归约在 `float64` 下均通过 `torch.autograd.gradcheck`；
- old/reference 冻结，微型 RNN 完成一次反向更新且梯度有限。
