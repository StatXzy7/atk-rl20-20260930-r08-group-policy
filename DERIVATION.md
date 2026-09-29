# GRPO 风格目标推导与变长归约审计

## 1. 数据与冻结策略

对 prompt `p_g` 采样 `G_g >= 2` 条 completion：

\[
y_{gi}=(y_{gi,1},\ldots,y_{gi,L_{gi}}),\qquad r_{gi}\in\mathbb R.
\]

奖励只用于构造优势，不进入计算图。`old policy` `π_old` 与 `reference policy` `π_ref` 都冻结；只有当前策略 `π_θ` 接收梯度。

组优势使用组内均值和总体标准差（`unbiased=False`，除以 `G_g`）：

\[
\bar r_g=\frac1{G_g}\sum_{i=1}^{G_g}r_{gi},
\qquad
\sigma_g=\sqrt{\frac1{G_g}\sum_{i=1}^{G_g}(r_{gi}-\bar r_g)^2},
\]

\[
A_{gi}=
\begin{cases}
\dfrac{r_{gi}-\bar r_g}{\sigma_g}, & \sigma_g>0,\\[6pt]
0, & \sigma_g=0.
\end{cases}
\]

`σ_g == 0` 时不走 epsilon 保护，也不产生除零；该组所有 token 的策略项严格为零。

## 2. PPO 式 clipped 策略项

对每个由 old policy 采样得到的 response token `y_{gi,t}`，定义当前策略与 old policy 的概率比率：

\[
\rho_{gi,t}(\theta)=
\frac{\pi_\theta(y_{gi,t}\mid p_g,y_{gi,<t})}
     {\pi_{\mathrm{old}}(y_{gi,t}\mid p_g,y_{gi,<t})}
=
\exp\left(
\ell_\theta(y_{gi,t}\mid x_{gi,t})-
\ell_{\mathrm{old}}(y_{gi,t}\mid x_{gi,t})
\right).
\]

序列级优势广播给该序列的所有有效 response token：

\[
\ell^{\mathrm{policy}}_{gi,t}
=
\min\left(
\rho_{gi,t}A_{gi},
\operatorname{clip}(\rho_{gi,t},1-\epsilon,1+\epsilon)A_{gi}
\right).
\]

prompt token、EOS 之后的 PAD 和右侧 padding 都不进入掩码。EOS 本身是已采样 response token，因此计入目标。

## 3. 两种不同归约

### 3.1 序列等权

先在每条序列内对有效 response token 平均，再对所有已采样序列平均：

\[
J^{\mathrm{seq}}_{\mathrm{policy}}
=
\frac1N\sum_{i=1}^{N}
\frac1{L_i}\sum_{t=1}^{L_i}\ell^{\mathrm{policy}}_{i,t}.
\]

记每条序列的序列均值为 `P_i`，则每条序列权重都是 `1/N`，与其长度无关。

### 3.2 token 等权

把所有有效 response token 汇在一起平均：

\[
J^{\mathrm{tok}}_{\mathrm{policy}}
=
\frac{\sum_{i=1}^{N}\sum_{t=1}^{L_i}\ell^{\mathrm{policy}}_{i,t}}
     {\sum_{i=1}^{N}L_i}
=
\sum_{i=1}^{N}\frac{L_i}{L_{\mathrm{tot}}}P_i.
\]

长度为 `L_i` 的序列权重是 `L_i/L_tot`。

### 3.3 为什么不是数值实现差别

`\bar L=L_tot/N` 时：

\[
J^{\mathrm{tok}}_{\mathrm{policy}}-J^{\mathrm{seq}}_{\mathrm{policy}}
=
\frac{\operatorname{Cov}_i(L_i,P_i)}{\bar L}.
\]

因此只有在所有序列等长，或长度与序列均值没有协方差时，二者才偶然相同。长度复制会直接改变序列权重：

- 两条序列长度都是 1，序列权重分别为 `1/2, 1/2`，token 权重也是 `1/2, 1/2`。
- 若第二条复制为两个相同 response token，序列权重仍是 `1/2, 1/2`，但 token 权重变成 `1/3, 2/3`。

这定义了不同的经验风险：序列等权把每条回答当一个等权样本；token 等权让长回答在批处理中拥有更大话语权。后者还会改变旧前缀分布的经验测度。

当前实现明确支持不同组大小，但在序列等权模式下，每条已采样 completion 等权；因此更大的组会自然拥有更多序列项。

## 4. 显式条件 KL

不使用 `(\log π_θ-\log π_ref)` 在单个采样 token 上的期望作为“精确 KL”。在每个已采样前缀 `x` 上，对小词表中的全部合法 token `v` 显式求和：

\[
\operatorname{KL}_x(\pi_\theta\|\pi_{\mathrm{ref}})
=
\sum_{v\in\mathcal V_{\mathrm{legal}}}
\pi_\theta(v\mid x)
\log
\frac{\pi_\theta(v\mid x)}{\pi_{\mathrm{ref}}(v\mid x)}.
\]

PAD 不属于合法词表：其 logit 为 `-inf`、概率为 0，不参与 KL。所有参与计算的合法 logits 都是有限值，因此合法概率严格为正；代码还会检查这一条件。

KL 是独立正则项，采用与策略项相同的归约：

\[
J^{\mathrm{seq}}=J^{\mathrm{seq}}_{\mathrm{policy}}
-\beta K^{\mathrm{seq}},
\qquad
J^{\mathrm{tok}}=J^{\mathrm{tok}}_{\mathrm{policy}}
-\beta K^{\mathrm{tok}}.
\]

优化器最小化 `loss=-J`。

## 5. 条件 KL 与整条新轨迹 KL 的区别

当前实现中的 KL 是

\[
\mathbb E_{x\sim\pi_{\mathrm{old}}}
\left[
\operatorname{KL}\bigl(\pi_\theta(\cdot\mid x)\Vert\pi_{\mathrm{ref}}(\cdot\mid x)\bigr)
\right]
\]

的 old-policy 采样前缀蒙特卡洛估计。它在每个固定前缀上对条件分布做精确词表求和，但前缀来自 `π_old`。

整条新策略 completion 轨迹的 KL 是

\[
\operatorname{KL}\bigl(\pi_\theta(Y\mid p)\Vert\pi_{\mathrm{ref}}(Y\mid p)\bigr)
=
\mathbb E_{Y\sim\pi_\theta}
\left[
\sum_t
\log\frac{\pi_\theta(y_t\mid p,y_{<t})}
          {\pi_{\mathrm{ref}}(y_t\mid p,y_{<t})}
\right],
\]

其中包含 EOS 所诱导的长度分布。二者不等价：

- 前缀分布不同：当前正则只访问 old policy 已访问到的状态；全轨迹 KL 访问当前策略自己产生的新状态。
- 当前策略改变一步 token 概率后，后续前缀分布也会改变；前缀固定条件 KL 不追踪这种状态分布迁移。
- 全轨迹 KL 对新轨迹和新长度分布求期望；当前实现是旧轨迹样本上的局部、逐步惩罚。
- token 等权归约还会按旧采样长度加权；序列等权归约则先平均每条旧轨迹的前缀 KL。

因此，这里的术语应理解为“old-policy 采样前缀上的显式条件 KL 正则”，不能声称它等于整条新轨迹的精确 KL。
