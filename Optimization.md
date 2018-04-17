# 优化

## 目标函数

$$J(\theta) = \mathbb{E}_{(x, y) {\sim} p_{data}} L(f(x, \theta), y)$$

>通常，我们优化的是训练集上的目标函数，但我们更希望真实数据分布上的期望，然而真实数据是不可知的，故最小化训练集上的期望损失，称为经验风险最小化

## 类型

>批量或确定性算法：单次使用整个训练集

>随机或在线算法：单次只使用一个样本

>介于前两者之间，单次使用固定个数样本

#### 小批量因素

>均值标准差 ${\sigma}/{\sqrt{n}}$，故更大的批量会计算更精确的梯度估计，但收益是小于线性的

>过小的批量难以充分利用多核架构，不会减少计算时间，故批量应该足够大

>批量大小和内存占用成正比

>使用 GPU 时，使用 2 的幂数可以获得更少的运行时间

>小批量有正则化的效果，批量越小，正则化效果越强

>由于批量越小，估计方差越大，需要更小的学习率，故总的运算量更大

>理论上，小批量应该是随机抽取的，但实践中，通常将样本打乱一遍然后多次遍历，故第一次遍历的估计是无偏的，之后是有偏的

## 挑战

#### 病态

>随机梯度下降卡住，很小的步长也会增加代价函数

>很多情况下，梯度范数不会在训练中显著缩小，但 $g^THg$ 的增长会超过一个数量级，故尽管梯度很强，学习也会很缓慢，因为学习率必须收缩以弥补更强的曲率

#### 局部极小值

>通过相互交换潜变量可以得到等价的深度学习模型，故局部极小值非常多，称为权重空间对称性

>局部极小值的代价远比全局最小点的大，可能造成神经网络的训练困难

#### 高原、鞍点和其它平坦区域

>在高维空间，相对于极小点，鞍点的数量更多

>实验中，梯度下降总能逃离鞍点

>对牛顿法，有明显的问题，这可以解释二阶方法在神经网络训练中无法取代梯度下降

>可能存在恒值、宽广的区域，通常有较大的代价

#### 悬崖

>较大的权重相乘可能导致悬崖式的斜率较大区域，梯度更新很容易跳过这类区域

>梯度截断，可以避免这种情况

#### 长期依赖／梯度爆炸或消失

>当计算图较深时，由于梯度传播时的衰减或放大，使得上层难以学习。梯度消失使得代价函数难以改进，而梯度爆炸使得学习非常不稳定

#### 非精确梯度

>神经网络中梯度是难以处理的，而大多数优化算法需要精确的梯度或 Hessian 矩阵

## TODO: 参数初始化策略

## 算法

#### 基础

- 随机梯度下降

$$\theta \leftarrow \theta - \epsilon g(\theta)$$

>小批量+梯度+衰减的学习率

>常用的一种学习率，$\tau$ 为停止衰减的迭代次数，停止迭代时的学习率通常为开始时的 1%
>
>$$\epsilon_k = (1-\alpha) \epsilon_0 + \alpha \epsilon_{\tau}, \alpha = k / \tau$$

- 动量

$$\begin{align}
&v \leftarrow \alpha v - \epsilon g(\theta) \\ &\theta \leftarrow \theta + v
\end{align}$$

>加速学习

- Nesterov 动量

$$\begin{align}
&\widetilde {\theta} \leftarrow \theta + \alpha v \\
&v \leftarrow \alpha v - \epsilon g(\widetilde {\theta}) \\
&\theta \leftarrow \theta + v
\end{align}$$

#### 自适应学习率算法

- AdaGrad

$$\begin{align}
&r \leftarrow r + g(\theta) \odot g(\theta) \\
&\theta \leftarrow \theta - \frac {\epsilon} {\delta + \sqrt{r}} \odot g(\theta)
\end{align}$$

>对深度神经网络来说，梯度平方的累积导致了学习率过早和过量减小

- RMSProp

$$\begin{align}
&r \leftarrow \rho r + (1 - \rho) g(\theta) \odot g(\theta) \\
&\theta \leftarrow \theta - \frac {\epsilon} {\sqrt{\delta + r}} \odot g(\theta)
\end{align}$$

- Adam

$$\begin{align}
&t \leftarrow t + 1 \\
&s \leftarrow \rho_1 s + (1 - \rho_1) g(\theta) \\
&r \leftarrow \rho_2 r + (1 - \rho_2) g(\theta) \odot g(\theta) \\
&\dot{s} \leftarrow \frac {s} {1-\rho_1^t} \\
&\dot{r} \leftarrow \frac {r} {1-\rho_2^t}\\
&\theta \leftarrow \theta - \frac {\epsilon \dot {s}} {\sqrt {\delta + \dot {r}}} \odot g(\theta)
\end{align}$$

#### 二阶近似方法

- 牛顿法

$$\theta \leftarrow \theta - H^{-1} g(\theta)$$

>二阶收敛更快，对于二次函数更是一次到位

>牛顿法只适用于 Hessian 正定的情况，所以深度学习难以应用牛顿法

- 共轭梯度

>共轭梯度法确保梯度沿前一方向大小不变，且所有方向相互垂直，故在k维参数空间最多只需k次线搜索即可到达极小值

>应用于神经网络时使用非线性共轭梯度

- 牛顿近似法

>Broyden-Fletcher-Goldfard-Shanno 构造矩阵近似 $H$，不需要精确的线搜索

1. 初始化：初始点 $x_0$ 以及近似逆 Hessian 矩阵 $B_0^−1$。通常为单位矩阵。

2. 计算线搜索方向：

$$p_k = −B_k^{−1} \bigtriangledown f(x_k)$$

3. 用”Backtracking line search“算法沿搜索方向找到下一个迭代点

$$x_{k+1} = x_k + α_k p_k$$

4. 根据 Armijo–Goldstein 准则，判断是否停止

5. 计算 $y_k = \bigtriangledown f(x_{k+1}) − \bigtriangledown f(x_k)$

6. 迭代近似逆Hessian矩阵

$$B_{k+1}^{−1} = (I − \frac {s_k y_k^T} {y_k^T sk}) B_k^{-1} (I − \frac {y_k s_k^T} {y_k^T s_k}) + \frac {s_k s_k^T} {y_k^T s_k}$$

>L-BFGS 优化BFGS的存储问题

## 优化策略与元算法

#### 批标准化

>某一层的参数更新很大程度取决于其它层

>对每个隐含层的输入进行局部标准化，然后再激活，显然不使用于批量小的情形

>激活的参数需要学习

>预测时，使用训练时的平均期望和平均方差

#### 坐标下降

>反复循环地最优化单一变量

>块坐标下降：单一变量改为多个变量的组合

>适用于：不同变量可分成相对独立的组，或一组变量的优化明显比所有变量的优化的效率更高

>问题：一个变量很大程度上影响另一个变量的最优值，效果较差

#### Polyak 平均

>算法在参数空间的轨迹的平均值

>非凸问题时，使用衰减的方式 $\overline {\theta}_t = \alpha \overline {\theta}_{t-1} + (1 - \alpha) \theta_t$

#### 监督预训练

>逐层训练

>迁移学习：模型前k层直接用于其他任务，需要任务之间存在统计关系

#### TODO: 模型设计
#### TODO: 延拓法
