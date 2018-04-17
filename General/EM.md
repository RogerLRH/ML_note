# EM算法 Expectation-maximization

>期望最大算法是一种从不完全数据或有数据丢失的数据集（存在隐含变量）中求解概率模型参数的最大似然估计方法

>假设我们想估计知道A和B两个参数，在开始状态下二者都是未知的，但如果知道了A的信息就可以得到B的信息，反过来知道了B也就得到了A。可以考虑首先赋予A某种初值，以此得到B的估计值，然后从B的当前值出发，重新估计A的取值，这个过程一直持续到收敛为止

$$\begin{align} \sum_i log \sum_{z_i} p(x_i, z_i, \theta) & = \sum_i log \sum_{z_i} Q_i(z_i) \frac {p(x_i, z_i, \theta)} {Q_i(z_i)} \\& \geq \sum_i \sum_{z_i} Q_i(z_i) log \frac {p(x_i, z_i, \theta)} {Q_i(z_i)} \end{align}$$

$$\sum_{z_i} Q_i(z_i) = 1$$

>当且仅当 $\frac {p(x_i, z_i, \theta)} {Q_i(z_i)} = c$ 时上式等号成立，即

$$Q_i(z_i) = \frac {p(x_i, z_i, \theta)} {\sum_z p(x_i, z, \theta)} = \frac {p(x_i, z_i, \theta)} {p(x_i, \theta)} = p(z_i | x_i, \theta)$$

## 过程

>E步骤：固定 $\theta$，
>$$Q_i(z_i) = p(z_i | x_i, \theta)$$

>M步骤：固定 $Q_i$，似然函数最大化
>$$\theta' = \arg \max_{\theta} \sum_i \sum_{z_i} Q_i(z_i) log \frac {p(x_i, z_i, \theta)} {Q_i(z_i)}$$

## 证明

$$\begin{align} l(\theta_{t+1}) &\geq \sum_i \sum_{z_i} Q_i^t(z_i) log \frac {p(x_i, z_i, \theta_{t+1})} {Q_i^t(z_i)} \\&\geq \sum_i \sum_{z_i} Q_i^t(z_i) log \frac {p(x_i, z_i, \theta_{t})} {Q_i^t(z_i)} \\&= l(\theta_t) \end{align}$$
