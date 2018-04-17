# 线性回归

## 公式

$$y(x,w) = w_0 + \sum_{j=1}^m {w_j \phi_j(x)}$$

>$\phi_j$ 是基函数，对特征进行转换。$m$ 是特征数。$w_0$ 是偏置系数。

>线性，是指每个特征对应一个参数

## 基函数

#### 多项式

$$\phi(x) = \sum_{j=0}^n a_j x^j$$

>局限性：通常只能拟合局部，造成过拟合

>解决：将输入划分不同子空间，分别拟合

#### 高斯

$$\phi(x) = exp(- \frac {(x - \mu)^2} {2s^2})$$

>由于参数 $w$ 的存在，不需要标准化

#### sigmoid

$$\phi(x) = \sigma(\frac {x - \mu} {s})$$

>归一化特征

#### TODO: 傅立叶

## 损失函数

#### 最小二乘

$$E_D(W) = \frac12 \sum_{i=1}^n (t_i - W^T \Phi(x_i))^2$$

>当目标值加上高斯噪音时，最小二乘等价于最大似然
$$\begin{align} lnp(T | X, W, \beta) &= \sum_{i=1}^n ln \mathcal {N} (t_i | W^T \Phi(x_i), \beta^{-1}) \\&= \frac {n} 2 ln \beta - \frac {n} 2 ln(2 \pi) - \beta E_D(W) \end{align}$$

#### 正则化

>L1：Lasso回归

>L2：Ridge回归

>L1+L2： Elastic Net回归

#### 局部加权平均

$$E_D(W) = \frac12 \sum_{i=1}^n p_i(t_i - W^T \Phi(x_i))^2$$

>通常，$p_i = exp(- \frac {(x - x_i)^2} 2)$

>非参数方法，每次预测都要重新计算，不适用于大样本

## 优点

>实现简单，计算简单

## 缺点

>分类精度不高

>容易过拟合

# 逻辑回归

## 构成

>本质上是线性回归

>在 sigmoid 函数 $\sigma(x) = \frac 1 {1 + exp(-x)}$ 下转化为概率

>多分类用 softmax 函数 $\sigma(x_i) = \frac {exp(x_i)} {\sum_j exp(x_j)}$
