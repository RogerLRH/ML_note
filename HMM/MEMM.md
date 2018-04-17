# 最大熵马尔可夫模型 Maximum Entropy Markov Model

## 构成

>状态序列，马尔可夫链

>观察序列决定状态序列

>特征函数 $f_a(y_i, y_{i-1}, X, i)$，表示观察序列 $X$ 和状态 $y_i, y_{i-1}$ 是否满足某一条件 $a$，为零一函数
>$$P(y_i | y_{i-1}, X) = \frac {exp(\sum_a {\lambda_a f_a(y_i, y_{i-1}, X, i)})} {Z(y_{i-1}, X)}$$

## 问题

>标注偏置问题：局部归一化
