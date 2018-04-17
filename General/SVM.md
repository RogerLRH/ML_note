# 支持向量机

## 线性可分

>以分界面将数据分成不同的类别，在线性可分的情况下，该分界面可以在一定范围内平移。这个可平移的间隔称为分类间隔。在仅平移的情况下，间隔中间的分界面是最佳的

>SVM 的目标是最大化分类间隔

>点到面的距离
>$$\frac{|w^T x + b|}{||w||_2}$$

>y 的取值令正例为1，反例为-1，则
>$$y_i (w^T x_i + b) = |w^T x_i + b|$$

>由于同时缩放参数 w 和 b 不影响分类间隔，约束 $y_i (w^T x_i + b) \geq 1$，这等于
>$$\max \frac 1 {||w||_2} = \min \frac12 ||w||_2^2$$

>引入拉格朗日乘子
>$$L(w, b, \alpha) = \frac12 ||w||_2^2 - \sum_{i=1}^n \alpha_i (y_i (w^T x_i + b) - 1)$$

>原问题转为（强对偶）
>$$\min_{w,b} \theta(w) = \min_{w,b} \max_{\alpha \geq 0} L(w, b, \alpha) = \max_{\alpha \geq 0} \min_{w, b} L(w, b, \alpha)$$

>解最小化，可推导出$w = \sum_{i=1}^n \alpha_i y_i x_i，\sum_{i=1}^n \alpha_i y_i = 0$，故分类函数为
>$$f(x) = \sum_{i=1}^n \alpha_i y_i \langle x_i, x \rangle + b$$
>显然，$\alpha_i = 0$ 时，数据点对结果无影响，而其它数据点则满足 $y(w^T x + b) = 1$，所以，这些在分类边界上的点被称为支持向量

>问题转化为
>$$\max_{\alpha} (\sum_{i=1}^n \alpha_i -\frac12 \sum_{i=1}^n \sum_{j=1}^n y_i y_j \langle x_i, x_j \rangle \alpha_i \alpha_j)$$
>$$s.t., \alpha_i \geq 0, i = 1, ..., n$$
>$$\sum_{i=1}^n \alpha_i y_i = 0$$
>TODO: 利用 SMO 算法求解，可得 $\alpha$

## 线性不可分

>实际问题中，往往线性不可分

#### 核函数

>$$f(x) = w^T \phi(x) + b$$
>通过空间变换，使得数据在新空间线性可分
>$$f(x) = \sum_{i=1}^n \alpha_i y_i \langle \phi(x_i), \phi(x) \rangle + b$$

>内积的值是标量，因此，变换后的内积等价于某种到实数的映射，这样，不需要显式地构造变换，这样的映射被称为核函数
>
>这样避免变换到高维空间的计算困难的问题

>常用核函数:
>>多项式核：$k(x_1, x_2) = (\langle x_1, x_2 \rangle + R)^d$
>>
>>高斯核：$k(x_1, x_2) = exp(- \frac {||x_1 - x_2||^2} {2 \sigma^2})$
>>
>>线性核：即无变换，是形式的统一

#### 松弛变量

>允许部分误分类，约束改为 $y (w^T x_i + b) \geq 1 - \xi_i$

>目标函数改为 $\frac12 ||w||^2 + C \sum_{i=1}^n \xi_i$

>SMO 解决的问题增加限制 $\alpha_i \leq C$

## 优点

>解决小样本下机器学习问题

>解决非线性问题

>无局部极小值问题

>可以很好的处理高维数据集

>泛化能力比较强

## 缺点

>对于核函数的高维映射解释力不强，尤其是径向基函数

>对缺失数据敏感
