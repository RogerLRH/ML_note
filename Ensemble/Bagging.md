# bagging

>bagging 也叫自举汇聚法（bootstrap aggregating），是一种在原始数据集上通过有放回抽样重新选出S个新数据集来训练分类器的集成技术。也就是说这些新数据集是允许重复的。

>使用训练出来的分类器集合来对新样本进行分类，然后用多数投票或者对输出求均值的方法统计所有分类器的分类结果，结果最高的类别即为最终标签。

>一般集成方法可以使用不同的算法和不同的目标函数训练成完全不同的模型，bagging 允许多次重复使用完全一样的模型