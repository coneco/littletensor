# LittleTensor

简易版 TensorFlow，方便初学者理解原理

核心功能：计算图，自动梯度，自定义 Op

已实现的 Op：

```
Constant
Variable (待测试)
Placeholder (待测试)
Sum
AssignAdd (待测试)
compute_gradient (待测试)
```

未来可能会有：分布式计算


## Requirements

numpy

## Usage

``` python
import numpy as np
import littletensor as lt

g = lt.Graph()

a, = g.make_tensor(Constant, "a", np.ones([2,2]))       # make tensor 会返回列表，所以要加逗号
b, = g.make_tensor(Constant, "b", np.ones([2,2] * 2))

c, = g.make_tensor(Sum, "c", [a, b])

sess = lt.Session(graph)
print(sess.run([c]))
```