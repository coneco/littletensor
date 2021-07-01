# LittleTensor

简易版 TensorFlow，方便初学者理解原理

核心功能：计算图，自动梯度，自定义 Op

已实现的 Op：

```
Constant
Variable
Placeholder
Sum
AssignAdd
compute_gradient
```

## Requirements

python3
numpy

## Usage

``` python
import numpy as np
import littletensor as lt

g = lt.Graph()

a, = g.make_tensor(lt.Constant, "a", np.ones([2,2]))       # make tensor 会返回列表，所以要加逗号
b, = g.make_tensor(lt.Constant, "b", np.ones([2,2]) * 2)

c, = g.make_tensor(lt.Sum, "c", [a, b])

grad_a, = lt.compute_gradient(g, c, [a])

sess = lt.Session(g)
print(sess.run([grad_a]))
```

## TODO

1. 扩充 Op 库
2. 分布式计算