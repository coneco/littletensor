from collections import deque
import numpy as np

from littletensor.graph import Graph
from littletensor.op import Constant, OP_TO_GRADIENT
from littletensor.tensor import Tensor
from littletensor.math_ops import Sum

def compute_gradient(graph: Graph, y: Tensor, x: list) -> list:
    """
        计算 dy / dx
        返回一个与 x 长度相同的列表，对应 x 中每个 tensor 的 gradient tensor
        计算过程中加入的 op 会自动加入 graph 内
        原则上要求 y 是一个标量，否则退化为计算 reduce_sum(y) 到 x 的梯度
    """

    assert y in graph.ops
    for tns in x:
        assert tns in graph.ops, "tensor %s 不在图内" % (tns.name,) 

    # 所有的 tensor 与其对应的 gradient_tensor 的映射
    all_gradients = {}
    
    # 在这个 tensor 输入的所有 Op 计算完毕前，在这里维护一个列表保存已经计算的梯度
    # 计算完毕后通过 Sum Op 转入 all_gradients
    inner_gradients = {}

    # 添加一个值为 1 的 constant 代表 y 的 梯度
    all_gradients[y.name] = graph.make_op(Constant, "gradient_%s" % (y.name,), np.ones(y.shape))

    # 每个 op 带 gradient 出边的数量
    gradient_counts = {}
    for name, op in graph.ops:
        gradient_counts[name] = len([edge for edge in op.out_edges if edge.gradient])
    
    # 强设 y 为初始遍历节点
    gradient_counts[y.op.name] = 0
    # 反向拓扑排序
    # 此处要求 x 不得产生 y 之外的梯度依赖
    travel = deque()
    travel.append(y.op)
    while len(travel):
        cur_op = travel.popleft()
        if cur_op.__class__ not in OP_TO_GRADIENT:
            continue
        cur_output_grad = [all_gradients[tns.name] for tns in cur_op.outputs]
        cur_input_grad = OP_TO_GRADIENT[cur_op.__class__](graph, cur_op, cur_output_grad)
        for idx, tns in enumerate(cur_op.inputs):
            if tns.name not in inner_gradients:
                inner_gradients[tns.name] = []
            inner_gradients[tns.name].append(cur_input_grad[idx])
        for edge in cur_op.in_edges:
            if not edge.gradient:
                continue
            gradient_counts[edge.source.name] -= 1
            if gradient_counts[edge.source.name] == 0:
                for tns in edge.source.outputs:
                    if tns.name not in inner_gradients or len(inner_gradients[tns.name]) == 0:
                        all_gradients[tns.name] = graph.make_op(Constant, "gradient_%s" % (tns.name,), np.zeros(tns.shape))
                    elif len(inner_gradients[tns.name]) == 1:
                        all_gradients[tns.name] = inner_gradients[tns.name][0]
                    all_gradients[tns.name] = graph.make_op(Sum, "gradient_%s" % (tns.name,), all_gradients[tns.name])
                travel.append(edge.source)
    
    return [all_gradients[tns.name] for tns in x]
