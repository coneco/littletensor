import numpy as np
from numpy import ndarray

from littletensor.tensor import Tensor
from littletensor.edge import Edge
from littletensor.utils import isinstancelist, RESOURCES_KEY

class Op(object):

    def __init__(self, name: str, inputs: list, output_shapes: list):
        super().__init__()
        self.id = -1
        self.name = name

        assert isinstancelist(inputs, Tensor), "inputs 应当是一个 Tensor 列表"
        self.inputs = [tns for tns in inputs]

        # Op 对象初始化时创建输出的 Tensor 对象
        self.outputs = [Tensor("%s_%d" % (name, idx), shp) for idx, shp in enumerate(output_shapes)]
        # 并将 Tensor 的 op 指针指向 self
        for tns in self.outputs:
            tns.op = self
        

        self.in_edges = []

        for tns in self.inputs:
            # Op 对象初始化时创建输入 Edge 对象
            new_edge = Edge(tns.op, self, tns)
            self.in_edges.append(new_edge)
            # 并更新依赖 Op 的 out_edges
            tns.op.out_edges.append(new_edge)

        # 此时该 Op 还没有依赖项
        self.out_edges = []


    def compute(self, context: dict):
        # 该函数子类实现
        # context，dict，是一个计算图中计算完毕的 Tensor 名到其值的映射
        # 该函数应当创建输出 Tensor 的 value，放置于context，除了 Placeholder / Variable / Constant
        # 该函数禁止修改输入 Tensor，除了 Assign_add
        pass

# 该词典维护 OP 实现类到其梯度计算函数的映射
OP_TO_GRADIENT = {}

class Placeholder(Op):

    def __init__(self, name: str, shape: list):
        super().__init__(name, [], [shape])

    def compute(self, context: dict):
        assert self.outputs[0].name in context, "placeholder %s not feed." % (self.name, )

class Constant(Op):

    def __init__(self, name: str, value: ndarray):
        super().__init__(name, [], [list(value.shape)])
        self.value = value
    
    def compute(self, context: dict):
        context[self.outputs[0].name] = self.value

class Variable(Op):

    def __init__(self, name: str, init_value: ndarray):
        super().__init__(name, [], [list(init_value.shape)])
        self.init_value = init_value
    
    def compute(self, context: dict):
        assert RESOURCES_KEY in context
        assert isinstance(context[RESOURCES_KEY], dict)
        assert self.outputs[0].name in context[RESOURCES_KEY], "variable %s not initialized." % (self.name, )
        context[self.outputs[0].name] = context[RESOURCES_KEY][self.outputs[0].name]


class AssignAdd(Op):

    def __init__(self, name: str, var: Variable, value: Tensor):
        assert var.shape == value.shape
        super().__init__(name, [value], [var.shape])
        self.var = var
        self.in_edges[0].gradient = False

    def compute(self, context: dict):
        assert RESOURCES_KEY in context
        assert isinstance(context[RESOURCES_KEY], dict)
        assert self.var.name in context[RESOURCES_KEY], "variable %s not initialized." % (self.name, )
        context[RESOURCES_KEY][self.var.outputs[0].name] += context[self.inputs[0].name]
