import numpy as np
from numpy import ndarray

from littletensor.tensor import Tensor
from littletensor.graph import Graph
from littletensor.op import Op, OP_TO_GRADIENT
from littletensor.utils import isinstancelist

class Sum(Op):

    def __init__(self, name: str, inputs: list):
        assert len(inputs) >= 1, "Sum 至少有一个输入"
        for inp in inputs:
            assert inp.shape == inputs[0].shape, "Sum 的输入 shape 必须一致"
        super().__init__(name, inputs, [inputs[0].shape])

    def compute(self, context: dict):
        output_matrix = np.zeros(self.outputs[0].shape)
        for tns in self.inputs:
            output_matrix += context[tns.name]
        context[self.outputs[0].name] = output_matrix

def sum_gradient(graph: Graph, op: Op, output_gradients: list):
    assert isinstancelist(output_gradients, Tensor)
    assert len(output_gradients) == len(op.inputs)

    return [output_gradients[0]] * len(op.inputs)

OP_TO_GRADIENT[Sum] = sum_gradient