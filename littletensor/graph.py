from littletensor.tensor import Tensor
from littletensor.op import Op, Variable


class Graph(object):

    def __init__(self):
        super().__init__()
        self.ops = {}
        self.tensors = {}
        self.variables = {}
        self.resources = {}

        self.in_edge_count = {}
        self.finalized = False

    def finalize(self):
        self.finalized = True

    def insert_op(self, op: Op) -> tuple:
        assert not self.finalized
        assert op.name not in self.ops
        self.ops[op.name] = op

        for tns in op.outputs:
            assert tns.name not in self.tensors
            self.tensors[tns.name] = tns

        if isinstance(op, Variable):
            self.variables[op.name] = op
        return tuple(op.outputs)

    def make_op(self, op_cls: type, *args, **kwargs):
        assert not self.finalized
        op = op_cls(*args, **kwargs)
        return self.insert_op(op)

    def make_tensor(self, op_cls: type, *args, **kwargs):
        return self.make_op(op_cls, *args, **kwargs)

    def initialize_all_variables(self, resources: dict):
        for var in self.variables.items():
            resources[var.outputs[0].name] = var.init_value.copy()

    def prune_for(self, outputs: list):
        # TODO: 实现子图裁剪
        # return a new graph optimized for outputs
        return self
