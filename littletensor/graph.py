from collections import deque

from littletensor.op import Op, Variable


class Graph(object):

    def __init__(self):
        super().__init__()
        self.ops = {}
        self.topo = []
        self.tensors = {}
        self.variables = {}

        self.finalized = False

    def finalize(self):
        self.finalized = True

    def insert_op(self, op: Op) -> tuple:
        assert not self.finalized
        assert op.name not in self.ops
        self.ops[op.name] = op
        self.topo.append(op)

        for tns in op.outputs:
            assert tns.name not in self.tensors
            self.tensors[tns.name] = tns

        if isinstance(op, Variable):
            self.variables[op.name] = op
        return tuple(op.outputs)

    def make_op(self, op_cls: type, *args, **kwargs) -> tuple:
        assert not self.finalized
        op = op_cls(*args, **kwargs)
        return self.insert_op(op)

    def make_tensor(self, op_cls: type, *args, **kwargs) -> tuple:
        return self.make_op(op_cls, *args, **kwargs)

    def initialize_all_variables(self, resources: dict):
        for var in self.variables.values():
            resources[var.outputs[0].name] = var.init_value.copy()

    def prune_for(self, outputs: list) -> list:
        # TODO: 实现子图裁剪
        # 返回一个子图的拓扑序
        subgraph = set()
        bfs = deque()

        for tns in outputs:
            bfs.append(tns.op)

        while len(bfs):
            cur = bfs.popleft()
            subgraph.add(cur.name)
            for tns in cur.inputs:
                if tns.op.name not in subgraph:
                    bfs.append(tns.op)

        return [op for op in self.topo if op.name in subgraph]
