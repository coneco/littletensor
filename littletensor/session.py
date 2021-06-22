import copy
from collections import deque

from littletensor.graph import Graph
from littletensor.tensor import Tensor
from littletensor.utils import RESOURCES_KEY

class Session(object):
    def __init__(self, graph: Graph):
        super().__init__()

        self.graph = graph
        self.sub_graphs = {}

        self.resources = {}

    def initialize(self):
        self.graph.initialize_all_variables(self.resources)

    def run(self, outputs: list, feed_dict: dict = {}) -> list:
        
        if not self.graph.finalized:
            self.sub_graphs = {}
            self.graph.finalize()

        graph = self.graph
        # TODO: 实现子图裁剪
        # query = tuple(sorted([tns.name for tns in outputs]))
        # if query not in self.sub_graphs:
        #     self.sub_graphs[query] = self.graph.prune_for(outputs)
        # graph = self.sub_graphs[query]

        # 计算各 Op 入度
        in_edge_count = {}
        for name, op in graph.ops.items():
            in_edge_count[name] = len(op.in_edges)

        context = {RESOURCES_KEY: self.resources}
        
        for tns, value in feed_dict.items():
            context[tns.name] = value

        # 拓扑排序计算各个 tensor 填入 context
        travel = deque()

        for name, op in graph.ops.items():
            if in_edge_count[name] == 0:
                travel.append(op)
        
        while len(travel):
            cur_op = travel.popleft()
            cur_op.compute(context)
            for edge in cur_op.out_edges:
                in_edge_count[edge.sink.name] -= 1
                if in_edge_count[edge.sink.name] == 0:
                    travel.append(edge.sink)
        
        return [context[tns.name] for tns in outputs]
