class Edge(object):
    
    def __init__(self, source, sink, tensor):
        super().__init__()
        self.id = -1
        self.source = source
        self.sink = sink

        # Edge 与 Tensor 是多对一的关系，对应某个 op 的输出输入了多个 op
        self.tensor = tensor

        # 标记这条边是否参与梯度计算
        self.gradient = True