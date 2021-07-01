import copy
import numpy as np

from littletensor.utils import RESOURCES_KEY


class Tensor(object):

    def __init__(self, name: str, shape: list, dtype: type = np.float32):
        super().__init__()
        assert name != RESOURCES_KEY, "%s 是为 context 预留的 key" % (RESOURCES_KEY, )
        self.id = -1
        self.name = name
        self.shape = copy.deepcopy(shape)
        self.dtype = dtype
        self.op = None  # 该指针恒指向输出该 tensor 的 op