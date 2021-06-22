import numpy as np
import littletensor as lt

g = lt.Graph()

a, = g.make_tensor(lt.Constant, "a", np.ones([2,2]))       # make tensor 会返回列表，所以要加逗号
b, = g.make_tensor(lt.Constant, "b", np.ones([2,2]) * 2)

c, = g.make_tensor(lt.Sum, "c", [a, b])

sess = lt.Session(g)
print(sess.run([c]))