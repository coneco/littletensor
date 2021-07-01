from littletensor.op import AssignAdd
import numpy as np
import littletensor as lt

g = lt.Graph()

a, = g.make_tensor(lt.Variable, "a", np.ones([2,2], np.float32))       # make tensor 会返回列表，所以要加逗号
b, = g.make_tensor(lt.Placeholder, "b", [2,2], np.float32)

c, = g.make_tensor(lt.Sum, "c", [a, b])

grad_a, = lt.compute_gradient(g, c, [a])

apply_grad, = g.make_tensor(lt.AssignAdd, "apply_grad", a, grad_a)

sess = lt.Session(g)
sess.initialize()
print(sess.run([b, grad_a, apply_grad], feed_dict={b: np.ones([2,2], np.float32)*2}))