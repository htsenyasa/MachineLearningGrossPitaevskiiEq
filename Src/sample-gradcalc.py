from __future__ import print_function
import torch
from torch.autograd import Variable

# Pytorch uses automatic differentiation
# The idea is that every operation acts on variable can be expressed in term of elementary functions and operators. (In some limits of couse)
# In the light of this idea, when a variable created via Variable function, pytorch framework starts to trace every operation that is acted on it.
# Let say i created a = Variable(..)
# Then, b = a * a
# And finally z = 2 * b
# Here z is function of a implicitly. So we need to imply chain rule to calculate it is derivative.
# dz / da = 4 * a
#

a = torch.ones(1,0)
a = Variable(a, requires_grad = True)
b = a * a
z = 2 * b

z.backward()
print(a.grad)

# z = 2 * a^2 -> dz/da = 4 * a 
