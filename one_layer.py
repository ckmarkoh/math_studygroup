import numpy as np

w1 = 0
w2 = 0
b =0

eta = 0.001
loss = 0
acc = 0
to_print = 10000
for i in range(100000):
  x1 =np.random.rand()
  x2 =np.random.rand()
  z_ = float(x1+x2 > 1)
  y = w1*x1+w2*x2+b
  z = 1/(1+np.exp(-y))
  loss += -z_*np.log(z) - (1-z_)*np.log(1-z)
  w1 -= eta*(z-z_)*x1
  w2 -= eta*(z-z_)*x2
  b -= eta*(z-z_)
  acc += float(z_ == float(z > 0.5))
  if (i+1) % to_print == 0:
    print loss, acc/float(to_print)
    acc = 0
    loss = 0
    
