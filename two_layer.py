import numpy as np
w11 = 0.5 - np.random.rand()
w12 = 0.5 - np.random.rand()
b1  = 0.5 - np.random.rand()

w21 = 0.5 - np.random.rand()
w22 = 0.5 - np.random.rand()
b2  = 0.5 - np.random.rand()

w31 = 0.5 - np.random.rand()
w32 = 0.5 - np.random.rand()
b3  = 0.5 - np.random.rand()

w41 = 0.5 - np.random.rand()
w42 = 0.5 - np.random.rand()
w43 = 0.5 - np.random.rand()
b4  = 0.5 - np.random.rand()


eta = 0.01
loss = 0
acc = 0
to_print = 50000
for i in range(10000000):
  x1 =np.random.rand()
  x2 =np.random.rand()
  z_ = float((x1+x2)< 0.5 or (x1+x2)> 1.5)

  y1 = w11*x1+w12*x2+b1
  z1 = 1/(1+np.exp(-y1))

  y2 = w21*x1+w12*x2+b2
  z2 = 1/(1+np.exp(-y2))

  y3 = w31*x1+w32*x2+b3
  z3 = 1/(1+np.exp(-y3))

  y4 = w41*z1+w42*z2+w43*z3+b4
  z4 = 1/(1+np.exp(-y4))

  loss += -z_*np.log(z4) - (1-z_)*np.log(1-z4)

  dw41 = eta*(z4-z_)*z1
  dw42 = eta*(z4-z_)*z2
  dw43 = eta*(z4-z_)*z3
  db4  = eta*(z4-z_)

  dw11 = eta*(z4-z_)*w41*z1*(1-z1)*x1
  dw12 = eta*(z4-z_)*w41*z1*(1-z1)*x2
  db1  = eta*(z4-z_)*w41*z1*(1-z1)

  dw21 = eta*(z4-z_)*w42*z2*(1-z2)*x1
  dw22 = eta*(z4-z_)*w42*z2*(1-z2)*x2
  db2  = eta*(z4-z_)*w42*z2*(1-z2)

  dw31 = eta*(z4-z_)*w43*z3*(1-z3)*x1
  dw32 = eta*(z4-z_)*w43*z3*(1-z3)*x2
  db3  = eta*(z4-z_)*w43*z3*(1-z3)

  w41 -= dw41
  w42 -= dw42
  w43 -= dw43

  w31 -= dw31
  w32 -= dw32
  w21 -= dw21
  w22 -= dw22
  w11 -= dw11
  w12 -= dw12

  b4 -= db4
  b3 -= db3
  b2 -= db2
  b1 -= db1

  acc += float(z_ == float(z4 > 0.5))
  if (i+1) % to_print == 0:
    print loss, acc/float(to_print)
    acc = 0
    loss = 0
    
