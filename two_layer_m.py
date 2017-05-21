import numpy as np
np.random.seed(1)

w1 = 0.5 - np.random.rand(2,3)
b1 = np.zeros(3,)
w2 = 0.5 - np.random.rand(3,1)
b2 = np.zeros(1,)


eta = 0.01

loss = 0
acc = 0


for i in range(100000):
  x = np.random.rand(10,2)

  y1 = np.dot(x,w1)+b1
  z1 = 1./(1+np.exp(-y1))

  y2 = np.dot(z1,w2)+b2
  z2 = 1./(1+np.exp(-y2))

  z_ = np.logical_or(np.sum(x,axis=1) <= 0.5, 
    np.sum(x,axis=1) >= 1.5).astype(float)[:,np.newaxis]

  dw2 = eta*np.dot(z1.T,z2-z_)
  #db2 = eta*np.dot(np.ones((10,1)).T,z2-z_)
  db2 = eta*np.sum(z2-z_,axis=0)

  dz1 = (np.dot(z2-z_,w2.T) * z1 * (1-z1))

  dw1 = eta*np.dot(x.T,dz1)
  #db1 = eta*np.dot(np.ones((10,3)).T,dz1)
  db1 = eta*np.sum(dz1,axis=0) 

  w2 -= dw2
  b2 -= db2

  w1 -= dw1
  b1 -= db1

  loss += np.sum(-z_*np.log(z2)-(1-z_)*np.log(1-z2))

  acc += np.sum(((z2 > 0.5).astype(float) == z_).astype(float))
  if (i+1) % 1000 == 0:
    print loss,acc/(10*1000.)
    loss = 0
    acc = 0

