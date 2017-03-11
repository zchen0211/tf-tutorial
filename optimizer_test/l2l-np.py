import numpy as np
import glog as log


# problem
batch_size = 100000
dim = 3
sgd_step = 1
sample_step = 50000
visualize_step = 50

lr_fast = 1e-1
lr_slow = 1e-3

np.random.seed(123)

transform = np.array([[1., 2., 2.], [0., 1., 2.], [0., 0., 1.]]).astype(np.float32)
transform_inv = np.linalg.inv(transform)
    
Ws = np.identity(dim).astype(np.float32)
# Ws = transform_inv

for i in range(sample_step):
  if (i+1) % visualize_step == 0:
    print '################################################################'

  # sample problem:
  W_true = np.random.normal(size=[1, dim]).astype(np.float32)

  Wf = np.zeros((1, dim))
  
  for sgd_i in range(sgd_step):
    # sample x
    xnp1 = np.random.normal(size=[dim, batch_size]).astype(np.float32)
    x_transform1 = np.matmul(transform, xnp1)

    y1 = np.matmul(W_true, x_transform1)

    # raw loss
    if (i+1) % visualize_step == 0:
      loss = 0.5 * np.mean(np.square(y1 - np.matmul(Wf, np.matmul(Ws, x_transform1))))
      print 'raw loss: ', loss
      print 'raw fast learner: ', Wf
      print ' '

    # fast learner update
    Wf_grad = lr_fast * np.matmul(np.matmul(Ws, x_transform1), np.transpose(y1))
    Wf = Wf + np.transpose(Wf_grad) / float(batch_size)
    # print 'fast learner:'

    # loss after faste learner update
    if (i+1) % visualize_step == 0:
      loss = 0.5 * np.mean(np.square(y1 - np.matmul(Wf, np.matmul(Ws, x_transform1))))
      print 'loss after fast update: ', loss
      print 'current fast learner: ', Wf 
      print ' '

    # slow learner update
    xnp2 = np.random.normal(size=[dim, batch_size]).astype(np.float32)
    x_transform2 = np.matmul(transform, xnp2)

    y2 = np.matmul(W_true, x_transform2)
    predict_y2 = np.matmul(Wf, np.matmul(Ws, x_transform2))
    delta_y2 = predict_y2 - y2

    if (i+1) % visualize_step == 0:
      loss = 0.5 * np.mean(np.square(y2 - np.matmul(Wf, np.matmul(Ws, x_transform2))))
      print 'raw loss before slow update: ', loss
      print 'current fast learner: ', Wf 
      print ' '

    grad_Ws = lr_fast * np.matmul(x_transform1, np.transpose(y1)) / float(batch_size)
    grad_Ws = np.matmul(grad_Ws, delta_y2)
    grad_Ws = np.matmul(grad_Ws, np.transpose(x_transform2)) / float(batch_size)
    grad_Ws = np.matmul(Ws, (grad_Ws + np.transpose(grad_Ws)))
    grad_Ws = grad_Ws
    # print grad_Ws
    Ws = Ws - lr_slow * grad_Ws

    if (i+1) % visualize_step == 0:
      loss = 0.5 * np.mean(np.square(y2 - np.matmul(Wf, np.matmul(Ws, x_transform2)))) # / float(batch_size)
      print 'loss after slow update: ', loss
      print 'Current slow learner: '
      print Ws

