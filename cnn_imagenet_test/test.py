import numpy as np


data1 = np.load('meta1')
data2 = np.load('meta2')

different = []
same = []
for k in data1.keys():
  val1 = data1[k]
  val2 = data2[k]
  diff = np.abs(val1-val2)
  if diff.max() > 0.01:
    different.append(k)
  else:
    same.append(k)

for k in same:
  print 'same: ', k

for k in different:
  print 'diff: ', k
