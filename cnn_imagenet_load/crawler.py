import numpy as np
import urllib
import os
import glog as log
import argparse

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('start', type=int, help='start id')
parser.add_argument('end', type=int, help='end id')

def download(start, end):
  parse_dict = np.load('parse_dict')
  image = urllib.URLopener()
  for k in parse_dict.keys()[start:end]:
    # makedir of k
    log.info('crawling images of class %s' % k)
    data_path = os.path.join('/media/DATA/ImageNet/Extra/', k)
    if not os.path.exists(data_path):
      os.mkdir(data_path)
      cnt = 0
      for link in parse_dict[k][:500]:
        fn = os.path.join(data_path, '%s_%d.jpg' %(k, cnt))
        cnt += 1
        if cnt % 20 == 0: log.info('%d images' % cnt)
      # print fn
      try: 
        image.retrieve(link, fn)
      except IOError:
        cnt -= 1
    # print len(parse_dict[k])


if __name__ == '__main__':
  args = parser.parse_args()
  print args.start
  print args.end
  download(args.start, args.end)
