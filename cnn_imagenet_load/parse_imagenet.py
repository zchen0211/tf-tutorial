
def parse_meta(fn_synset='imagenet_lsvrc_2015_synsets.txt', fn_metadata='imagenet_metadata.txt'):
  # synset file: 1, 000 lines
  # each line: nxxxxxx
  with open(fn_synset, 'r') as fi:
    lines = fi.readlines()

  # dict: nxxxx -> 0,1,...,999
  id_2_num = {}
  for item in lines:
    id_2_num[item[:-1]] = len(id_2_num)

  # metadata file: 21,842 lines
  # each line: nxxxx -> words
  with open(fn_metadata, 'r') as fi:
    lines = fi.readlines()

  # split result
  num_2_des = {}
  for line in lines:
    id_, meta_ = line.split('\t')
    if id_ in id_2_num:
      num_ = id_2_num[id_]
      num_2_des[num_] = meta_[:-1]

  return num_2_des

def parse_val(fn = '/media/DATA/ImageNet/val.txt'):
  # 50,000 lines
  # each line: ILSVRC2012_val_xxxx.JPEG id(0-999)
  with open(fn, 'r') as fi:
    lines = fi.readlines()
  fn_2_id = {}
  id_2_fn = {}
  for line in lines:
    fn, id_ = line.split(' ')
    id_ = int(id_[:-1])
    fn_2_id[fn] = id_
    if id_ in id_2_fn:
      id_2_fn[id_].append(fn)
    else:
      id_2_fn[id_] = [fn]
  return fn_2_id, id_2_fn

if __name__ == '__main__':
  # num_2_des = parse_meta()
  # print num_2_des.keys()

  val_fn_2_id, val_id_2_fn = parse_val()
  print len(val_fn_2_id)
  print [len(val_id_2_fn[item]) for item in val_id_2_fn]

  id_ = 388
  data_path = '/media/DATA/ImageNet/val2012'
  ims = []
  for fn in val_id_2_fn[id_][:10]:
    fn = os.path.join(data_path, fn)
    im = imread(fn)
