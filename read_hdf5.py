import h5py
import pdb

with h5py.File('test.h5','r') as hf:
  label = hf.get('label')
  pdb.set_trace()
  print label
