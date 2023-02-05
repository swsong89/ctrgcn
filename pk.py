import pickle
import os




with open(os.path.join( 'work_dir/ntu120/xsub/sectrgcn_b/','epoch96_test_score.pkl'), 'rb') as r4:
  r4 = list(pickle.load(r4).items())
  a = 1