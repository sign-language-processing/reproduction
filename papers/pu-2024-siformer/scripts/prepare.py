"""Controlled FE→AA preparation using unchanged pinned author functions."""
import argparse, ast, contextlib, gzip, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0,'/opt/siformer/rectifications')
from flexion_extension_rectification import rectify_finger_flexion_and_extension as fe
from abduction_adduction_rectification import rectify_finger_abduction_and_addiction as aa
p=argparse.ArgumentParser();p.add_argument('dataset',choices=['lsa64','wlasl100']);p.add_argument('out');p.add_argument('--tiny',action='store_true');a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
def sha(f):
 h=hashlib.sha256()
 with Path(f).open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
if (out/'manifest.json').exists():
 m=json.loads((out/'manifest.json').read_text())
 for name,h in m['files'].items():assert sha(out/name)==h
 print(json.dumps(m));raise SystemExit()
start=time.time();revision='09a7c1c575849edbd5e245dc6d1c80ddce94188a'
for name in ['abduction_adduction_ranges.csv','flexion_extension_ranges.csv']:
 (out/name).write_bytes(subprocess.check_output(['git','-C','/opt/siformer','show',revision+':active_motion/'+name]))
root=Path('/datasets/lsa64/siformer' if a.dataset=='lsa64' else '/datasets/WLASL/siformer')
sources={'LSA64_60fps.csv':'52a169473cf199cc5432eab988eaacfab5b527fcb339a24952bc1202b2e247dd'} if a.dataset=='lsa64' else {'WLASL100_train_25fps.csv':'3027464fe8e53afafaf3f7d859a43df11529a146af3f15a09b90abdd3b21716b','WLASL100_val_25fps.csv':'19829dcb1bacef8e57fc3c85dbdd86437d495825c90bb7313c6b90bfd48a15fd'}
frames={}
for name,h in sources.items():
 assert sha(root/name)==h
 frame=pd.read_csv(root/name)
 # Every function transforms a row independently. Batches only bound memory and
 # permit CPU preparation to resume; author summary counters are not inputs.
 if a.tiny:frame=frame.iloc[:80].copy()
 parts=[]
 for start_row in range(0,len(frame),64):
  target=out/(name+'.part'+str(start_row)+'.csv')
  if not target.exists():
   source=out/'temporary.csv';frame.iloc[start_row:start_row+64].to_csv(source,index=False)
   with gzip.open(out/(name+'.part'+str(start_row)+'.log.gz'),'wt') as log,contextlib.redirect_stdout(log):
    first=fe(str(source),str(out/'flexion_extension_ranges.csv'),alpha=.4);first.to_csv(out/'temporary-fe.csv',index=False)
    second=aa(str(out/'temporary-fe.csv'),str(out/'abduction_adduction_ranges.csv'),alpha=.4)
   for k in second.columns:
    if isinstance(second.iloc[0][k],list):assert np.isfinite(np.asarray(second[k].tolist(),dtype=float)).all()
   second.to_csv(target,index=False)
  parts.append(pd.read_csv(target));print(json.dumps({'source':name,'completed_rows':min(start_row+64,len(frame))}),flush=True)
 frames[name]=pd.concat(parts,ignore_index=True)
 if a.dataset=='lsa64':
  train,test=train_test_split(np.arange(len(frame)),test_size=.2,random_state=42,stratify=None if a.tiny else frame['labels'])
  frames[name].iloc[train].to_csv(out/'train.csv',index=False);frames[name].iloc[test].to_csv(out/'test.csv',index=False)
  np.savez(out/'split.npz',train=train,test=test)
 else:frames[name].to_csv(out/('train.csv' if 'train' in name else 'test.csv'),index=False)
for name in ['temporary.csv','temporary-fe.csv']:(out/name).unlink(missing_ok=True)
counts={n:len(pd.read_csv(out/(n+'.csv'))) for n in ['train','test']}
m={'dataset':a.dataset,'tiny':a.tiny,'source_sha256':sources,'source_commit':'979a14ed15ed0f20afd77d447ad23c4f4107a2c3','motion_table_commit':revision,'rectification':['FE','AA'],'alpha':.4,'batch_rows':64,'seed':42,'counts':counts,'seconds':time.time()-start,'files':{f.name:sha(f) for f in out.iterdir() if f.is_file()}}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m))
