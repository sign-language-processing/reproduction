"""Invoke published train.py; collect its evaluation evidence and resume check."""
import argparse, datetime, hashlib, json, os, subprocess, time
from pathlib import Path
import sys
sys.path.insert(0, "/opt/siformer")
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('dataset',choices=['lsa64','wlasl100']);p.add_argument('data');p.add_argument('out');p.add_argument('--preflight',action='store_true');a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);os.chdir(out)
def sha(f):
 h=hashlib.sha256()
 with Path(f).open('rb') as s:
  for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
if (out/'metrics.json').exists():print((out/'metrics.json').read_text());raise SystemExit()
manifest=json.loads((Path(a.data)/'manifest.json').read_text())
if not a.preflight:
 assert manifest['counts']==({'train':2541,'test':636} if a.dataset=='lsa64' else {'train':3200,'test':800}), 'Unexpected full split counts'
for n in ['train.csv','test.csv']:assert sha(Path(a.data)/n)==manifest['files'][n]
os.environ['REPRO_EVIDENCE']=str(out)
start=time.time();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
command=['python','-u','/opt/siformer/train.py','--experiment_name',a.dataset,'--num_classes','64' if a.dataset=='lsa64' else '100','--seed','42','--batch_size','24','--num_worker','4','--training_set_path',str(Path(a.data)/'train.csv'),'--validation_set','from-file','--validation_set_path',str(Path(a.data)/'test.csv'),'--epochs']
subprocess.run(command+['2' if a.preflight else '100'],check=True)
resume_verified=False
if a.preflight:
 state=torch.load(out/'resume.pth',weights_only=False,map_location='cpu');assert state['epoch']==2
 before=max(int(v['step']) for v in state['optimizer']['state'].values())
 subprocess.run(command+['3'],check=True)
 state=torch.load(out/'resume.pth',weights_only=False,map_location='cpu')
 assert state['epoch']==3 and max(int(v['step']) for v in state['optimizer']['state'].values())>before
 resume_verified=True
history=[json.loads(s) for s in (out/'epoch-metrics.jsonl').read_text().splitlines()]
best=max(history,key=lambda x:x['validation_accuracy']);epoch=best['epoch'];evidence=np.load(out/f'evaluation-epoch-{epoch}.npz',allow_pickle=False)
raw=json.loads((out/f'evaluation-epoch-{epoch}.json').read_text())
assert np.isfinite(evidence['logits']).all(), 'Non-finite evaluator logits'
correct=int((evidence['logits'].argmax(1)==evidence['labels']).sum());assert correct==raw['correct'] and correct/len(evidence['labels'])==best['validation_accuracy']
checkpoint=out/'out-checkpoints'/a.dataset/f'checkpoint_v_{(epoch+8)//10}.pth'
assert len(evidence['labels'])==manifest['counts']['test']
if not a.preflight:assert checkpoint.exists(), 'No selected validation checkpoint was saved by upstream'
if checkpoint.exists():
 model=torch.load(checkpoint,weights_only=False,map_location='cpu');assert hasattr(model,'state_dict')
result={'dataset':a.dataset,'preflight':a.preflight,'epochs':history[-1]['epoch'],'selected_epoch':epoch,'checkpoint_selection':'Highest released-validation accuracy across all epochs; earliest tie. The released held-out partition is also used for checkpoint selection.','accuracy_percent':correct/len(evidence['labels'])*100,'correct':correct,'total':len(evidence['labels']),'resume_verified':resume_verified,'selected_checkpoint':str(checkpoint.relative_to(out)) if checkpoint.exists() else None,'selected_predictions':f'evaluation-epoch-{epoch}.npz','history':history,'started_at_utc':utc,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.time()-start,'data_manifest_sha256':sha(Path(a.data)/'manifest.json'),'source_commit':'979a14ed15ed0f20afd77d447ad23c4f4107a2c3','adapter_sha256':sha(__file__),'patch_sha256':{f.name:sha(f) for f in Path('/opt/repro-patches').glob('*.patch')},'torch_version':torch.__version__,'device':torch.cuda.get_device_name(),'seed':42,'decoder_heads':9,'files':{str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f.name!='stdout.log'}}
(out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
