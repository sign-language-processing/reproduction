"""Stage immutable generated training clips and invoke the pinned native trainer."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--deadline',required=True,type=float);p.add_argument('--resume',action='store_true');p.add_argument('--resume-sha');a=p.parse_args()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(a.manifest)==a.manifest_sha
manifest=json.loads(Path(a.manifest).read_text());assert manifest['source_run']=='phx-pseudo-full-v1' and manifest['ranks']==256 and manifest['split']=='train'
out=Path(a.output);out.mkdir(exist_ok=True);local=Path('/tmp/cico-adaptation-clips');local.mkdir(exist_ok=True);rows=[]
for r in manifest['clips']:
 source=Path('/outputs/phx-pseudo-full-v1/pseudo')/r['path'];dest=local/r['path'];dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists() or sha(dest)!=r['sha256']:shutil.copyfile(source,dest)
 assert sha(dest)==r['sha256'];rows.append(dest)
assert len(rows)>0
info={'video_path':[str(p) for p in rows],'class_label':[str(r['class_id']) for r in manifest['clips']],'class_name':[str(r['class_id']) for r in manifest['clips']],'frame':[r['frames'] for r in manifest['clips']],'split':['train']*len(rows)}
info_path=out/'training-info.json';info_path.write_text(json.dumps(info,indent=2));os.environ['CICO_PSEUDO_INFO']=str(info_path)
weights='/outputs/training-inputs/bsl5k.pth.tar';assert sha(weights)=='6430592464a357dfdaa7f31973cb684663237655fdf23f3999608d162167fc6f'
checkpoint=out/'native/checkpoint.pth.tar'
command=['python','main.py','--datasetname','phoenix2014','--pretrained',weights,'--train-batch','4','--test-batch','3','--lr','.01','--coef','1','--workers','0','--num_figs','0','--snapshot','5','--num-classes','5383','--epochs','15','--checkpoint',str(out/'native')]
if a.resume:
 import torch
 assert checkpoint.exists() and sha(checkpoint)==a.resume_sha;state=torch.load(checkpoint,map_location='cpu',weights_only=True);assert 0<state['epoch']<=15 and 'rng_state' in state
 command+=['--resume',str(checkpoint)]
else:assert not checkpoint.exists()
(out/'native-command.json').write_text(json.dumps(command))
if not (a.resume and state['epoch']==15):subprocess.run(command,cwd='/upstream/CiCo/I3D_trainer',check=True,timeout=max(1,a.deadline-time.time()-60))
import torch
state=torch.load(checkpoint,map_location='cpu',weights_only=True);assert state['epoch']==15
(out/'complete.json').write_text(json.dumps({'epoch':15,'training_clips':len(rows),'validation_clips':0,'checkpoint_sha256':sha(checkpoint),'input_manifest_sha256':a.manifest_sha,'selection':'final epoch 15, no validation or test selection'},indent=2))
