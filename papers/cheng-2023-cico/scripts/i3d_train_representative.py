"""Real distinct training clips, native empty-validation path and throughput."""
from pathlib import Path
import os,sys,json,subprocess,time,hashlib,shutil
import torch
from simple_video_utils.metadata import video_metadata
out=Path(sys.argv[1]);source=Path('/upstream/CiCo/I3D_trainer');os.chdir(source)
base=Path('/outputs/phx-pseudo-full-v1/pseudo');clips=[];records=[]
for receipt in sorted(base.glob('rank-*.json')):
 for a in json.loads(receipt.read_text())['artifacts']:
  if a['path'].endswith('.mp4'):
   p=base/a['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==a['sha256'];clips.append(p);records.append(a)
   if len(clips)==32:break
 if len(clips)==32:break
assert len(clips)==32
local=Path('/tmp/cico-train-representative');local.mkdir(exist_ok=True)
rows=[]
for p in clips:
 q=local/p.parent.name/p.name;q.parent.mkdir(exist_ok=True);shutil.copyfile(p,q);rows.append(q)
info={'video_path':[str(p) for p in rows],'class_label':[str(int(p.parent.name)) for p in rows],'class_name':[p.parent.name for p in rows],'frame':[int(video_metadata(str(p)).nb_frames) for p in rows],'split':['train']*len(rows)}
info_path=out/'diagnostic-info.json';info_path.write_text(json.dumps(info,indent=2));os.environ['CICO_PSEUDO_INFO']=str(info_path)
command=['python','main.py','--datasetname','phoenix2014','--pretrained','/outputs/training-inputs/bsl5k.pth.tar','--train-batch','4','--test-batch','3','--lr','.01','--coef','1','--workers','0','--num_figs','0','--snapshot','1','--num-classes','5383','--epochs','2','--checkpoint',str(out/'native')]
(out/'command.json').write_text(json.dumps(command));t=time.monotonic();subprocess.run(command,check=True,timeout=600)
checkpoint=torch.load(out/'native/checkpoint.pth.tar',map_location='cpu',weights_only=True);assert checkpoint['epoch']==2
train=[json.loads((out/f'native/epoch-train-{i}.json').read_text()) for i in range(2)];val=[json.loads((out/f'native/epoch-val-{i}.json').read_text()) for i in range(2)];assert all(x['batches']==0 for x in val)
report={'training_clips':32,'validation_clips':0,'native_empty_validation_verified':True,'epochs':2,'wall_seconds':time.monotonic()-t,'train_epochs':train,'selected_checkpoint':'Final epoch 2 for diagnosis; full recipe selects final epoch15.','input_artifacts':records}
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
