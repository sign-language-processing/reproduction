"""Real-video native I3D preflight; no final target values are claimed."""
from pathlib import Path
import sys,os,json,time,hashlib,shutil,subprocess
import numpy as np
import torch
from simple_video_utils.metadata import video_metadata
from simple_video_utils.frames import read_frames_exact
out=Path(sys.argv[1]);start=time.monotonic()
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
source=Path('/upstream/CiCo/I3D_feature_extractor');sys.path.insert(0,str(source));os.chdir(source)
from models.i3d import InceptionI3d
from datasets.phoenix2014 import PHOENIX2014
sourcepaths=sorted(Path('/datasets/rwth-phoenix-2014-t/videos/train').glob('*.mp4'))[:3]
local=Path('/tmp/phx-probe/train');local.mkdir(parents=True,exist_ok=True)
report={'seed':0,'input_files':[],'weights_sha256':None}
for p in sourcepaths:
 dest=local/p.name;shutil.copyfile(p,dest);assert sha(dest)==sha(p)
 meta=video_metadata(str(dest));frames=list(read_frames_exact(str(dest),0,15))
 report['input_files'].append({'path':str(p),'sha256':sha(dest),'metadata':str(meta),'decoded_shape':list(np.stack(frames).shape)})
torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(4)
weight=Path('/outputs/training-inputs/bsl5k.pth.tar');report['weights_sha256']=sha(weight)
assert report['weights_sha256']=='6430592464a357dfdaa7f31973cb684663237655fdf23f3999608d162167fc6f'
model=InceptionI3d(num_classes=5383,include_embds=True,num_in_frames=16).cuda()
state=torch.load(weight,map_location='cpu',weights_only=True)['state_dict'];model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
dataset=PHOENIX2014(root_path=str(local.parent),split='train',rank=255,gpu_collation=256,setname='val')
loader=torch.utils.data.DataLoader(dataset,batch_size=18,shuffle=False,num_workers=0,collate_fn=dataset.collate_fn)
model.eval();torch.cuda.reset_peak_memory_stats();times=[];scores=[];batches=[]
for i,data in enumerate(loader):
 t=time.monotonic();data['rgb']=data['rgb'].cuda();data=dataset.gpu_collater(data)
 with torch.no_grad():value=model(data['rgb'])
 torch.cuda.synchronize();times.append(time.monotonic()-t)
 scores.extend(value['logits'].softmax(-1).max(-1).values.cpu().tolist());batches.append(data['rgb'][:4].detach())
 assert value['embds'].shape[1]==1024
 if i>=5:break
report.update(inference_batch_seconds=times,inference_windows=len(scores),mean_max_confidence=float(np.mean(scores)),above_pseudo_threshold=sum(x>.6 for x in scores),inference_peak_allocated_bytes=torch.cuda.max_memory_allocated())
# Diagnostic optimizer steps use the native pseudo-label rule, not ground truth.
# They verify training mechanics without representing full adaptation results.
optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=0)
training=[];model.train()
for batch in batches[:3]:
 with torch.no_grad():
  model.eval();y=model(batch)['logits'].argmax(-1);model.train()
 t=time.monotonic();optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(batch)['logits'],y);loss.backward();optimizer.step();torch.cuda.synchronize()
 training.append({'loss':loss.item(),'seconds':time.monotonic()-t,'batch_size':len(batch)})
assert all(np.isfinite(x['loss']) for x in training)
checkpoint=out/'probe-checkpoint.pt';torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},checkpoint)
restored=InceptionI3d(num_classes=5383,include_embds=True,num_in_frames=16).cuda();restored_opt=torch.optim.SGD(restored.parameters(),lr=.01,momentum=.9,weight_decay=0)
r=torch.load(checkpoint,weights_only=True);restored.load_state_dict(r['model']);restored_opt.load_state_dict(r['optimizer'])
assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
a=optimizer.state_dict();b=restored_opt.state_dict();assert a['param_groups']==b['param_groups']
assert all(torch.equal(v,b['state'][k][name]) for k,entry in a['state'].items() for name,v in entry.items())
restored.train();restored_opt.zero_grad();loss=torch.nn.functional.cross_entropy(restored(batches[0])['logits'],y[:len(batches[0])]);loss.backward();restored_opt.step();assert torch.isfinite(loss)
report.update(training_steps=training,checkpoint_sha256=sha(checkpoint),fresh_model_optimizer_restore=True,post_restore_loss=loss.item(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),wall_seconds=time.monotonic()-start,gpu=torch.cuda.get_device_name())
(out/'probe-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
