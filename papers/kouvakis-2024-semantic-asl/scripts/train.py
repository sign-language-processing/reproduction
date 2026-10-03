"""Recognition-only reconstruction from Fig3 parameter counts; no channel simulation."""
import argparse, hashlib, json, os, time, subprocess
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['mnist','rgb','lexset'],default='mnist');parser.add_argument('--preflight',action='store_true');parser.add_argument('--run-id');args=parser.parse_args()
start=time.monotonic();started=datetime.now(timezone.utc).isoformat();torch.manual_seed(42);torch.set_num_threads(8)
slug={'mnist':'sign-language-mnist','rgb':'asl-semcom-rgb','lexset':'synthetic-asl-alphabet'}[args.dataset]
root=Path('/datasets')/slug;manifest=json.loads((root/'manifest.json').read_text())
out=Path('/results')/(args.run_id or (args.dataset+('-preflight' if args.preflight else '-full')));out.mkdir(parents=True,exist_ok=True)
if (out/'metrics.json').exists():print((out/'metrics.json').read_text());raise SystemExit(0)
letters='ABCDEFGHIKLMNOPQRSTUVWXY'
if args.dataset=='mnist':
 def load(split):
  paths=sorted((root/'files').rglob('sign_mnist_'+split+'.csv'));assert paths
  p=paths[0];raw=np.loadtxt(p,delimiter=',',skiprows=1,dtype=np.uint8)
  y=raw[:,0].astype(np.int64);y[y>9]-=1
  x=torch.from_numpy(raw[:,1:].reshape(-1,1,28,28)).cuda()
  return x,torch.tensor(y,device='cuda')
 train_x,train_y=load('train');test_x,test_y=load('test')
 assert len(train_y)==27455 and len(test_y)==7172
 def batch(x,idx):return F.interpolate(x[idx].float()/255,size=(100,100),mode='bilinear',align_corners=False).expand(-1,3,-1,-1)
else:
 files=sorted(p for p in (root/'files').rglob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png'])
 def load(split):
  chosen=[p for p in files if split in str(p).lower() and p.parent.name.upper() in letters and len(p.parent.name)==1]
  x=np.stack([np.asarray(Image.open(p).convert('RGB').resize((100,100),Image.Resampling.BILINEAR)) for p in chosen]);y=np.array([letters.index(p.parent.name.upper()) for p in chosen])
  return torch.from_numpy(x.transpose(0,3,1,2)).cuda(),torch.tensor(y,device='cuda')
 train_x,train_y=load('train');test_x,test_y=load('test')
 assert (len(train_y),len(test_y))==({'rgb':(10490,1800),'lexset':(21600,2400)}[args.dataset])
 def batch(x,idx):return x[idx].float()/255

def make_model():
 layers=[];ch=3
 for i,(cout,k) in enumerate([(128,7),(128,5),(128,2),(128,2),(32,2)]):
  layers.extend([nn.Conv2d(ch,cout,k),nn.ReLU()]);ch=cout
  if i<4:layers.append(nn.MaxPool2d(2))
 return nn.Sequential(*layers,nn.Flatten(),nn.Linear(288,128),nn.ReLU(),nn.Linear(128,24)).cuda()
model=make_model()
assert sum(p.numel() for p in model.parameters())==616504
opt=torch.optim.Adam(model.parameters(),lr=1e-3)
checkpoint=out/'checkpoint.pt';epochs=1 if args.preflight else {'mnist':90,'rgb':80,'lexset':60}[args.dataset]
first_epoch=0;history=[]
if os.environ.get('REQUIRE_CHECKPOINT')=='1' and not checkpoint.exists():raise RuntimeError('Recovery requires a durable checkpoint; fresh restart refused')
if checkpoint.exists():
 state=torch.load(checkpoint,weights_only=True);model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);first_epoch=state['epoch'];history=state['history'];torch.set_rng_state(state['rng']);torch.cuda.set_rng_state(state['cuda_rng'])
peak=0;train_seconds=0
for epoch in range(first_epoch,epochs):
 model.train();idx=torch.randperm(len(train_y),device='cuda')[:256] if args.preflight else torch.randperm(len(train_y),device='cuda');correct=0;total_loss=0
 torch.cuda.synchronize();t=time.monotonic()
 for b in idx.split(64):
  opt.zero_grad(set_to_none=True);pred=model(batch(train_x,b));loss=F.cross_entropy(pred,train_y[b]);loss.backward();opt.step();correct+=(pred.argmax(1)==train_y[b]).sum().item();total_loss+=loss.item()*len(b)
 torch.cuda.synchronize();elapsed=time.monotonic()-t;train_seconds+=elapsed
 history.append({'epoch':epoch+1,'online_training_accuracy_percent':100*correct/len(idx),'training_loss':total_loss/len(idx),'seconds':elapsed,'examples':len(idx)})
 torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'history':history,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},out/'checkpoint.partial')
 (out/'checkpoint.partial').replace(checkpoint)
 print(json.dumps(history[-1]),flush=True)
 if not args.preflight:
  import modal
  modal.Volume.from_name('repro-2248c066-results',version=2).commit()
state=torch.load(checkpoint,weights_only=True)
assert all(torch.equal(v,state['model'][k]) for k,v in model.state_dict().items())
if args.preflight:
 model=make_model();opt=torch.optim.Adam(model.parameters(),lr=1e-3)
model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
assert all(torch.equal(v,state['model'][k]) for k,v in model.state_dict().items())
assert all(torch.equal(opt.state_dict()['state'][k][field],v[field]) for k,v in state['optimizer']['state'].items() for field in v)
del state
if args.preflight:
 model.train();b=torch.arange(64,device='cuda');opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(batch(train_x,b)),train_y[b]);loss.backward();opt.step();resume_loss=loss.item()
else:resume_loss=None
model.eval();predictions=[];eval_n=128 if args.preflight else len(test_y)
with torch.no_grad():
 for b in torch.arange(eval_n,device='cuda').split(64):predictions.extend(model(batch(test_x,b)).argmax(1).cpu().tolist())
truth=test_y[:eval_n].cpu().tolist();cm=np.zeros((24,24),dtype=np.int64)
for y,p in zip(truth,predictions):cm[y,p]+=1
np.savez_compressed(out/'predictions.npz',prediction=np.array(predictions),label=np.array(truth),confusion=cm)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
result={'scope':'conditional preflight' if args.preflight else 'recognition full train/test, validation not reconstructed','dataset':args.dataset,'started_at_utc':started,'finished_at_utc':datetime.now(timezone.utc).isoformat(),'parameters':616504,'train_count':len(train_y),'test_count':len(test_y),'evaluated':eval_n,'test_accuracy_percent':100*int(np.trace(cm))/eval_n,'correct':int(np.trace(cm)),'history':history,'checkpoint_resume_verified':True,'resume_loss':resume_loss,'examples_per_second':sum(x['examples'] for x in history)/sum(x['seconds'] for x in history),'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),'wall_time_seconds':time.monotonic()-start,'seed':42,'optimizer':'Adam lr0.001 default betas, batch64','torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'modal_task_id':os.environ.get('MODAL_TASK_ID'),'modal_image_id':os.environ.get('MODAL_IMAGE_ID'),'dataset_manifest_sha256':digest(root/'manifest.json'),'checkpoint_sha256':digest(checkpoint),'predictions_sha256':digest(out/'predictions.npz')}
result['per_class']=[{'class_id':i,'letter':letters[i],'support':int(cm[i].sum()),'precision_percent':100*float(cm[i,i])/max(1,int(cm[:,i].sum())),'recall_percent':100*float(cm[i,i])/max(1,int(cm[i].sum())),'f1_percent':200*float(cm[i,i])/max(1,int(cm[i].sum()+cm[:,i].sum()))} for i in range(24)]
for name,cmd in [('pip-freeze.txt',['pip','freeze']),('nvidia-smi.txt',['nvidia-smi'])]:(out/name).write_text(subprocess.check_output(cmd,text=True))
(out/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
