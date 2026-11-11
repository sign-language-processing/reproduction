"""Paper reconstruction using unmodified pinned DINOv2 backbone."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
sys.path.insert(0,'/opt/dinov2')
from dinov2.hub.backbones import dinov2_vits14

p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
if (out/'metrics.json').exists():print((out/'metrics.json').read_text());raise SystemExit()
started=time.time();start_utc=dt.datetime.now(dt.timezone.utc).isoformat()
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
torch.set_num_threads(4)
assert torch.cuda.is_available()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
root=Path('/datasets/bdsl49-v6-recognition');manifest=json.loads((root/'manifest.json').read_text())
# Stage the same hash-pinned ZIPs sequentially, avoiding thousands of cold
# remote opens; validate every extracted image against the canonical manifest.
import zipfile
local=Path('/tmp/bdsl49');local.mkdir(exist_ok=True)
for name,record in manifest['archives'].items():
 archive=Path('/tmp')/name
 print('Staging archive',name,flush=True)
 shutil.copyfile(root/name,archive)
 assert sha(archive)==record['sha256'],name
 with zipfile.ZipFile(archive) as z:
  for entry in z.infolist():
   if not (local/entry.filename).resolve().is_relative_to(local.resolve()):raise RuntimeError('Unsafe archive path')
  z.extractall(local)
for line in (root/'files.sha256').read_text().splitlines():
 digest,rel=line.split('  ',1)
 assert sha(local/rel)==digest,rel
print('Verified all 14714 local images',flush=True)
normalize=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
train_transform=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(.4),transforms.RandomApply([transforms.RandomAffine(10,translate=(.1,.1),scale=(.9,1.1))],p=.4),transforms.RandomApply([transforms.ColorJitter(brightness=.2,contrast=.2,saturation=.2,hue=.1)],p=.4),transforms.ToTensor(),normalize])
deterministic_transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),normalize])
test_transform=train_transform  # Section4 and Figure4 explicitly specify test augmentation.
class Images(Dataset):
 def __init__(self,split):
  self.items=[];self.transform=train_transform if split=='train' else test_transform
  for f in sorted(local.rglob('*')):
   if f.suffix.lower() in ('.jpg','.jpeg','.png') and split in f.parts:self.items.append((f,int(f.parts[f.parts.index(split)+1])))
 def __len__(self):return len(self.items)
 def __getitem__(self,index):
  f,label=self.items[index]
  with Image.open(f) as image:x=self.transform(image.convert('RGB'))
  return x,label,index
train_data=Images('train');test_data=Images('test')
assert (len(train_data),len(test_data))==(11774,2940)
full_test_items=[str(p.relative_to(local)) for p,_ in test_data.items]
if a.preflight:
 def balanced_subset(data,n):
  return Subset(data,[i for label in range(49) for i in [j for j,(_,y) in enumerate(data.items) if y==label][:n]])
 train_data=balanced_subset(train_data,4);test_data=balanced_subset(test_data,2)
generator=torch.Generator().manual_seed(42)
train_loader=DataLoader(train_data,batch_size=32,shuffle=True,num_workers=4,pin_memory=True,prefetch_factor=2,generator=generator)
test_loader=DataLoader(test_data,batch_size=32,num_workers=4,pin_memory=True,prefetch_factor=2)
# Verify the immutable official backbone artifact before deserialization.
weights=Path(os.environ['TORCH_HOME'])/'hub/checkpoints/dinov2_vits14_pretrain.pth'
expected_weights_sha256='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
if not weights.exists():
 weights.parent.mkdir(parents=True,exist_ok=True)
 torch.hub.download_url_to_file('https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',str(weights),hash_prefix=expected_weights_sha256)
assert sha(weights)==expected_weights_sha256, 'Official backbone checksum mismatch'
model=nn.Sequential(dinov2_vits14(pretrained=True),nn.Sequential(nn.Linear(384,256),nn.ReLU(),nn.Linear(256,49))).cuda()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-6);loss_fn=nn.CrossEntropyLoss()
checkpoint=out/'last.pt';history=[];start_epoch=0
if checkpoint.exists():
 saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);history=saved['history'];start_epoch=saved['epoch'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng']);generator.set_state(saved['loader_rng'])
(out/'freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
(out/'hardware.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
(out/'model.txt').write_text(str(model))
def evaluate(loader=test_loader):
 model.eval();loss=0.;truth=[];pred=[];indices=[];logits=[]
 with torch.no_grad():
  for x,y,i in loader:
   x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);z=model(x)
   loss+=loss_fn(z,y).item()*len(y);truth.extend(y.cpu().tolist());pred.extend(z.argmax(1).cpu().tolist());indices.extend(i.tolist());logits.append(z.cpu().numpy())
 return loss/len(truth),np.array(truth),np.array(pred),np.array(indices),np.concatenate(logits)
for epoch in range(start_epoch,2 if a.preflight else 30):
 t=time.time();model.train();loss=0.;correct=0;count=0
 for x,y,_ in train_loader:
  x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True);z=model(x);l=loss_fn(z,y);l.backward();optimizer.step();loss+=l.item()*len(y);correct+=(z.argmax(1)==y).sum().item();count+=len(y)
 torch.cuda.synchronize();train_seconds=time.time()-t
 test_loss,truth,pred,indices,logits=evaluate();torch.cuda.synchronize()
 row={'epoch':epoch+1,'train_loss':loss/count,'train_accuracy':correct/count,'test_loss':test_loss,'test_accuracy':float(np.mean(truth==pred)),'train_seconds':train_seconds,'epoch_seconds':time.time()-t};history.append(row);print(json.dumps(row),flush=True)
 torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,'history':history,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'loader_rng':generator.get_state()},out/'last.tmp');os.replace(out/'last.tmp',checkpoint)
(out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
# Verify roundtrip and execute a resumed training step on preflight only.
model.eval();x,y,_=next(iter(test_loader));x=x.cuda();y=y.cuda()
with torch.no_grad():before=model(x).cpu()
saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
with torch.no_grad():after=model(x).cpu()
assert torch.allclose(before,after,rtol=1e-5,atol=1e-6)
resume=False
if a.preflight:
 before_steps=[float(v['step']) for v in optimizer.state.values() if 'step' in v];model.train();optimizer.zero_grad(set_to_none=True);loss_fn(model(x),y).backward();optimizer.step();after_steps=[float(v['step']) for v in optimizer.state.values() if 'step' in v];assert all(b+1==c for b,c in zip(before_steps,after_steps));resume=True;model.load_state_dict(saved['model'])
random.seed(314159);np.random.seed(314159);torch.manual_seed(314159)
test_loss,truth,pred,indices,logits=evaluate()
cm=np.zeros((49,49),dtype=np.int64);np.add.at(cm,(truth,pred),1)
precision=np.divide(cm.diagonal(),cm.sum(0),out=np.zeros(49),where=cm.sum(0)!=0);recall=np.divide(cm.diagonal(),cm.sum(1),out=np.zeros(49),where=cm.sum(1)!=0);f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(49),where=precision+recall!=0)
np.savez(out/'predictions.npz',truth=truth,prediction=pred,indices=indices,logits=logits,confusion_matrix=cm)
(out/'test-paths.txt').write_text('\n'.join(full_test_items)+'\n')
# The deterministic view is secondary diagnostic evidence only.
# Primary augmented evaluation was chosen from Section4/Figure4 before scores.
random.seed(314159);np.random.seed(314159);torch.manual_seed(314159)
diagnostic_data=Images('test');diagnostic_data.transform=deterministic_transform
if a.preflight:diagnostic_data=balanced_subset(diagnostic_data,2)
diagnostic_loader=DataLoader(diagnostic_data,batch_size=32,num_workers=4,pin_memory=True,prefetch_factor=2)
diagnostic_loss,diagnostic_truth,diagnostic_pred,diagnostic_indices,diagnostic_logits=evaluate(diagnostic_loader)
diagnostic_cm=np.zeros((49,49),dtype=np.int64);np.add.at(diagnostic_cm,(diagnostic_truth,diagnostic_pred),1)
diagnostic_f1=np.divide(2*diagnostic_cm.diagonal(),diagnostic_cm.sum(0)+diagnostic_cm.sum(1),out=np.zeros(49),where=(diagnostic_cm.sum(0)+diagnostic_cm.sum(1))!=0)
np.savez(out/'deterministic-test-predictions.npz',truth=diagnostic_truth,prediction=diagnostic_pred,indices=diagnostic_indices,logits=diagnostic_logits,confusion_matrix=diagnostic_cm)
deterministic_diagnostic={'seed':314159,'accuracy':float(np.mean(diagnostic_truth==diagnostic_pred)),'macro_f1':float(diagnostic_f1.mean()),'test_count':len(diagnostic_truth),'test_loss':diagnostic_loss,'protocol':'deterministic resize and normalization; secondary diagnostic, never score-selected'}
weights=Path(os.environ['TORCH_HOME'])/'hub/checkpoints/dinov2_vits14_pretrain.pth'
m={'deterministic_test_diagnostic':deterministic_diagnostic,'preflight':a.preflight,'accuracy':float(np.mean(truth==pred)),'correct':int(np.sum(truth==pred)),'test_count':len(truth),'macro_f1':float(f1.mean()),'weighted_f1':float(np.average(f1,weights=cm.sum(1))),'test_loss':test_loss,'epochs_completed':len(history),'seed':42,'evaluation_seed':314159,'evaluation_protocol':'same transforms as training as Section4 and Figure4; fixed single-draw seed314159','checkpoint_selection':'final epoch30','parameter_count':sum(p.numel() for p in model.parameters()),'started_at_utc':start_utc,'finished_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'wall_seconds':time.time()-started,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),'checkpoint_reload_verified':True,'optimizer_resume_verified':resume,'history':history,'torch':torch.__version__,'cuda':torch.version.cuda,'dataset_manifest_sha256':sha(root/'manifest.json'),'weights_sha256':sha(weights),'train_script_sha256':sha(__file__),'files':{f.name:sha(f) for f in out.iterdir() if f.is_file()}}
(out/'metrics.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m))
