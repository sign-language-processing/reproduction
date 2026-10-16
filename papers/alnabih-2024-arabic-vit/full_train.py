"""Declared reconstruction: authoritative corpus split and fixed library defaults."""
import argparse, hashlib, io, json, math, os, random, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor, get_linear_schedule_with_warmup

p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--preflight',action='store_true');args=p.parse_args()
out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
if (out/'metrics.json').exists():
 print((out/'metrics.json').read_text());raise SystemExit(0)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 return h.hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
started=utc();wall=time.monotonic();random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(8)
source=Path('/datasets/arasl-database-grayscale/data/train-00000-of-00001-aa6a48ea2f282316.parquet')
assert sha(source)=='7c6d9b276f5960bf9fb0efc99c7df3d3854b0690101751f74ab30d68a125d3a3'
table=pq.read_table(source);assert len(table)==54049
labels=table['label'].to_numpy();assert len(set(labels))==32
ids=np.random.default_rng(42).permutation(len(table));train_ids=ids[:37834];val_ids=ids[37834:45941];test_ids=ids[45941:]
np.savez(out/'split-indices.npz',train=train_ids,validation=val_ids,test=test_ids)
if args.preflight:train_ids=train_ids[:64];val_ids=val_ids[:32];test_ids=test_ids[:32]
name='google/vit-large-patch16-224-in21k';revision='6074eaf2211423e928c93b93ef773d5da618aa7e'
processor=ViTImageProcessor.from_pretrained(name,revision=revision)
model=ViTForImageClassification.from_pretrained(name,revision=revision,num_labels=32).cuda()
optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=0)
epochs=2 if args.preflight else 3;steps_per_epoch=math.ceil(len(train_ids)/8)
scheduler=get_linear_schedule_with_warmup(optimizer,0,epochs*steps_per_epoch)
def batch(indices):
 rows=table.take(indices).to_pylist();images=[Image.open(io.BytesIO(r['image']['bytes'])).convert('RGB') for r in rows]
 return processor(images=images,return_tensors='pt')['pixel_values'].cuda(),torch.tensor([r['label'] for r in rows],device='cuda')
def evaluate(indices):
 model.eval();pred=[];loss_sum=0.;t=time.monotonic()
 with torch.no_grad():
  for i in range(0,len(indices),8):
   x,y=batch(indices[i:i+8]);z=model(x,labels=y);loss_sum+=z.loss.item()*len(y);pred.extend(z.logits.argmax(1).cpu().tolist())
 torch.cuda.synchronize();pred=np.array(pred);correct=int(np.sum(pred==labels[indices]))
 return {'correct':correct,'count':len(indices),'accuracy_percent':100*correct/len(indices),'loss':loss_sum/len(indices),'seconds':time.monotonic()-t},pred
checkpoint=out/'last.pt';history=[];start_epoch=0
if checkpoint.exists():
 state=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);history=state['history'];start_epoch=state['epoch'];torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);del state
for name_,cmd in [('pip-freeze.txt',['pip','freeze']),('nvidia-smi.txt',['nvidia-smi'])]:(out/name_).write_text(subprocess.check_output(cmd,text=True))
torch.cuda.reset_peak_memory_stats()
reload_verified=False
for epoch in range(start_epoch,epochs):
 model.train();ordered=np.random.default_rng(42+epoch).permutation(train_ids);loss_sum=0.;correct=0;t=time.monotonic()
 for i in range(0,len(ordered),8):
  x,y=batch(ordered[i:i+8]);optimizer.zero_grad(set_to_none=True);z=model(x,labels=y);z.loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step();loss_sum+=z.loss.item()*len(y);correct+=int((z.logits.argmax(1)==y).sum().item())
  if (i//8+1)%500==0:print(json.dumps({'epoch':epoch+1,'step':i//8+1,'steps':steps_per_epoch,'seconds':time.monotonic()-t}),flush=True)
 torch.cuda.synchronize();seconds=time.monotonic()-t;val,_=evaluate(val_ids)
 row={'epoch':epoch+1,'train_loss':loss_sum/len(ordered),'train_accuracy_percent':100*correct/len(ordered),'train_seconds':seconds,'examples_per_second':len(ordered)/seconds,'validation':val};history.append(row);print(json.dumps(row),flush=True)
 state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch+1,'history':history,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
 torch.save(state,out/'last.tmp');os.replace(out/'last.tmp',checkpoint);del state
 (out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
 # Actual optimizer/scheduler/model reload at every epoch also verifies resume on the exact loop.
 x,_=batch(val_ids[:8]);model.eval()
 with torch.no_grad():before=model(x).logits.cpu()
 state=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);del state
 with torch.no_grad():after=model(x).logits.cpu()
 assert torch.allclose(before,after,rtol=1e-5,atol=1e-6);reload_verified=True
 import modal
 modal.Volume.from_name('repro-0285c237-results',version=1).commit()
test,pred=evaluate(test_ids);np.savez(out/'test-predictions.npz',indices=test_ids,labels=labels[test_ids],predictions=pred)
m={'preflight':args.preflight,'scope':'documented reconstruction; original split and schedule unavailable','started_at_utc':started,'finished_at_utc':utc(),'wall_seconds':time.monotonic()-wall,'history':history,'test':test,'epochs':epochs,'seed':42,'checkpoint_selection':'final epoch','checkpoint_reload_verified':reload_verified or start_epoch==epochs,'dataset_examples':54049,'split_counts':{'train':len(train_ids),'validation':len(val_ids),'test':len(test_ids)},'model':name,'model_revision':revision,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'modal_task_id':os.environ.get('MODAL_TASK_ID'),'modal_image_id':os.environ.get('MODAL_IMAGE_ID'),'script_sha256':sha(__file__),'files':{f.name:{'sha256':sha(f),'bytes':f.stat().st_size} for f in out.iterdir() if f.is_file()}}
(out/'metrics.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m),flush=True)
