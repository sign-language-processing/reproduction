"""Bounded native trainer microcase; every call uses the repro-sign wrapper."""
from pathlib import Path
import modal
HERE=Path(__file__).resolve().parent
app=modal.App('repro-cico-trainer-probe')
data=modal.Volume.from_name('datasets',version=2)
cache=modal.Volume.from_name('huggingface-cache',version=2)
outputs=modal.Volume.from_name('cheng-2023-cico-results',version=2)
image=(modal.Image.from_registry('ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291')
 .apt_install('git').pip_install('opencv-python-headless==4.11.0.86','beartype==0.19.0','simple-video-utils==0.0.6','av==19.0.1','mock==5.1.0','humanize==4.11.0','tensorboard==2.18.0','zsvision==0.7.12','mergedeep==1.3.4')
 .run_commands('git clone https://github.com/FangyunWei/SLRT.git /upstream && cd /upstream && git checkout 38a4f7b00da7a858d59b7fabe5093876a84db8e0'))
for name in ['0004-trainer-data-paths-and-splits.patch','0005-trainer-project-video-decoder.patch','0006-trainer-python-callable.patch','0007-trainer-initialization-and-checkpoint-recovery.patch','0009-trainer-recovery-evidence.patch']:
 image=image.add_local_file(HERE.parent/'patches'/name,'/repro/'+name,copy=True).run_commands('cd /upstream && git apply --unidiff-zero /repro/'+name)
image=image.env({'HF_HOME':'/cache/huggingface','HF_HUB_CACHE':'/cache/huggingface/hub','OMP_NUM_THREADS':'4'}).add_local_file(HERE/'i3d_train_probe.py','/repro/i3d_train_probe.py').add_local_file(HERE/'i3d_train_representative.py','/repro/i3d_train_representative.py')
@app.function(image=image,gpu='A100-80GB',cpu=4,memory=16384,timeout=900,retries=0,volumes={'/datasets':data.read_only(),'/cache/huggingface':cache,'/outputs':outputs})
def run(run_id:str,representative:bool=False):
 import subprocess,json,datetime,time,re,threading,os
 if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,100}',run_id):raise ValueError('Invalid run ID')
 out=Path('/outputs')/run_id;out.mkdir(exist_ok=False);t=time.monotonic()
 record={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'modal_app_id':app.app_id,'function_call_id':modal.current_function_call_id(),'native_exit_code':None}
 (out/'started.json').write_text(json.dumps(record,indent=2));outputs.commit()
 timer=threading.Timer(890,lambda:os._exit(124));timer.daemon=True;timer.start()
 try:
  with (out/'console.log').open('w') as f:p=subprocess.run(['python','/repro/i3d_train_representative.py' if representative else '/repro/i3d_train_probe.py',str(out)],stdout=f,stderr=subprocess.STDOUT,timeout=820)
  record['native_exit_code']=p.returncode
 finally:
  record.update(finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_time_seconds=time.monotonic()-t);(out/'execution.json').write_text(json.dumps(record,indent=2));outputs.commit();timer.cancel()
 print(json.dumps(record));print((out/'console.log').read_text()[-16000:])
 if record['native_exit_code']:raise RuntimeError('Native trainer probe failed')

@app.function(image=image,cpu=2,memory=8192,timeout=300,retries=0,volumes={'/datasets':data.read_only(),'/cache/huggingface':cache,'/outputs':outputs})
def compare(run_id:str):
 import torch,json,datetime,time
 out=Path('/outputs')/run_id;out.mkdir(exist_ok=False);t=time.monotonic()
 report={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'modal_app_id':app.app_id,'function_call_id':modal.current_function_call_id(),'epochs':{}}
 for epoch in [1,2]:
  a=torch.load(f'/outputs/native-trainer-probe-v1/split/checkpoint_{epoch:03d}.pth.tar',map_location='cpu',weights_only=True)
  b=torch.load(f'/outputs/native-trainer-probe-v1/continuous/checkpoint_{epoch:03d}.pth.tar',map_location='cpu',weights_only=True)
  diffs={k:float((v.float()-b['state_dict'][k].float()).abs().max()) for k,v in a['state_dict'].items() if not torch.equal(v,b['state_dict'][k])}
  ra,rb=a['rng_state'],b['rng_state']
  report['epochs'][epoch]={'different_tensors':len(diffs),'max_parameter_difference':max(diffs.values(),default=0),'largest_differences':sorted(diffs.items(),key=lambda x:-x[1])[:5],'torch_rng_equal':torch.equal(ra['torch'],rb['torch']),'cuda_rng_equal':all(torch.equal(x,y) for x,y in zip(ra['cuda'],rb['cuda'])),'python_rng_equal':ra['python']==rb['python'],'numpy_rng_equal':ra['numpy']==rb['numpy']}
 report['wall_seconds']=time.monotonic()-t;report['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(out/'report.json').write_text(json.dumps(report,indent=2));outputs.commit();print(json.dumps(report))
