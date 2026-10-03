"""Single-GPU guarded native PHOENIX training-pseudo extraction stage."""
from pathlib import Path
import modal
HERE=Path(__file__).resolve().parent
app=modal.App('repro-cico-phx-pseudo')
data=modal.Volume.from_name('datasets',version=2)
cache=modal.Volume.from_name('huggingface-cache',version=2)
outputs=modal.Volume.from_name('cheng-2023-cico-results',version=2)
image=(modal.Image.from_registry('ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291')
 .apt_install('git').pip_install('opencv-python-headless==4.11.0.86','beartype==0.19.0','simple-video-utils==0.0.6','av==19.0.1','mock==5.1.0','humanize==4.11.0','tensorboard==2.18.0')
 .run_commands('git clone https://github.com/FangyunWei/SLRT.git /upstream && cd /upstream && git checkout 38a4f7b00da7a858d59b7fabe5093876a84db8e0')
 .add_local_file(HERE.parent/'patches/0001-extractor-video-decoder.patch','/repro/0001.patch',copy=True)
 .add_local_file(HERE.parent/'patches/0002-pseudo-video-decoder.patch','/repro/0002.patch',copy=True)
 .add_local_file(HERE.parent/'patches/0003-pseudo-range-concatenation.patch','/repro/0003.patch',copy=True)
 .run_commands('cd /upstream && git apply /repro/0001.patch && git apply /repro/0002.patch && git apply /repro/0003.patch')
 .env({'HF_HOME':'/cache/huggingface','HF_HUB_CACHE':'/cache/huggingface/hub','OMP_NUM_THREADS':'4'})
 .add_local_file(HERE/'i3d_extract.py','/repro/i3d_extract.py'))
@app.function(image=image,gpu='A100-80GB',cpu=4,memory=32768,timeout=14400,retries=0,volumes={'/datasets':data.read_only(),'/cache/huggingface':cache,'/outputs':outputs})
def run(run_id:str,resume_after_stopped_app:str=''):
 import subprocess,json,datetime,time,re,threading,os,hashlib
 if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,100}',run_id):raise ValueError('Invalid run ID')
 out=Path('/outputs')/run_id;out.mkdir(exist_ok=True)
 def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
 identity={'upstream':'38a4f7b00da7a858d59b7fabe5093876a84db8e0','source':sha('/repro/i3d_extract.py'),'wrapper':sha(__file__),'patches':[sha('/repro/'+n+'.patch') for n in ['0001','0002','0003']],'manifest_sha256':'4974f59634d771679d32c7b7031115506286e9dd1247ba42d41900323cb8d53a','weights_sha256':'6430592464a357dfdaa7f31973cb684663237655fdf23f3999608d162167fc6f','mode':'pseudo','split':'train','max_seconds':14400,'max_segments':2}
 planfile=out/'plan.json';now=time.time()
 if (out/'pseudo/complete.json').exists():raise ValueError('Stage already completed; duplicate execution forbidden')
 if not planfile.exists():
  # Atomic claim rejects duplicate first calls and fails closed after init crash.
  (out/'initial-claim').mkdir(exist_ok=False)
 if planfile.exists():
  plan=json.loads(planfile.read_text());assert plan['identity']==identity
  if not resume_after_stopped_app or resume_after_stopped_app!=plan['last_app_id']:raise ValueError('Resume requires independently stopped previous app identity')
  if not list((out/'pseudo').glob('rank-*.json')):raise ValueError('No closed rank checkpoint: fresh full restart forbidden')
  if plan['segments']>=2:raise ValueError('Segment ceiling')
 else:
  if resume_after_stopped_app:raise ValueError('Cannot resume absent run')
  plan={'identity':identity,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_epoch':now+14400,'segments':0}
 remaining=plan['deadline_epoch']-time.time()
 if remaining<120:raise ValueError('Original deadline reached')
 plan.update(segments=plan['segments']+1,last_app_id=app.app_id)
 planfile.write_text(json.dumps(plan,indent=2));outputs.commit()
 segment=plan['segments'];record={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_epoch':plan['deadline_epoch'],'modal_app_id':app.app_id,'function_call_id':modal.current_function_call_id(),'segment':segment,'native_exit_code':None}
 (out/f'segment-{segment}-started.json').write_text(json.dumps(record,indent=2));outputs.commit()
 timer=threading.Timer(max(1,plan['deadline_epoch']-time.time()),lambda:os._exit(124));timer.daemon=True;timer.start()
 stop=threading.Event()
 def commits():
  while not stop.wait(30):outputs.commit()
 thread=threading.Thread(target=commits,daemon=True);thread.start();t=time.monotonic()
 try:
  with (out/f'segment-{segment}-pip-freeze.txt').open('w') as f:subprocess.run(['python','-m','pip','freeze'],stdout=f,check=True,timeout=30)
  with (out/f'segment-{segment}-hardware.txt').open('w') as f:subprocess.run(['nvidia-smi'],stdout=f,check=True,timeout=30)
  command=['python','/repro/i3d_extract.py','--output',str(out/'pseudo'),'--manifest','/outputs/phx-raw-manifest-v1/manifest.json','--manifest-sha',identity['manifest_sha256'],'--weights','/outputs/training-inputs/bsl5k.pth.tar','--weights-sha',identity['weights_sha256'],'--mode','pseudo','--split','train']
  (out/f'segment-{segment}-command.json').write_text(json.dumps(command))
  with (out/f'segment-{segment}-console.log').open('w') as f:p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,plan['deadline_epoch']-time.time()-90))
  record['native_exit_code']=p.returncode
 except BaseException as e:
  record['exception']=repr(e);raise
 finally:
  stop.set();thread.join(timeout=5)
  record.update(finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_time_seconds=time.monotonic()-t)
  (out/f'segment-{segment}-execution.json').write_text(json.dumps(record,indent=2));outputs.commit();timer.cancel()
 print(json.dumps(record));print((out/f'segment-{segment}-console.log').read_text()[-3000:])
 if record['native_exit_code']:raise RuntimeError('Native pseudo stage failed; no automatic retry')
