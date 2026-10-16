"""Single-GPU guarded native PHOENIX train/test feature extraction stage."""
from pathlib import Path
import modal
HERE=Path(__file__).resolve().parent
app=modal.App('repro-cico-phx-features')
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
def run(run_id:str,weights_path:str,weights_sha:str,resume_after_stopped_app:str=''):
 import subprocess,json,datetime,time,re,threading,os,hashlib,signal
 if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,100}',run_id):raise ValueError('Invalid run ID')
 if not re.fullmatch(r'[0-9a-f]{64}',weights_sha) or not weights_path.startswith('/outputs/') or '..' in Path(weights_path).parts:raise ValueError('Invalid weights pin')
 out=Path('/outputs')/run_id;out.mkdir(exist_ok=True)
 def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
 identity={'upstream':'38a4f7b00da7a858d59b7fabe5093876a84db8e0','source':sha('/repro/i3d_extract.py'),'wrapper':sha(__file__),'patches':[sha('/repro/'+n+'.patch') for n in ['0001','0002','0003']],'manifest_sha256':'4974f59634d771679d32c7b7031115506286e9dd1247ba42d41900323cb8d53a','weights_sha256':weights_sha,'weights_path':weights_path,'mode':'features','splits':['train','test'],'max_seconds':14400,'max_segments':2}
 planfile=out/'plan.json';now=time.time()
 if (out/'complete.json').exists():raise ValueError('Stage already completed; duplicate execution forbidden')
 if not planfile.exists():
  # Atomic claim rejects duplicate first calls and fails closed after init crash.
  (out/'initial-claim').mkdir(exist_ok=False)
 if planfile.exists():
  plan=json.loads(planfile.read_text());assert plan['identity']==identity
  if not resume_after_stopped_app or resume_after_stopped_app!=plan['last_app_id']:raise ValueError('Resume requires independently stopped previous app identity')
  if not list(out.glob('*/rank-*.json')):raise ValueError('No closed rank checkpoint: fresh full restart forbidden')
  if plan['segments']>=2:raise ValueError('Segment ceiling')
 else:
  if resume_after_stopped_app:raise ValueError('Cannot resume absent run')
  plan={'identity':identity,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_epoch':now+14400,'segments':0}
 remaining=plan['deadline_epoch']-time.time()
 if remaining<120:raise ValueError('Original deadline reached')
 next_segment=plan['segments']+1;(out/f'segment-{next_segment}-claim').mkdir(exist_ok=False)
 plan.update(segments=next_segment,last_app_id=app.app_id)
 planfile.write_text(json.dumps(plan,indent=2));outputs.commit()
 segment=plan['segments'];record={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_epoch':plan['deadline_epoch'],'modal_app_id':app.app_id,'function_call_id':modal.current_function_call_id(),'segment':segment,'native_exit_code':None,'stage_exit_code':None,'completion_only':False}
 (out/f'segment-{segment}-started.json').write_text(json.dumps(record,indent=2));outputs.commit()
 child=[None]
 def kill_child():
  if child[0] is not None:
   try:os.killpg(child[0].pid,signal.SIGKILL)
   except ProcessLookupError:pass
 def deadline_stop():
  kill_child();os._exit(124)
 timer=threading.Timer(max(1,plan['deadline_epoch']-time.time()),deadline_stop);timer.daemon=True;timer.start()
 stop=threading.Event()
 def commits():
  while not stop.wait(30):outputs.commit()
 thread=threading.Thread(target=commits,daemon=True);thread.start();t=time.monotonic()
 try:
  with (out/f'segment-{segment}-pip-freeze.txt').open('w') as f:subprocess.run(['python','-m','pip','freeze'],stdout=f,check=True,timeout=30)
  with (out/f'segment-{segment}-hardware.txt').open('w') as f:subprocess.run(['nvidia-smi'],stdout=f,check=True,timeout=30)
  for split in ['train','test']:
   if (out/split/'complete.json').exists():
    finished=json.loads((out/split/'complete.json').read_text());assert finished['identity']=={'mode':'features','split':split,'probe':False,'manifest_sha256':identity['manifest_sha256'],'weights_sha256':weights_sha,'entrypoint_sha256':identity['source']}
    assert finished['ranks']==list(range(256 if split=='train' else 16))
    for rank in finished['ranks']:
     receipt=json.loads((out/split/f'rank-{rank:03d}.json').read_text())
     for a in receipt['artifacts']:assert sha(out/split/a['path'])==a['sha256']
    continue
   command=['python','/repro/i3d_extract.py','--output',str(out/split),'--manifest','/outputs/phx-raw-manifest-v1/manifest.json','--manifest-sha',identity['manifest_sha256'],'--weights',weights_path,'--weights-sha',weights_sha,'--mode','features','--split',split]
   (out/f'segment-{segment}-{split}-command.json').write_text(json.dumps(command))
   with (out/f'segment-{segment}-console.log').open('a') as f:
    child[0]=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    record['native_exit_code']=child[0].wait(timeout=max(1,plan['deadline_epoch']-time.time()-90))
   if record['native_exit_code']:break
  if all((out/x/'complete.json').exists() for x in ['train','test']) and record['native_exit_code'] in [None,0]:
   record.update(stage_exit_code=0,completion_only=record['native_exit_code'] is None)
   (out/'complete.json').write_text(json.dumps({'splits':['train','test'],'weights_sha256':weights_sha,'completion_only':record['completion_only']}))
 except BaseException as e:
  record['exception']=repr(e);raise
 finally:
  kill_child()
  stop.set();thread.join(timeout=5)
  record.update(finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_time_seconds=time.monotonic()-t)
  (out/f'segment-{segment}-execution.json').write_text(json.dumps(record,indent=2));outputs.commit();timer.cancel()
 print(json.dumps(record));log=out/f'segment-{segment}-console.log'
 if log.exists():print(log.read_text()[-3000:])
 if record['stage_exit_code']!=0 or (record['native_exit_code']!=0 and not record['completion_only']):raise RuntimeError('Native feature stage failed; no automatic retry')
