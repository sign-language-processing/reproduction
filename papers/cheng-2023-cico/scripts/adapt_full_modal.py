"""Bounded native trainer microcase; every call uses the repro-sign wrapper."""
from pathlib import Path
import modal
HERE=Path(__file__).resolve().parent
app=modal.App('repro-cico-phx-adaptation')
data=modal.Volume.from_name('datasets',version=2)
cache=modal.Volume.from_name('huggingface-cache',version=2)
outputs=modal.Volume.from_name('cheng-2023-cico-results',version=2)
image=(modal.Image.from_registry('ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291')
 .apt_install('git').pip_install('opencv-python-headless==4.11.0.86','beartype==0.19.0','simple-video-utils==0.0.6','av==19.0.1','mock==5.1.0','humanize==4.11.0','tensorboard==2.18.0','zsvision==0.7.12','mergedeep==1.3.4')
 .run_commands('git clone https://github.com/FangyunWei/SLRT.git /upstream && cd /upstream && git checkout 38a4f7b00da7a858d59b7fabe5093876a84db8e0'))
for name in ['0004-trainer-data-paths-and-splits.patch','0005-trainer-project-video-decoder.patch','0006-trainer-python-callable.patch','0007-trainer-initialization-and-checkpoint-recovery.patch','0009-trainer-recovery-evidence.patch']:
 image=image.add_local_file(HERE.parent/'patches'/name,'/repro/'+name,copy=True).run_commands('cd /upstream && git apply --unidiff-zero /repro/'+name)
image=image.env({'HF_HOME':'/cache/huggingface','HF_HUB_CACHE':'/cache/huggingface/hub','OMP_NUM_THREADS':'4'}).add_local_file(HERE/'i3d_adapt.py','/repro/i3d_adapt.py')
@app.function(image=image,gpu='A100-80GB',cpu=4,memory=16384,timeout=10800,retries=0,volumes={'/datasets':data.read_only(),'/cache/huggingface':cache,'/outputs':outputs})
def run(run_id:str,manifest_sha:str,resume_after_stopped_app:str=''):
 import subprocess,json,datetime,time,re,threading,os,hashlib,signal
 if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,100}',run_id) or not re.fullmatch(r'[0-9a-f]{64}',manifest_sha):raise ValueError('Invalid run or manifest')
 out=Path('/outputs')/run_id;out.mkdir(exist_ok=True)
 def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
 identity={'upstream':'38a4f7b00da7a858d59b7fabe5093876a84db8e0','source':sha('/repro/i3d_adapt.py'),'wrapper':sha(__file__),'patches':{p.name:sha(p) for p in sorted(Path('/repro').glob('*.patch'))},'manifest_sha256':manifest_sha,'max_seconds':10800,'max_segments':2,'epochs':15,'batch':4,'seed':0,'selection':'final epoch','validation_rows':0}
 planfile=out/'plan.json';now=time.time()
 if (out/'complete.json').exists():raise ValueError('Already complete')
 if not planfile.exists():(out/'initial-claim').mkdir(exist_ok=False)
 if planfile.exists():
  plan=json.loads(planfile.read_text());assert plan['identity']==identity
  if not resume_after_stopped_app or resume_after_stopped_app!=plan['last_app_id']:raise ValueError('Prior app must be independently verified stopped')
  if plan['segments']>=2 or not (out/'native/checkpoint.pth.tar').exists():raise ValueError('No recovery budget/checkpoint')
 else:
  if resume_after_stopped_app:raise ValueError('Absent run')
  plan={'identity':identity,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_epoch':now+10800,'segments':0}
 if plan['deadline_epoch']-time.time()<120:raise ValueError('Original deadline reached')
 next_segment=plan['segments']+1;(out/f'segment-{next_segment}-claim').mkdir(exist_ok=False)
 plan.update(segments=next_segment,last_app_id=app.app_id);planfile.write_text(json.dumps(plan,indent=2));outputs.commit()
 segment=plan['segments'];record={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_epoch':plan['deadline_epoch'],'modal_app_id':app.app_id,'function_call_id':modal.current_function_call_id(),'segment':segment,'native_exit_code':None}
 if segment>1:
  record['resume_checkpoint_sha256']=sha(out/'native/checkpoint.pth.tar')
  record['prior_stopped_app_id']=resume_after_stopped_app
 (out/f'segment-{segment}-started.json').write_text(json.dumps(record,indent=2));outputs.commit()
 child=[None]
 def kill_child():
  if child[0] is not None:
   try:os.killpg(child[0].pid,signal.SIGKILL)
   except ProcessLookupError:pass
 def deadline_stop():
  kill_child();os._exit(124)
 timer=threading.Timer(max(1,plan['deadline_epoch']-time.time()),deadline_stop);timer.daemon=True;timer.start();stop=threading.Event()
 def commits():
  while not stop.wait(30):outputs.commit()
 thread=threading.Thread(target=commits,daemon=True);thread.start();t=time.monotonic()
 try:
  for label,cmd in [('pip-freeze',['python','-m','pip','freeze']),('hardware',['nvidia-smi'])]:
   with (out/f'segment-{segment}-{label}.txt').open('w') as f:subprocess.run(cmd,stdout=f,check=True,timeout=30)
  command=['python','/repro/i3d_adapt.py','--output',str(out),'--manifest','/outputs/phx-pseudo-full-v1/closed-training-manifest.json','--manifest-sha',manifest_sha,'--deadline',str(plan['deadline_epoch'])]
  if segment>1:command+=['--resume','--resume-sha',record['resume_checkpoint_sha256']]
  (out/f'segment-{segment}-command.json').write_text(json.dumps(command))
  with (out/f'segment-{segment}-console.log').open('w') as f:
   child[0]=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   record['native_exit_code']=child[0].wait(timeout=max(1,plan['deadline_epoch']-time.time()-60))
 except BaseException as e:record['exception']=repr(e);raise
 finally:
  kill_child()
  stop.set();thread.join(timeout=5);record.update(finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_time_seconds=time.monotonic()-t)
  if (out/'native/checkpoint.pth.tar').exists():record['terminal_checkpoint_sha256']=sha(out/'native/checkpoint.pth.tar')
  (out/f'segment-{segment}-execution.json').write_text(json.dumps(record,indent=2));outputs.commit();timer.cancel()
 print(json.dumps(record));print((out/f'segment-{segment}-console.log').read_text()[-3000:])
 if record['native_exit_code']:raise RuntimeError('Native adaptation failed; no automatic retry')

@app.function(image=image,cpu=4,memory=8192,timeout=1800,retries=0,volumes={'/datasets':data.read_only(),'/cache/huggingface':cache,'/outputs':outputs})
def collect_pseudo(run_id:str):
 import json,hashlib,datetime,time,re,collections
 from simple_video_utils.metadata import video_metadata
 if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,100}',run_id):raise ValueError('Invalid ID')
 out=Path('/outputs')/run_id;out.mkdir(exist_ok=False);base=Path('/outputs/phx-pseudo-full-v1/pseudo');t=time.monotonic()
 def sha(p):
  with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
 complete=json.loads((base/'complete.json').read_text());assert complete['ranks']==list(range(256)) and complete['identity']['mode']=='pseudo' and complete['identity']['split']=='train'
 manifest={'source_run':'phx-pseudo-full-v1','split':'train','ranks':256,'complete_sha256':sha(base/'complete.json'),'rank_receipts':[],'clips':[]};total_windows=0
 for rank in range(256):
  receipt=base/f'rank-{rank:03d}.json';r=json.loads(receipt.read_text());assert r['rank']==rank;total_windows+=r['windows'];manifest['rank_receipts'].append({'rank':rank,'sha256':sha(receipt)})
  for a in r['artifacts']:
   path=base/a['path'];assert path.is_relative_to(base) and path.stat().st_size==a['bytes'] and sha(path)==a['sha256']
   if path.suffix=='.mp4':
    frames=int(video_metadata(str(path)).nb_frames);assert frames>0;label=int(path.parent.name);assert 0<=label<5383
    manifest['clips'].append(dict(a,frames=frames,class_id=label))
 assert total_windows==720914 and len({r['path'] for r in manifest['clips']})==len(manifest['clips'])
 manifest.update(total_windows=total_windows,clip_count=len(manifest['clips']),class_counts=dict(sorted(collections.Counter(r['class_id'] for r in manifest['clips']).items())),total_bytes=sum(r['bytes'] for r in manifest['clips']))
 target=base.parent/'closed-training-manifest.json'
 encoded=(json.dumps(manifest,indent=2)+'\n').encode()
 if target.exists():assert target.read_bytes()==encoded
 else:
  temp=target.with_suffix('.partial');temp.write_bytes(encoded);temp.replace(target)
 report={'modal_app_id':app.app_id,'function_call_id':modal.current_function_call_id(),'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.monotonic()-t,'manifest_sha256':sha(target),'clips':len(manifest['clips']),'bytes':manifest['total_bytes'],'windows':total_windows}
 (out/'report.json').write_text(json.dumps(report,indent=2));outputs.commit();print(json.dumps(report))
