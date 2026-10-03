"""Bounded native CLCL training preflight using true global batch512."""
from pathlib import Path
import modal
HERE=Path(__file__).resolve().parent
app=modal.App('repro-cico-clcl-training')
data=modal.Volume.from_name('datasets',version=2)
cache=modal.Volume.from_name('huggingface-cache',version=2)
outputs=modal.Volume.from_name('cheng-2023-cico-results',version=2)
GPU_BASE="ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291"
REV="38a4f7b00da7a858d59b7fabe5093876a84db8e0"
image = (
    modal.Image.from_registry(GPU_BASE)
    .apt_install("git")
    .pip_install(
        "ftfy==6.3.1",
        "regex==2024.11.6",
        "boto3==1.35.99",
        "nltk==3.9.1",
        "textaugment==2.0.0",
        "textblob==0.17.1",
        "gensim==4.4.0",
        "opencv-python-headless==4.11.0.86",
    )
    .run_commands(
        f"git clone https://github.com/FangyunWei/SLRT.git /upstream && cd /upstream && git checkout {REV}"
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface/hub",
            "OMP_NUM_THREADS": "4",
        }
    )
)
for name in ['0010-clcl-phoenix-aware-path.patch','0011-clcl-atomic-recovery-and-selection.patch','0012-clcl-native-initialization-audit.patch']:
 image=image.add_local_file(HERE.parent/'patches'/name,'/repro/'+name,copy=True).run_commands('cd /upstream && git apply /repro/'+name)
image=image.add_local_file(HERE/'clcl_train.py','/repro/clcl_train.py')
@app.function(image=image,gpu='A100-80GB',cpu=4,memory=32768,timeout=10800,retries=0,volumes={'/datasets':data.read_only(),'/cache/huggingface':cache,'/outputs':outputs})
def run(run_id:str,agnostic_manifest:str,agnostic_sha:str,aware_manifest:str,aware_sha:str,preflight:bool=False,resume_after_stopped_app:str=''):
 import subprocess,json,datetime,time,re,threading,os,signal,hashlib
 if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,100}',run_id):raise ValueError('Invalid run ID')
 for path,expected in [(agnostic_manifest,agnostic_sha),(aware_manifest,aware_sha)]:
  if not path.startswith('/outputs/') or '..' in Path(path).parts or not re.fullmatch(r'[0-9a-f]{64}',expected):raise ValueError('Invalid input identity')
 def sha(p):
  with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
 out=Path('/outputs')/run_id;out.mkdir(exist_ok=True);max_seconds=1800 if preflight else 10800;max_segments=2 if preflight else 3
 identity={'upstream':REV,'source':sha('/repro/clcl_train.py'),'wrapper':sha(__file__),'patches':{p.name:sha(p) for p in sorted(Path('/repro').glob('*.patch'))},'agnostic_manifest':agnostic_manifest,'agnostic_sha':agnostic_sha,'aware_manifest':aware_manifest,'aware_sha':aware_sha,'preflight':preflight,'max_seconds':max_seconds,'max_segments':max_segments,'epochs':200,'batch':512,'accumulation':1,'seed':42,'sampler_seed':0,'alpha':.9}
 planfile=out/'plan.json'
 if (out/'complete.json').exists() or (preflight and (out/'preflight-resumed.json').exists()):raise ValueError('Already complete')
 if not planfile.exists():(out/'initial-claim').mkdir(exist_ok=False)
 if planfile.exists():
  plan=json.loads(planfile.read_text());assert plan['identity']==identity
  if not resume_after_stopped_app or resume_after_stopped_app!=plan['last_app_id']:raise ValueError('Prior app must be independently verified stopped')
  if plan['segments']>=max_segments or not (out/'native/last-full-state.pt').exists():raise ValueError('No recovery allowance/checkpoint')
 else:
  if resume_after_stopped_app:raise ValueError('Absent run')
  plan={'identity':identity,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_epoch':time.time()+max_seconds,'segments':0}
 if plan['deadline_epoch']-time.time()<120:raise ValueError('Immutable deadline reached')
 segment=plan['segments']+1;(out/f'segment-{segment}-claim').mkdir(exist_ok=False)
 plan.update(segments=segment,last_app_id=app.app_id);planfile.write_text(json.dumps(plan,indent=2));outputs.commit()
 record={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_epoch':plan['deadline_epoch'],'modal_app_id':app.app_id,'function_call_id':modal.current_function_call_id(),'segment':segment,'driver_exit_code':None,'native_training_exit_code':None}
 if segment>1:record.update(resume_checkpoint_sha256=sha(out/'native/last-full-state.pt'),prior_stopped_app_id=resume_after_stopped_app)
 (out/f'segment-{segment}-started.json').write_text(json.dumps(record,indent=2));outputs.commit();child=[None];t=time.monotonic()
 def kill_child():
  if child[0] is not None:
   try:os.killpg(child[0].pid,signal.SIGKILL)
   except ProcessLookupError:pass
 def deadline_stop():
  kill_child();os._exit(124)
 timer=threading.Timer(max(1,plan['deadline_epoch']-time.time()),deadline_stop);timer.daemon=True;timer.start();stop=threading.Event()
 def commits():
  while not stop.wait(30):outputs.commit()
 thread=threading.Thread(target=commits,daemon=True);thread.start()
 try:
  for name,command in [('pip-freeze',['python','-m','pip','freeze']),('hardware',['nvidia-smi'])]:
   with (out/f'segment-{segment}-{name}.txt').open('w') as f:subprocess.run(command,stdout=f,check=True,timeout=30)
  command=['python','/repro/clcl_train.py','--output',str(out),'--agnostic-manifest',agnostic_manifest,'--agnostic-sha',agnostic_sha,'--aware-manifest',aware_manifest,'--aware-sha',aware_sha,'--deadline',str(plan['deadline_epoch'])]
  if preflight:command+=['--preflight']
  if segment>1:command+=['--resume','--resume-sha',record['resume_checkpoint_sha256']]
  (out/f'segment-{segment}-command.json').write_text(json.dumps(command))
  if (out/'native-execution.json').exists():(out/'native-execution.json').unlink()
  with (out/f'segment-{segment}-console.log').open('w') as f:
   child[0]=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);record['driver_exit_code']=child[0].wait(timeout=max(1,plan['deadline_epoch']-time.time()-60))
  report_path=out/('preflight-resumed.json' if preflight and segment>1 else 'preflight.json' if preflight else 'complete.json')
  if record['driver_exit_code']==0:
   report=json.loads(report_path.read_text());record['completion_only']=report['completion_only'];record['native_training_exit_code']=None if report['completion_only'] else 0
 except BaseException as e:record['exception']=repr(e);raise
 finally:
  kill_child();stop.set();thread.join(timeout=5);record.update(finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_time_seconds=time.monotonic()-t)
  if (out/'native-execution.json').exists():record['native_training_exit_code']=json.loads((out/'native-execution.json').read_text())['native_exit_code']
  if (out/'native/last-full-state.pt').exists():record['terminal_checkpoint_sha256']=sha(out/'native/last-full-state.pt')
  (out/f'segment-{segment}-execution.json').write_text(json.dumps(record,indent=2));outputs.commit();timer.cancel()
 print(json.dumps(record))
 if record['driver_exit_code']!=0:raise RuntimeError('Native CLCL failed; no automatic restart')
