from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parent
app=modal.App('repro-2248c066-semantic-asl')
data=modal.Volume.from_name('datasets',version=2)
cache=modal.Volume.from_name('huggingface-cache',version=2)
outputs=modal.Volume.from_name('repro-2248c066-results',create_if_missing=True,version=2)
ENV={'HF_HOME':'/cache/huggingface','HF_HUB_CACHE':'/cache/huggingface/hub'}
data_image=(modal.Image.debian_slim(python_version='3.11').env(ENV).add_local_file(ROOT/'data.sh','/app/data.sh'))
@app.function(image=data_image,timeout=1800,volumes={'/datasets':data,'/cache/huggingface':cache})
def populate():
 import subprocess
 subprocess.run(['bash','/app/data.sh'],check=True)
 data.commit()

image=(modal.Image.from_registry('ghcr.io/sign-language-processing/reproduction:latest')
       .env(ENV).add_local_file(ROOT/'train.py','/app/train.py'))
@app.function(image=image,gpu='A100-80GB',timeout=1800,memory=16384,volumes={'/datasets':data.read_only(),'/cache/huggingface':cache,'/results':outputs})
def preflight():
 import subprocess
 subprocess.run(['python','/app/train.py','--dataset','mnist','--preflight'],check=True)
 subprocess.run(['python','/app/train.py','--dataset','rgb','--preflight'],check=True)
 outputs.commit()
@app.function(image=image,gpu='A100-80GB',timeout=21600,memory=16384,volumes={'/datasets':data.read_only(),'/cache/huggingface':cache,'/results':outputs})
def train(dataset: str):
 import subprocess
 subprocess.run(['python','/app/train.py','--dataset',dataset],check=True)
 outputs.commit()

@app.function(image=data_image,timeout=300,volumes={'/results':outputs.read_only(),'/cache/huggingface':cache})
def evidence():
 import hashlib,json
 from datetime import datetime,timezone
 started=datetime.now(timezone.utc).isoformat()
 records=[]
 for p in sorted(Path('/results').rglob('*')):
  if p.is_file():
   item={'path':str(p.relative_to('/results')),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}
   if p.name=='metrics.json':item['metrics']=json.loads(p.read_text())
   records.append(item)
 print(json.dumps(records))
 print(json.dumps({'audit_started_at_utc':started,'audit_finished_at_utc':datetime.now(timezone.utc).isoformat()}))

lexset_data_image=(modal.Image.debian_slim(python_version='3.11').pip_install('Pillow==11.1.0').env(ENV)
 .add_local_file(ROOT/'data.sh','/app/data.sh').add_local_file(ROOT/'lexset_data.py','/app/lexset_data.py'))
@app.function(image=lexset_data_image,cpu=4,memory=8192,timeout=3600,retries=0,volumes={'/datasets':data,'/cache/huggingface':cache,'/results':outputs})
def populate_lexset(run_id: str="lexset-acquisition-v1"):
 import subprocess,json,os,time
 from datetime import datetime,timezone
 out=Path('/results')/run_id;out.mkdir(parents=True,exist_ok=True)
 receipt={'started_at_utc':datetime.now(timezone.utc).isoformat(),'modal_app_id':app.app_id,'modal_function_call_id':modal.current_function_call_id(),'modal_task_id':os.environ.get('MODAL_TASK_ID')}
 (out/'execution-start.json').write_text(json.dumps(receipt,indent=2));outputs.commit()
 start=time.monotonic()
 with (out/'console.log').open('w') as f:
  p=subprocess.run(['bash','/app/data.sh','lexset'],stdout=f,stderr=subprocess.STDOUT,timeout=3500)
 receipt.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),exit_code=p.returncode,wall_time_seconds=time.monotonic()-start)
 (out/'execution.json').write_text(json.dumps(receipt,indent=2));outputs.commit();print(json.dumps(receipt));print((out/'console.log').read_text()[-12000:]);p.check_returncode()

lexset_image=(modal.Image.from_registry('ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291')
 .env(ENV).add_local_file(ROOT/'train.py','/app/train.py').add_local_file(ROOT/'modal_app.py','/app/lexset_modal_source.py'))
@app.function(image=lexset_image,gpu='A100-80GB',cpu=8,memory=16384,timeout=18000,retries=0,volumes={'/datasets':data.read_only(),'/cache/huggingface':cache,'/results':outputs})
def lexset_run(run_id: str, expected_manifest_sha256: str, preflight: bool=False):
 import subprocess,json,os,time,hashlib,re,threading
 from datetime import datetime,timezone
 assert re.fullmatch(r'lexset-(?:preflight|full)-[a-z0-9-]+',run_id), 'Invalid output identifier'
 out=Path('/results')/run_id;out.mkdir(parents=True,exist_ok=True)
 claim_path=out/'claim.json'
 now=time.time();call_id=modal.current_function_call_id()
 manifest=Path('/datasets/synthetic-asl-alphabet/manifest.json')
 manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest()
 assert manifest_hash==expected_manifest_sha256,'Audited manifest mismatch'
 wrapper_hash=hashlib.sha256(Path('/app/lexset_modal_source.py').read_bytes()).hexdigest()
 source_hash=hashlib.sha256(Path('/app/train.py').read_bytes()).hexdigest()
 max_seconds=1200 if preflight else 18000
 if claim_path.exists():
  claim=json.loads(claim_path.read_text())
  assert claim['function_call_id']==call_id and claim['manifest_sha256']==manifest_hash and claim['source_sha256']==source_hash
  assert claim['wrapper_sha256']==wrapper_hash and claim['preflight']==preflight and claim['max_seconds']==max_seconds
  assert claim['segments']<2 and time.time()<claim['deadline_unix']
  assert (out/'checkpoint.pt').exists(),'Never restart full training without checkpoint'
  claim['segments']+=1
 else:
  claim={'function_call_id':call_id,'started_at_utc':datetime.now(timezone.utc).isoformat(),'deadline_unix':now+max_seconds,'segments':1,'manifest_sha256':manifest_hash,'source_sha256':source_hash,'wrapper_sha256':wrapper_hash,'preflight':preflight,'max_seconds':max_seconds}
 watchdog=threading.Timer(max(0.01,claim['deadline_unix']-time.time()),lambda:os._exit(124));watchdog.daemon=True;watchdog.start()
 claim_path.write_text(json.dumps(claim,indent=2));outputs.commit()
 if (out/'metrics.json').exists():
  watchdog.cancel();return json.loads((out/'metrics.json').read_text())
 env=dict(os.environ,REQUIRE_CHECKPOINT='1' if claim['segments']>1 else '0')
 command=['python','/app/train.py','--dataset','lexset','--run-id',run_id]+(['--preflight'] if preflight else [])
 execution={'started_at_utc':datetime.now(timezone.utc).isoformat(),'modal_app_id':app.app_id,'modal_function_call_id':call_id,'modal_task_id':os.environ.get('MODAL_TASK_ID'),'modal_image_id':os.environ.get('MODAL_IMAGE_ID'),'command':command,'segment':claim['segments'],'exit_code':None}
 try:
  with (out/f"console-segment-{claim['segments']}.log").open('w') as f:
   p=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,claim['deadline_unix']-time.time()-60))
  execution['exit_code']=p.returncode
 except subprocess.TimeoutExpired:
  execution['exit_code']=124
 finally:
  execution.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),wall_time_seconds=time.time()-now)
  (out/f"execution-segment-{claim['segments']}.json").write_text(json.dumps(execution,indent=2));outputs.commit()
 watchdog.cancel()
 print(json.dumps(execution));print((out/f"console-segment-{claim['segments']}.log").read_text()[-3000:])
 if execution['exit_code']:raise RuntimeError(f"Training exit {execution['exit_code']}")
 return json.loads((out/'metrics.json').read_text())
