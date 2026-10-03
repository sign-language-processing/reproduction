"""Guarded training preparation; all invocations use the repro-sign wrapper."""
from pathlib import Path
import modal
HERE=Path(__file__).resolve().parent
app=modal.App('repro-cico-training-18c49909')
data_volume=modal.Volume.from_name('datasets',version=2)
cache_volume=modal.Volume.from_name('huggingface-cache',version=2)
outputs=modal.Volume.from_name('cheng-2023-cico-results',version=2)
image=(modal.Image.from_registry('ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291')
 .apt_install('git').pip_install('gdown==5.2.0')
 .run_commands('git clone https://github.com/FangyunWei/SLRT.git /upstream && cd /upstream && git checkout 38a4f7b00da7a858d59b7fabe5093876a84db8e0')
 .env({'HF_HOME':'/cache/huggingface','HF_HUB_CACHE':'/cache/huggingface/hub'})
 .add_local_file(HERE/'training_prepare.py','/repro/training_prepare.py'))
@app.function(image=image,cpu=4,memory=16384,timeout=3600,retries=0,volumes={'/datasets':data_volume.read_only(),'/cache/huggingface':cache_volume,'/outputs':outputs})
def prepare(run_id:str):
 import subprocess,json,datetime,time,os,re
 if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,100}',run_id):raise ValueError('Invalid run ID')
 out=Path('/outputs')/run_id;out.mkdir(exist_ok=False)
 record={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'modal_app_id':app.app_id,'function_call_id':modal.current_function_call_id(),'modal_task_id':os.environ.get('MODAL_TASK_ID'),'native_exit_code':None}
 (out/'started.json').write_text(json.dumps(record,indent=2));outputs.commit();start=time.monotonic()
 try:
  with (out/'console.log').open('w') as f:
   p=subprocess.run(['python','/repro/training_prepare.py'],stdout=f,stderr=subprocess.STDOUT,timeout=3500)
  record['native_exit_code']=p.returncode
 finally:
  record.update(finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_time_seconds=time.monotonic()-start)
  (out/'execution.json').write_text(json.dumps(record,indent=2));outputs.commit()
 print(json.dumps(record));print((out/'console.log').read_text()[-12000:])
 if record['native_exit_code']:raise RuntimeError('Preparation failed; inspect preserved native record')

native_image=(modal.Image.from_registry('ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291')
 .apt_install('git').pip_install('opencv-python-headless==4.11.0.86','beartype==0.19.0','simple-video-utils==0.0.6','mock==5.1.0','humanize==4.11.0','tensorboard==2.18.0')
 .run_commands('git clone https://github.com/FangyunWei/SLRT.git /upstream && cd /upstream && git checkout 38a4f7b00da7a858d59b7fabe5093876a84db8e0')
 .add_local_file(HERE.parent/'patches/0001-extractor-video-decoder.patch','/repro/decoder.patch',copy=True)
 .run_commands('cd /upstream && git apply /repro/decoder.patch')
 .env({'HF_HOME':'/cache/huggingface','HF_HUB_CACHE':'/cache/huggingface/hub','OMP_NUM_THREADS':'4'})
)
probe_image=native_image.add_local_file(HERE/'i3d_probe.py','/repro/i3d_probe.py')
@app.function(image=probe_image,gpu='A100-80GB',cpu=4,memory=32768,timeout=900,retries=0,volumes={'/datasets':data_volume.read_only(),'/cache/huggingface':cache_volume,'/outputs':outputs})
def i3d_probe(run_id:str):
 import subprocess,json,datetime,time,os,re,threading
 if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,100}',run_id):raise ValueError('Invalid run ID')
 out=Path('/outputs')/run_id;out.mkdir(exist_ok=False)
 record={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'modal_app_id':app.app_id,'function_call_id':modal.current_function_call_id(),'modal_task_id':os.environ.get('MODAL_TASK_ID'),'native_exit_code':None}
 (out/'started.json').write_text(json.dumps(record,indent=2));outputs.commit();start=time.monotonic()
 timer=threading.Timer(890,lambda:os._exit(124));timer.daemon=True;timer.start()
 try:
  with (out/'console.log').open('w') as f:
   p=subprocess.run(['python','/repro/i3d_probe.py',str(out)],stdout=f,stderr=subprocess.STDOUT,timeout=820)
  record['native_exit_code']=p.returncode
 finally:
  record.update(finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_time_seconds=time.monotonic()-start)
  (out/'execution.json').write_text(json.dumps(record,indent=2));outputs.commit();timer.cancel()
 print(json.dumps(record));print((out/'console.log').read_text()[-10000:])
 if record['native_exit_code']:raise RuntimeError('I3D probe failed; inspect native record')

audit_image=probe_image.add_local_file(HERE/'decoder_audit.py','/repro/decoder_audit.py')
@app.function(image=audit_image,cpu=2,memory=4096,timeout=300,retries=0,volumes={'/datasets':data_volume.read_only(),'/cache/huggingface':cache_volume,'/outputs':outputs})
def decoder_audit(run_id:str):
 import subprocess,json,datetime,time,re
 if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,100}',run_id):raise ValueError('Invalid run ID')
 out=Path('/outputs')/run_id;out.mkdir(exist_ok=False);t=time.monotonic()
 record={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'modal_app_id':app.app_id,'function_call_id':modal.current_function_call_id(),'native_exit_code':None}
 try:
  with (out/'console.log').open('w') as f:p=subprocess.run(['python','/repro/decoder_audit.py',str(out/'report.json')],stdout=f,stderr=subprocess.STDOUT,timeout=250)
  record['native_exit_code']=p.returncode
 finally:
  record.update(finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_time_seconds=time.monotonic()-t);(out/'execution.json').write_text(json.dumps(record,indent=2));outputs.commit()
 print(json.dumps(record));print((out/'console.log').read_text())
 if record['native_exit_code']:raise RuntimeError('Decoder audit failed')

manifest_image=probe_image.add_local_file(HERE/'phx_manifest.py','/repro/phx_manifest.py')
@app.function(image=manifest_image,cpu=4,memory=8192,timeout=1800,retries=0,volumes={'/datasets':data_volume.read_only(),'/cache/huggingface':cache_volume,'/outputs':outputs})
def phx_manifest(run_id:str):
 import subprocess,json,datetime,time,re,hashlib
 if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,100}',run_id):raise ValueError('Invalid run ID')
 out=Path('/outputs')/run_id;out.mkdir(exist_ok=False);t=time.monotonic()
 record={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'modal_app_id':app.app_id,'function_call_id':modal.current_function_call_id(),'native_exit_code':None}
 try:
  with (out/'console.log').open('w') as f:p=subprocess.run(['python','/repro/phx_manifest.py',str(out/'manifest.json')],stdout=f,stderr=subprocess.STDOUT,timeout=1700)
  record['native_exit_code']=p.returncode
 finally:
  record.update(finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_time_seconds=time.monotonic()-t)
  if (out/'manifest.json').exists():record['manifest_sha256']=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()
  (out/'execution.json').write_text(json.dumps(record,indent=2));outputs.commit()
 print(json.dumps(record));print((out/'console.log').read_text()[-5000:])
 if record['native_exit_code']:raise RuntimeError('PHOENIX manifest failed')

extract_image=(native_image
 .add_local_file(HERE.parent/'patches/0002-pseudo-video-decoder.patch','/repro/pseudo-decoder.patch',copy=True)
 .add_local_file(HERE.parent/'patches/0003-pseudo-range-concatenation.patch','/repro/pseudo-range.patch',copy=True)
 .run_commands('cd /upstream && git apply /repro/pseudo-decoder.patch && git apply /repro/pseudo-range.patch')
 .add_local_file(HERE/'i3d_extract.py','/repro/i3d_extract.py'))
@app.function(image=extract_image,gpu='A100-80GB',cpu=4,memory=32768,timeout=900,retries=0,volumes={'/datasets':data_volume.read_only(),'/cache/huggingface':cache_volume,'/outputs':outputs})
def pseudo_probe(run_id:str,mode:str="pseudo"):
 import subprocess,json,datetime,time,re,threading,os
 if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,100}',run_id):raise ValueError('Invalid run ID')
 if mode not in ['pseudo','features']:raise ValueError('Unknown mode')
 out=Path('/outputs')/run_id;out.mkdir(exist_ok=False);t=time.monotonic()
 record={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'modal_app_id':app.app_id,'function_call_id':modal.current_function_call_id(),'native_exit_code':None}
 (out/'started.json').write_text(json.dumps(record,indent=2));outputs.commit()
 timer=threading.Timer(890,lambda:os._exit(124));timer.daemon=True;timer.start()
 try:
  command=['python','/repro/i3d_extract.py','--output',str(out/mode),'--manifest','/outputs/phx-raw-manifest-v1/manifest.json','--manifest-sha','4974f59634d771679d32c7b7031115506286e9dd1247ba42d41900323cb8d53a','--weights','/outputs/training-inputs/bsl5k.pth.tar','--weights-sha','6430592464a357dfdaa7f31973cb684663237655fdf23f3999608d162167fc6f','--mode',mode,'--split','train','--probe']
  with (out/'console.log').open('w') as f:
   p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=240)
   if p.returncode==0:
    receipt=out/mode/'rank-255.json';original=json.loads(receipt.read_text())
    # Closed-rank replay verifies each artifact without recomputing it.
    first_mtime=receipt.stat().st_mtime_ns
    p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=240)
    assert p.returncode==0 and receipt.stat().st_mtime_ns==first_mtime
    # Simulate interruption after final-directory rename but before receipt.
    receipt.rename(out/mode/'rank-255.saved')
    p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=240)
    assert p.returncode==0
    recovered=json.loads(receipt.read_text());assert original['artifacts']==recovered['artifacts']
    (out/'recovery-proof.json').write_text(json.dumps({'closed_rank_skipped':True,'incomplete_rank_regenerated_identically':True,'artifact_count':len(recovered['artifacts'])},indent=2))
  record['native_exit_code']=p.returncode
 finally:
  record.update(finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_time_seconds=time.monotonic()-t);(out/'execution.json').write_text(json.dumps(record,indent=2));outputs.commit();timer.cancel()
 print(json.dumps(record));print((out/'console.log').read_text()[-10000:])
 if record['native_exit_code']:raise RuntimeError('Pseudo probe failed')

@app.function(image=image,cpu=2,memory=4096,timeout=900,retries=0,volumes={'/datasets':data_volume.read_only(),'/cache/huggingface':cache_volume,'/outputs':outputs})
def prepare_clip(run_id:str):
 import urllib.request,hashlib,json,time,datetime,re
 if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,100}',run_id):raise ValueError('Invalid ID')
 out=Path('/outputs')/run_id;out.mkdir(exist_ok=False);started=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic()
 expected='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af';url=f'https://openaipublic.azureedge.net/clip/models/{expected}/ViT-B-32.pt';target=Path('/outputs/training-inputs/ViT-B-32.pt')
 def sha(p):
  with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
 if not target.exists() or sha(target)!=expected:
  partial=target.with_suffix('.partial')
  with urllib.request.urlopen(url,timeout=120) as response,partial.open('wb') as f:
   while block:=response.read(8*1024*1024):f.write(block)
  assert sha(partial)==expected;partial.replace(target)
 report={'started_at_utc':started,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'modal_app_id':app.app_id,'function_call_id':modal.current_function_call_id(),'wall_seconds':time.monotonic()-t,'sha256':sha(target),'bytes':target.stat().st_size,'url':url,'permission':'Published OpenAI CLIP initialization used under upstream MIT license; no trained CiCo retrieval checkpoint used.'}
 (out/'report.json').write_text(json.dumps(report,indent=2));outputs.commit();print(json.dumps(report))
