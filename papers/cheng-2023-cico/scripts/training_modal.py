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

@app.function(image=native_image,cpu=4,memory=8192,timeout=1800,retries=0,volumes={'/datasets':data_volume.read_only(),'/cache/huggingface':cache_volume,'/outputs':outputs})
def csl_forecast(run_id:str):
 import json,pickle,hashlib,datetime,time,re,concurrent.futures
 from simple_video_utils.metadata import video_metadata
 if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,100}',run_id):raise ValueError('Invalid ID')
 out=Path('/outputs')/run_id;out.mkdir(exist_ok=False);t=time.monotonic()
 report={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'modal_app_id':app.app_id,'function_call_id':modal.current_function_call_id(),'purpose':'Read-only frame/window forecast, not a full data-integrity or permission audit.','splits':{}}
 files=list(Path('/datasets/csl-daily/videos').rglob('*.mp4'));paths={p.stem:p for p in files};assert len(paths)==len(files)
 for split in ['train','test']:
  label=Path('/upstream/CiCo/CLCL/data_csl')/f'{split}.pkl';labels=pickle.load(label.open('rb'));names=[v['video_name'] for group in labels.values() for v in group];assert len(names)==len(set(names)) and all(name in paths for name in names)
  def read(name):
   meta=video_metadata(str(paths[name]));frames=int(meta.nb_frames);assert frames>0
   return {'name':name,'frames':frames,'windows_16_stride1':max(1,frames-15),'bytes':paths[name].stat().st_size,'height':meta.height,'width':meta.width}
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(read,names))
  report['splits'][split]={'queries':len(labels),'videos':len(names),'label_sha256':hashlib.sha256(label.read_bytes()).hexdigest(),'frames':sum(r['frames'] for r in records),'windows_16_stride1':sum(r['windows_16_stride1'] for r in records),'bytes':sum(r['bytes'] for r in records),'records':records}
  (out/'report.json').write_text(json.dumps(report,indent=2));outputs.commit()
 report.update(finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-t);(out/'report.json').write_text(json.dumps(report,indent=2));outputs.commit();print(json.dumps({**{k:v for k,v in report.items() if k!='splits'},'splits':{k:{x:y for x,y in v.items() if x!='records'} for k,v in report['splits'].items()}}))

@app.function(image=image,cpu=4,memory=8192,timeout=1800,retries=0,volumes={'/datasets':data_volume.read_only(),'/cache/huggingface':cache_volume,'/outputs':outputs})
def collect_features(run_id:str,source_run:str,expected_weights_sha:str):
 import json,hashlib,pickle,datetime,time,re,concurrent.futures,zipfile
 import numpy as np
 if not all(re.fullmatch(r'[a-z0-9][a-z0-9-]{0,100}',x) for x in [run_id,source_run]) or not re.fullmatch(r'[0-9a-f]{64}',expected_weights_sha):raise ValueError('Invalid identity')
 out=Path('/outputs')/run_id;out.mkdir(exist_ok=False);base=Path('/outputs')/source_run;t=time.monotonic();started=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (out/'started.json').write_text(json.dumps({'started_at_utc':started,'modal_app_id':app.app_id,'function_call_id':modal.current_function_call_id()}));outputs.commit()
 def sha(p):
  with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
 root_complete=json.loads((base/'complete.json').read_text());assert root_complete['weights_sha256']==expected_weights_sha and root_complete['splits']==['train','test']
 plan=json.loads((base/'plan.json').read_text());identity=plan['identity']
 assert identity['upstream']=='38a4f7b00da7a858d59b7fabe5093876a84db8e0' and identity['source']=='b6f991eba3c3a6988c5f260208fe667c18c37ab61e9533d2a6fc82b4202ded8c' and identity['wrapper']=='ce2b2aabb707aa7700472cf9d0919a0511e6befbfaa7cca5d31a3feeee955dc6'
 assert identity['patches']==['3ae0055e9898cd74cc7fde36be67b13f2f5cd487690ba32061bc39aae94ae7ab', 'b8361b33eb885d952d1da3e72643e2f37877445de508f43cbf646838a5788cf9', 'b21034f3f84fb579c529ed306efb39c6aaf0269cb516f89c0953e9be8612cfbc'] and identity['weights_sha256']==expected_weights_sha and identity['mode']=='features' and identity['splits']==['train','test']
 split_completions={}
 rawpath=Path('/outputs/phx-raw-manifest-v1/manifest.json');assert sha(rawpath)=='4974f59634d771679d32c7b7031115506286e9dd1247ba42d41900323cb8d53a';raw=json.loads(rawpath.read_text());raw_by_name={r['path'].removesuffix('.mp4'):r for r in raw['records']};entries=[];receipts=[]
 for split in ['train','test']:
  split_completions[split]=sha(base/split/'complete.json');complete=json.loads((base/split/'complete.json').read_text());assert complete['ranks']==list(range(256 if split=='train' else 16))
  assert complete['identity']=={'mode':'features','split':split,'probe':False,'manifest_sha256':sha(rawpath),'weights_sha256':expected_weights_sha,'entrypoint_sha256':'b6f991eba3c3a6988c5f260208fe667c18c37ab61e9533d2a6fc82b4202ded8c'}
  for rank in complete['ranks']:
   receipt=base/split/f'rank-{rank:03d}.json';r=json.loads(receipt.read_text());assert r['rank']==rank;receipts.append({'split':split,'rank':rank,'sha256':sha(receipt)})
   for a in r['artifacts']:
    assert not Path(a['path']).is_absolute() and '..' not in Path(a['path']).parts and a['path'].endswith('.pkl') and Path(a['path']).parts[0]==f'rank-{rank:03d}'
    entries.append(dict(a,split=split,member=split+'/'+Path(a['path']).name))
 assert len({a['member'] for a in entries})==len(entries)==7738
 assert {a['member'].removesuffix('.pkl') for a in entries}=={k for k in raw_by_name if k.startswith(('train/','test/'))}
 def read(a):
  payload=(base/a['split']/a['path']).read_bytes();assert len(payload)==a['bytes'] and hashlib.sha256(payload).hexdigest()==a['sha256'];obj=pickle.loads(payload);features=obj['feature'];record=raw_by_name[a['member'].removesuffix('.pkl')]
  assert features.shape==(max(1,record['frames']-15),1024) and features.dtype==np.float32 and np.isfinite(features).all()
  assert Path(obj['name']).stem==Path(a['member']).stem
  return dict(a,shape=list(features.shape),dtype=str(features.dtype)),payload
 entries.sort(key=lambda a:a['member'])
 archive=out/'features.zip';partial=out/'features.zip.partial';records=[]
 with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z,concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  for start in range(0,len(entries),8):
   for record,payload in pool.map(read,entries[start:start+8]):
    z.writestr(zipfile.ZipInfo(record['member'],date_time=(1980,1,1,0,0,0)),payload);records.append(record)
 with zipfile.ZipFile(partial) as z:
  assert z.namelist()==[r['member'] for r in records]
  for r in records:
   payload=z.read(r['member']);assert len(payload)==r['bytes'] and hashlib.sha256(payload).hexdigest()==r['sha256']
 partial.replace(archive)
 manifest={'source_run':source_run,'plan_sha256':sha(base/'plan.json'),'split_completion_sha256':split_completions,'identity':identity,'weights_sha256':expected_weights_sha,'raw_manifest_sha256':sha(rawpath),'root_completion_sha256':sha(base/'complete.json'),'rank_receipts':receipts,'features':records,'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'archive_format':'ZIP_STORED; canonical native pickle bytes unchanged; deterministic member metadata.'}
 target=out/'manifest.json';target.write_text(json.dumps(manifest,indent=2)+'\n');report={'started_at_utc':started,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'modal_app_id':app.app_id,'function_call_id':modal.current_function_call_id(),'wall_seconds':time.monotonic()-t,'manifest_sha256':sha(target),'archive_sha256':manifest['archive_sha256'],'archive_bytes':manifest['archive_bytes'],'feature_files':len(records),'windows':sum(r['shape'][0] for r in records)}
 (out/'report.json').write_text(json.dumps(report,indent=2));outputs.commit();print(json.dumps(report))

@app.function(image=native_image,cpu=4,memory=8192,timeout=600,retries=0,volumes={'/datasets':data_volume.read_only(),'/cache/huggingface':cache_volume,'/outputs':outputs})
def csl_annotation_forecast(run_id:str):
 import json,pickle,hashlib,datetime,time,re,concurrent.futures
 from simple_video_utils.metadata import video_metadata
 if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,100}',run_id):raise ValueError('Invalid ID')
 out=Path('/outputs')/run_id;out.mkdir(exist_ok=False);t=time.monotonic()
 report={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'modal_app_id':app.app_id,'function_call_id':modal.current_function_call_id(),'purpose':'Canonical annotation frame-count forecast with16 actual-video consistency checks; no full data audit or trained target.','splits':{}}
 (out/'started.json').write_text(json.dumps(report));outputs.commit()
 annotation=Path('/datasets/csl-daily/sentence_label/csl2020ct_v2.pkl');payload=annotation.read_bytes();records=pickle.loads(payload)['info'];by_name={r['name']:r for r in records};assert len(by_name)==len(records)
 report['annotation_sha256']=hashlib.sha256(payload).hexdigest();report['annotation_rows']=len(records);report['annotation_length_definition']='Dataset sentence_label/README.txt defines info.length as number of video frames.'
 requested=[]
 for split in ['train','test']:
  label=Path('/upstream/CiCo/CLCL/data_csl')/f'{split}.pkl';labels=pickle.load(label.open('rb'));names=[v['video_name'] for group in labels.values() for v in group];assert len(names)==len(set(names)) and all(name in by_name for name in names)
  selected=[{'name':name,'frames':int(by_name[name]['length'])} for name in names];assert all(r['frames']>0 for r in selected);requested+=selected
  report['splits'][split]={'queries':len(labels),'videos':len(selected),'label_sha256':hashlib.sha256(label.read_bytes()).hexdigest(),'frames':sum(r['frames'] for r in selected),'windows_16_stride1':sum(max(1,r['frames']-15) for r in selected),'records':selected}
 (out/'report.json').write_text(json.dumps(report,indent=2));outputs.commit()
 sample=sorted(requested,key=lambda r:hashlib.sha256(r['name'].encode()).hexdigest())[:16]
 def check(r):
  path=Path('/datasets/csl-daily/videos')/(r['name']+'.mp4');meta=video_metadata(str(path));observed=int(meta.nb_frames);assert observed==r['frames']
  return dict(r,observed_frames=observed,bytes=path.stat().st_size)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:report['video_consistency_checks']=list(pool.map(check,sample))
 report.update(finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-t);(out/'report.json').write_text(json.dumps(report,indent=2));outputs.commit();print(json.dumps({**{k:v for k,v in report.items() if k!='splits'},'splits':{k:{x:y for x,y in v.items() if x!='records'} for k,v in report['splits'].items()}}))

@app.function(image=native_image,cpu=4,memory=8192,timeout=600,retries=0,volumes={'/datasets':data_volume.read_only(),'/cache/huggingface':cache_volume,'/outputs':outputs})
def csl_native_label_forecast(run_id:str):
 import json,pickle,hashlib,datetime,time,re,concurrent.futures
 from simple_video_utils.metadata import video_metadata
 if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,100}',run_id):raise ValueError('Invalid ID')
 out=Path('/outputs')/run_id;out.mkdir(exist_ok=False);t=time.monotonic()
 report={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'modal_app_id':app.app_id,'function_call_id':modal.current_function_call_id(),'purpose':'Pinned native-label num_frames forecast compared with canonical dataset annotation and selected actual videos; no complete data audit.','splits':{}}
 (out/'started.json').write_text(json.dumps(report));outputs.commit()
 annotation=Path('/datasets/csl-daily/sentence_label/csl2020ct_v2.pkl');payload=annotation.read_bytes();records=pickle.loads(payload)['info'];by_name={r['name']:r for r in records};report['annotation_sha256']=hashlib.sha256(payload).hexdigest();requested=[];differences=[]
 for split in ['train','test']:
  label=Path('/upstream/CiCo/CLCL/data_csl')/f'{split}.pkl';labels=pickle.load(label.open('rb'));items=[v for group in labels.values() for v in group];names=[r['video_name'] for r in items];assert len(names)==len(set(names))
  selected=[{'name':r['video_name'],'frames':int(r['num_frames'])} for r in items];assert all(r['frames']>0 for r in selected);requested+=selected
  for r in selected:
   other=by_name.get(r['name'])
   if other is None or int(other['length'])!=r['frames']:differences.append(dict(r,annotation_frames=None if other is None else int(other['length'])))
  report['splits'][split]={'queries':len(labels),'videos':len(selected),'label_sha256':hashlib.sha256(label.read_bytes()).hexdigest(),'frames':sum(r['frames'] for r in selected),'windows_16_stride1':sum(max(1,r['frames']-15) for r in selected),'records':selected}
 report['annotation_differences']=differences;(out/'report.json').write_text(json.dumps(report,indent=2));outputs.commit()
 sample={r['name']:r for r in sorted(requested,key=lambda r:hashlib.sha256(r['name'].encode()).hexdigest())[:16]}
 sample.update({r['name']:r for r in differences})
 def check(r):
  path=Path('/datasets/csl-daily/videos')/(r['name']+'.mp4');meta=video_metadata(str(path));observed=int(meta.nb_frames)
  return dict(r,observed_frames=observed,bytes=path.stat().st_size,native_frame_count_matches=observed==r['frames'])
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:report['video_consistency_checks']=list(pool.map(check,sample.values()))
 report.update(finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-t);(out/'report.json').write_text(json.dumps(report,indent=2));outputs.commit();print(json.dumps({**{k:v for k,v in report.items() if k!='splits'},'splits':{k:{x:y for x,y in v.items() if x!='records'} for k,v in report['splits'].items()}}))

@app.function(image=native_image,cpu=4,memory=8192,timeout=1200,retries=0,volumes={'/datasets':data_volume.read_only(),'/cache/huggingface':cache_volume,'/outputs':outputs})
def csl_direct_frame_forecast(run_id:str):
 import json,pickle,hashlib,datetime,time,re,concurrent.futures
 from simple_video_utils.metadata import video_metadata
 if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,100}',run_id):raise ValueError('Invalid ID')
 out=Path('/outputs')/run_id;out.mkdir(exist_ok=False);t=time.monotonic();report={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'modal_app_id':app.app_id,'function_call_id':modal.current_function_call_id(),'purpose':'Actual metadata via native-label-derived paths, no directory traversal; full count forecast only, not content-hash data audit.','splits':{}}
 (out/'started.json').write_text(json.dumps(report));outputs.commit()
 for split in ['train','test']:
  label=Path('/upstream/CiCo/CLCL/data_csl')/f'{split}.pkl';labels=pickle.load(label.open('rb'));items=[v for group in labels.values() for v in group];assert len({r['video_name'] for r in items})==len(items);records=[]
  def read(r):
   path=Path('/datasets/csl-daily/videos')/(r['video_name']+'.mp4');meta=video_metadata(str(path));frames=int(meta.nb_frames);assert frames>0
   return {'name':r['video_name'],'frames':frames,'label_num_frames':r['num_frames'],'bytes':path.stat().st_size,'height':meta.height,'width':meta.width}
  with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
   for start in range(0,len(items),256):
    records.extend(pool.map(read,items[start:start+256]));(out/'progress.json').write_text(json.dumps({'split':split,'completed':len(records),'total':len(items),'wall_seconds':time.monotonic()-t}));outputs.commit()
  report['splits'][split]={'queries':len(labels),'videos':len(records),'label_sha256':hashlib.sha256(label.read_bytes()).hexdigest(),'frames':sum(r['frames'] for r in records),'windows_16_stride1':sum(max(1,r['frames']-15) for r in records),'bytes':sum(r['bytes'] for r in records),'records':records};(out/'report.json').write_text(json.dumps(report,indent=2));outputs.commit()
 report.update(finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-t);(out/'report.json').write_text(json.dumps(report,indent=2));outputs.commit();print(json.dumps({**{k:v for k,v in report.items() if k!='splits'},'splits':{k:{x:y for x,y in v.items() if x!='records'} for k,v in report['splits'].items()}}))
