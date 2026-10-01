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
