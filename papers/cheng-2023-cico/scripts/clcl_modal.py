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
image=image.add_local_file(HERE/'clcl_probe.py','/repro/clcl_probe.py')
@app.function(image=image,gpu='A100-80GB',cpu=4,memory=32768,timeout=900,retries=0,volumes={'/datasets':data.read_only(),'/cache/huggingface':cache,'/outputs':outputs})
def probe(run_id:str):
 import subprocess,json,datetime,time,re,threading,os,signal
 if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,100}',run_id):raise ValueError('Invalid run ID')
 out=Path('/outputs')/run_id;out.mkdir(exist_ok=False);t=time.monotonic()
 record={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'modal_app_id':app.app_id,'function_call_id':modal.current_function_call_id(),'native_exit_code':None}
 (out/'started.json').write_text(json.dumps(record,indent=2));outputs.commit();child=[None]
 def kill_child():
  if child[0] is not None:
   try:os.killpg(child[0].pid,signal.SIGKILL)
   except ProcessLookupError:pass
 def deadline_stop():
  kill_child();os._exit(124)
 timer=threading.Timer(890,deadline_stop);timer.daemon=True;timer.start()
 try:
  with (out/'console.log').open('w') as f:
   child[0]=subprocess.Popen(['python','/repro/clcl_probe.py',str(out)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   record['native_exit_code']=child[0].wait(timeout=820)
 finally:
  kill_child();record.update(finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_time_seconds=time.monotonic()-t);(out/'execution.json').write_text(json.dumps(record,indent=2));outputs.commit();timer.cancel()
 print(json.dumps(record));print((out/'console.log').read_text()[-10000:])
 if record['native_exit_code']!=0:raise RuntimeError('Native CLCL mechanics failed')
