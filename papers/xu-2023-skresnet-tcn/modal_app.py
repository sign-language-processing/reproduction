"""Inventory and decode real LSA64 data; no invented model or target score."""
from pathlib import Path
import modal
app=modal.App('2be3c68e-skresnet-data-audit')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('simple-video-utils==0.7.4','av==18.0.0')
       .env({'HF_HOME':'/cache/huggingface','HF_HUB_CACHE':'/cache/huggingface/hub'}))
data=modal.Volume.from_name('datasets',version=2)
cache=modal.Volume.from_name('huggingface-cache',version=2)
outputs=modal.Volume.from_name('2be3c68e-skresnet-results',create_if_missing=True,version=2)
@app.function(image=image,cpu=2,memory=4096,timeout=900,volumes={'/datasets':data.with_mount_options(read_only=True),'/cache/huggingface':cache,'/outputs':outputs})
def audit(run_id:str):
    import json,hashlib,datetime,subprocess,collections,time,dataclasses
    from simple_video_utils.frames import read_frames_exact
    from simple_video_utils.metadata import video_metadata
    out=Path('/outputs')/run_id;out.mkdir(exist_ok=False)
    started=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic()
    root=Path('/datasets/lsa64');manifest=root/'manifest.json'
    files=sorted(root.rglob('*.mp4'));counts=collections.Counter(p.stem.split('_')[0] for p in files)
    assert len(files)==3200 and len(counts)==64,(len(files),len(counts))
    selected={p.stem.split('_')[0]:p for p in files};samples=[]
    for p in selected.values():
        meta=video_metadata(str(p));frames=list(read_frames_exact(str(p),start_frame=0,end_frame=2))
        assert len(frames)==3
        samples.append(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),metadata={k:getattr(meta,k) for k in ['width','height','fps','nb_frames','duration']},decoded_frames=len(frames),decoded_shape=list(frames[0].shape)))
    result=dict(scope='Data-only preflight; no model or classification evaluation is possible without the architecture.',video_count=len(files),class_counts=dict(counts),manifest=json.loads(manifest.read_text()),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),samples=samples,elapsed_seconds=time.monotonic()-t)
    (out/'audit.json').write_text(json.dumps(result,indent=2));(out/'freeze.txt').write_text(subprocess.check_output(['python','-m','pip','freeze'],text=True))
    execution=dict(started_at_utc=started,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=0,app_id=app.app_id,function_call_id=modal.current_function_call_id(),gpu_count=0)
    (out/'execution.json').write_text(json.dumps(execution,indent=2));outputs.commit();print(json.dumps(execution));print('Verified3200 videos,64classes,192decodedframes.')
@app.local_entrypoint()
def main(run_id:str='data-preflight-004'):audit.remote(run_id)
