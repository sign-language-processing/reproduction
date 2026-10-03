"""One bounded CPU-only architecture count audit."""
from pathlib import Path
import modal
cache = modal.Volume.from_name("huggingface-cache", version=2)
outputs = modal.Volume.from_name("2be3c68e-skresnet-results", version=2)
gpu_image = (modal.Image.from_registry("ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291")
.pip_install("timm==1.0.22", "torch-optimizer==0.3.0", "fvcore==0.1.5.post20221221")
.run_commands("git clone https://github.com/locuslab/TCN.git /opt/TCN && cd /opt/TCN && git checkout 2f8c2b817050206397458dfd1f5a25ce8a32fe65")
.env({"HF_HOME":"/cache/huggingface","HF_HUB_CACHE":"/cache/huggingface/hub"}))
app = modal.App("2be3c68e-skresnet-size-audit")
image = gpu_image.add_local_file(Path(__file__).with_name("size_audit.py"), "/opt/size_audit.py")
@app.function(image=image,cpu=4,memory=16384,timeout=900,retries=0,volumes={"/cache/huggingface":cache,"/outputs":outputs})
def audit():
    import subprocess,datetime,json,hashlib
    out=Path("/outputs/paper-sized-audit-002")
    out.mkdir(exist_ok=False)
    start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    command=["python","/opt/size_audit.py",str(out/"counts.json")]
    with (out/"console.log").open("w") as log:
        code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=780).returncode
    record=dict(started_at_utc=start,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=code,command=command,app_id=app.app_id,function_call_id=modal.current_function_call_id(),source_sha256=hashlib.sha256(Path("/opt/size_audit.py").read_bytes()).hexdigest())
    (out/"execution.json").write_text(json.dumps(record,indent=2))
    outputs.commit()
    print(json.dumps(record))
    return record
@app.local_entrypoint()
def main():
    result=audit.remote()
    if result["exit_code"]: raise SystemExit(result["exit_code"])
