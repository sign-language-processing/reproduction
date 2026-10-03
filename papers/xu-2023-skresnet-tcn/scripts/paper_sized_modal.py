"""Guarded bounded execution of the predeclared parameter-sized reconstruction."""
from pathlib import Path
import modal

app = modal.App("2be3c68e-skresnet-paper-sized")
data = modal.Volume.from_name("datasets", version=2)
cache = modal.Volume.from_name("huggingface-cache", version=2)
outputs = modal.Volume.from_name("2be3c68e-skresnet-results", version=2)
image = (modal.Image.from_registry("ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291")
    .pip_install("timm==1.0.22", "torch-optimizer==0.3.0", "fvcore==0.1.5.post20221221")
    .run_commands("git clone https://github.com/locuslab/TCN.git /opt/TCN && cd /opt/TCN && git checkout 2f8c2b817050206397458dfd1f5a25ce8a32fe65")
    .env({"HF_HOME":"/cache/huggingface", "HF_HUB_CACHE":"/cache/huggingface/hub"})
    .add_local_file(Path(__file__).with_name("conditional.py"), "/opt/conditional.py")
    .add_local_file(Path(__file__), "/opt/paper_sized_modal.py"))

@app.function(image=image, gpu="A100-80GB", cpu=8, memory=65536,
    timeout=18000, retries=0, max_containers=1,
    volumes={"/datasets":data.with_mount_options(read_only=True),"/cache/huggingface":cache,"/outputs":outputs})
def run(mode: str, run_id: str):
    import datetime, hashlib, json, os, subprocess, threading, time
    assert mode in ("preflight", "full")
    assert run_id.startswith("paper-sized-") and "/" not in run_id
    limit = 1800 if mode == "preflight" else 18000
    max_segments = 1 if mode == "preflight" else 3
    out = Path("/outputs") / run_id
    out.mkdir(exist_ok=True)
    def sha(p):
        h=hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda:f.read(8<<20), b""): h.update(block)
        return h.hexdigest()
    def write(p, value):
        temporary=p.with_suffix(p.suffix+".tmp")
        temporary.write_text(json.dumps(value,indent=2))
        os.replace(temporary,p)
    watchdog_stop=threading.Event()
    # A hard process deadline covers setup, child execution, hashing and final commits.
    # Read existing deadline first on replay; source/claim validation still fails closed below.
    existing_claim = out / "claim.json"
    absolute_deadline = (json.loads(existing_claim.read_text())["deadline_unix"]
                         if existing_claim.exists() else time.time()+limit)
    def watchdog():
        if not watchdog_stop.wait(max(0, absolute_deadline-time.time())):
            os._exit(124)
    threading.Thread(target=watchdog,daemon=True).start()
    source_hash = sha(Path("/opt/conditional.py"))
    wrapper_hash = sha(Path("/opt/paper_sized_modal.py"))
    call_id = modal.current_function_call_id()
    started = time.time()
    claim_path = out / "claim.json"
    execution_path = out / "execution.json"
    if execution_path.exists() and json.loads(execution_path.read_text())["exit_code"] == 0:
        watchdog_stop.set()
        return json.loads(execution_path.read_text())
    if claim_path.exists():
        claim=json.loads(claim_path.read_text())
        assert claim["call_id"] == call_id, "No new function call may reuse a run ID"
        assert claim["source_sha256"] == source_hash and claim["wrapper_sha256"] == wrapper_hash
        assert claim["mode"] == mode and claim["max_segments"] == max_segments
        assert (out/"last.pt").exists(), "Provider replay before durable checkpoint: refuse fresh training"
        assert len(claim["segments"]) < max_segments, "Segment ceiling reached"
    else:
        claim=dict(call_id=call_id,mode=mode,source_sha256=source_hash,wrapper_sha256=wrapper_hash,
            started_at_utc=datetime.datetime.fromtimestamp(started,datetime.timezone.utc).isoformat(),
            deadline_unix=absolute_deadline,max_segments=max_segments,segments=[])
    remaining=claim["deadline_unix"]-time.time()
    assert remaining > 180, "Original absolute deadline exhausted"
    segment=len(claim["segments"])+1
    claim["segments"].append(dict(number=segment,task_id=os.getenv("MODAL_TASK_ID"),
        started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        checkpoint_sha256=sha(out/"last.pt") if segment>1 else None))
    write(claim_path,claim)
    for source in ["conditional.py", "paper_sized_modal.py"]:
        (out/source).write_bytes((Path("/opt")/source).read_bytes())
    write(out/f"runtime-segment-{segment}.json",dict(image_id=os.getenv("MODAL_IMAGE_ID"),
        source_sha256=source_hash,wrapper_sha256=wrapper_hash,task_id=os.getenv("MODAL_TASK_ID")))
    outputs.commit()
    stop=threading.Event()
    def sync():
        while not stop.wait(30): outputs.commit()
    thread=threading.Thread(target=sync,daemon=True);thread.start()
    command=["python","-u","/opt/conditional.py",mode,"--output",str(out),"--paper-sized"]
    code=None
    interruption=None
    try:
        with (out/f"console-segment-{segment}.log").open("w") as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                timeout=max(1,claim["deadline_unix"]-time.time()-120))
            code=result.returncode
    except subprocess.TimeoutExpired:
        code=124
        interruption="TimeoutExpired"
    except BaseException as error:
        interruption=type(error).__name__
        raise
    finally:
        stop.set();thread.join()
        record=dict(app_id=app.app_id,function_call_id=call_id,mode=mode,run_id=run_id,segment=segment,
            started_at_utc=claim["started_at_utc"],
            segment_started_at_utc=claim["segments"][-1]["started_at_utc"],
            interruption=interruption,
            finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            deadline_utc=datetime.datetime.fromtimestamp(claim["deadline_unix"],datetime.timezone.utc).isoformat(),
            command=command,exit_code=code,source_sha256=source_hash,wrapper_sha256=wrapper_hash)
        write(out/f"execution-segment-{segment}.json",record)
        write(execution_path,record)
        artifacts={p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in out.iterdir()
            if p.is_file() and p.name != "evidence.json" and not p.name.endswith(".tmp")}
        write(out/"evidence.json",artifacts)
        outputs.commit()
    watchdog_stop.set()
    print(json.dumps(record),flush=True)
    return record

@app.local_entrypoint()
def main(mode: str="preflight", run_id: str="paper-sized-preflight-002"):
    result=run.remote(mode,run_id)
    if result["exit_code"] != 0: raise SystemExit(result["exit_code"] or 1)
