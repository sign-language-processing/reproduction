"""Use only via the repository modal_repro_sign.sh wrapper."""
import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
BASE = "ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291"
app = modal.App("734bfc1c-islrhphmoadl")
datasets = modal.Volume.from_name("datasets", create_if_missing=False)
cache = modal.Volume.from_name("huggingface-cache", create_if_missing=False)
outputs = modal.Volume.from_name("734bfc1c-islrhphmoadl-results", create_if_missing=True, version=2)
image = modal.Image.from_registry(BASE).add_local_dir(HERE, "/app", copy=True)
data_image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy==1.26.4", "Pillow==10.4.0").add_local_file(HERE / "data.py", "/app/data.py", copy=True)
ENV = {"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub",
       "TORCH_HOME": "/cache/huggingface/torch", "PYTHONHASHSEED": "42",
       "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONPATH": "/app", "OMP_NUM_THREADS": "4"}


@app.function(image=data_image, cpu=8, memory=8192, timeout=1800, retries=0,
              volumes={"/datasets": datasets, "/cache/huggingface": cache, "/results": outputs}, env=ENV)
def prepare_data(run_name: str):
    from data import prepare, write_json
    result = prepare()
    result["modal_call_id"] = modal.current_function_call_id()
    write_json(Path("/results") / run_name / "data-audit.json", result)
    datasets.commit()
    outputs.commit()
    return result


@app.function(image=image, gpu="T4", cpu=6, memory=16384, timeout=1800, retries=0,
              volumes={"/datasets": datasets.with_mount_options(read_only=True),
                       "/cache/huggingface": cache, "/results": outputs}, env=ENV)
def preflight(run_name: str):
    from experiment import preflight as run
    from data import write_json
    output = Path("/results") / run_name
    try:
        result = run(output)
        result["modal_call_id"] = modal.current_function_call_id()
        result["modal_image_id"] = image.object_id
        write_json(output / "preflight.json", result)
        return result
    finally:
        outputs.commit()


@app.function(image=image, gpu="T4", cpu=6, memory=16384, timeout=1800, retries=0,
              volumes={"/datasets": datasets.with_mount_options(read_only=True),
                       "/cache/huggingface": cache, "/results": outputs}, env=ENV)
def cached_preflight(run_name: str):
    from cached_search import preflight as run
    from data import write_json
    output = Path("/results") / run_name
    try:
        result = run(output)
        result["modal_call_id"] = modal.current_function_call_id()
        result["modal_image_id"] = image.object_id
        write_json(output / "preflight.json", result)
        return result
    finally:
        outputs.commit()


@app.function(image=image, gpu="T4", cpu=(6, 6), memory=(16384, 16384), timeout=43200, retries=0,
              max_containers=1, scaledown_window=2,
              volumes={"/datasets": datasets.with_mount_options(read_only=True),
                       "/cache/huggingface": cache, "/results": outputs}, env=ENV)
def full(run_name: str, full_budget_hours: float):
    from experiment import campaign
    from data import write_json
    path = Path("/results") / run_name
    ledger_path = path / "execution-ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"reserved_gpu_hours": 0, "calls": []}
    # Reserve a worst-case function timeout before starting, including failures.
    # Completed calls release unused reservation; no invisible over-budget retry.
    if full_budget_hours <= 0 or ledger["reserved_gpu_hours"] + 12 > full_budget_hours:
        raise RuntimeError("Reviewed campaign GPU-hour ceiling cannot cover the next bounded call")
    ledger["reserved_gpu_hours"] += 12
    call = {"function_call_id": modal.current_function_call_id(), "state": "reserved", "hours": 12}
    ledger["calls"].append(call)
    write_json(ledger_path, ledger)
    outputs.commit()
    started = time.monotonic()
    try:
        result = campaign(path)
        elapsed = (time.monotonic() - started) / 3600
        ledger["reserved_gpu_hours"] += elapsed - 12
        call.update(state="completed", hours=elapsed)
        write_json(ledger_path, ledger)
        return result
    finally:
        outputs.commit()


@app.function(image=data_image, cpu=(0.125, 0.125), memory=(256, 256),
              timeout=86400, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/results": outputs})
def orchestrate(run_name: str, full_budget_hours: float, deadline_utc: float,
                previous_call_id: str = ""):
    """One cloud-owned stage per controller; successors require no local host."""
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
        temporary.replace(path)
    path = Path("/results") / run_name / "cloud-orchestration.json"
    state = {"controller_call_id": modal.current_function_call_id(),
             "deadline_utc": deadline_utc, "gpu_hour_ceiling": full_budget_hours,
             "previous_call_id": previous_call_id, "state": "running"}
    active = None
    try:
        if previous_call_id:
            # Adopt the detached call already running in the original app.
            state["state"] = "waiting_for_handoff_call"
        else:
            if deadline_utc - time.time() < 43200:
                raise RuntimeError("Campaign wall ceiling cannot cover another bounded GPU call")
            active = full.spawn(run_name, full_budget_hours)
            state["gpu_call_id"] = active.object_id
        write_json(path, state)
        outputs.commit()
        if previous_call_id:
            # Runtime call IDs in the old ledger are not portable spawn handles.
            # Its committed completion record is the cross-app handoff signal.
            wait_until = min(deadline_utc, time.time() + 43500)
            while True:
                outputs.reload()
                ledger = json.loads((path.parent / "execution-ledger.json").read_text())
                prior = next(c for c in ledger["calls"] if c["function_call_id"] == previous_call_id)
                if prior["state"] == "completed":
                    result = {"complete": False, "stage": "handoff_call_completed"}
                    break
                if time.time() >= wait_until:
                    raise RuntimeError("Prior GPU call has no committed completion; stop rather than duplicate it")
                time.sleep(30)
        else:
            result = active.get(timeout=max(1, deadline_utc - time.time()))
        outputs.reload()
        state["result"] = result
        if result.get("complete", False):
            state["state"] = "completed"
            write_json(path, state)
            outputs.commit()
            return state
        state["state"] = "stage_completed"
        write_json(path, state)
        outputs.commit()
        successor = orchestrate.spawn(run_name, full_budget_hours, deadline_utc)
        state.update(state="successor_queued", next_controller_call_id=successor.object_id)
        write_json(path, state)
        outputs.commit()
        return state
    except BaseException as exc:
        if isinstance(exc, TimeoutError) and active is not None:
            active.cancel()
        state.update(state="stopped", error_type=type(exc).__name__, detail=str(exc))
        write_json(path, state)
        outputs.commit()
        raise


@app.function(image=data_image, cpu=(0.125, 0.125), memory=(256, 256),
              timeout=300, retries=0, volumes={"/results": outputs})
def resume_cloud(run_name: str, full_budget_hours: float, deadline_utc: float,
                 interrupted_call_id: str, charged_hours: float):
    """One-off audited reconciliation after confirmed cancellation of old app."""
    if not 0 < charged_hours <= 12:
        raise ValueError("Invalid bounded interruption charge")
    path = Path("/results") / run_name / "execution-ledger.json"
    ledger = json.loads(path.read_text())
    prior = next(c for c in ledger["calls"] if c["function_call_id"] == interrupted_call_id)
    if prior["state"] == "reserved":
        ledger["reserved_gpu_hours"] += charged_hours - prior["hours"]
        prior.update(state="interrupted", hours=charged_hours,
                     accounting="Conservative 0.15-hour charge exceeds observed old app lifetime 0.13 hours; not billed usage.")
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(ledger, indent=2))
        temp.replace(path)
        outputs.commit()
    else:
        raise RuntimeError("Reconciliation already applied or prior call state differs; do not submit twice")
    call = orchestrate.spawn(run_name, full_budget_hours, deadline_utc)
    return {"controller_call_id": call.object_id, "interrupted_charge_hours": charged_hours}


@app.local_entrypoint()
def main(mode: str = "preflight", run_name: str = "preflight-01", full_budget_hours: float = 0,
         previous_call_id: str = "", deadline_utc: float = 0):
    if not run_name or Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("run_name must be a single directory component")
    if mode == "cloud-resume":
        if full_budget_hours <= 0 or deadline_utc <= time.time() or not previous_call_id:
            raise ValueError("Audited resume requires budget, deadline and interrupted call ID")
        result = resume_cloud.remote(run_name, full_budget_hours, deadline_utc, previous_call_id, 0.15)
        print(json.dumps(result), flush=True)
        return
    if mode == "cloud":
        if full_budget_hours <= 0 or deadline_utc <= time.time():
            raise ValueError("Cloud campaign requires an approved budget and future deadline")
        call = orchestrate.spawn(run_name, full_budget_hours, deadline_utc, previous_call_id)
        print(json.dumps({"controller_call_id": call.object_id, "run_name": run_name}), flush=True)
        return
    functions = {"prepare": prepare_data, "preflight": preflight, "cached-preflight": cached_preflight, "full": full}
    if mode not in functions:
        raise ValueError("mode must be prepare, preflight or full")
    if mode == "full":
        if full_budget_hours <= 0:
            raise ValueError("Full campaign requires an explicitly reviewed --full-budget-hours ceiling")
        while True:
            result = full.remote(run_name, full_budget_hours)
            print(json.dumps(result, allow_nan=False), flush=True)
            if result["complete"]:
                break
    else:
        result = functions[mode].remote(run_name)
        # Full scientific evidence is retained in the output volume, not dumped
        # into the console for every preflight.
        summary = {k: v for k, v in result.items() if k != "validation_metrics"}
        print(json.dumps(summary, indent=2, allow_nan=False))
