"""Bounded, recorded Modal execution of the declared conditional reconstruction."""

from pathlib import Path
import modal

app = modal.App("2be3c68e-skresnet-conditional")
data = modal.Volume.from_name("datasets", version=2)
cache = modal.Volume.from_name("huggingface-cache", version=2)
outputs = modal.Volume.from_name("2be3c68e-skresnet-results", version=2)
env = {"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"}
source = Path(__file__).with_name("conditional.py")
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.2.6",
        "simple-video-utils==0.7.4",
        "av==18.0.0",
        "opencv-python-headless==4.11.0.86",
    )
    .env(env)
    .add_local_file(source, "/opt/conditional.py")
)
gpu_image = (
    modal.Image.from_registry(
        "ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291"
    )
    .pip_install("timm==1.0.22", "torch-optimizer==0.3.0", "fvcore==0.1.5.post20221221")
    .run_commands(
        "git clone https://github.com/locuslab/TCN.git /opt/TCN && cd /opt/TCN && git checkout 2f8c2b817050206397458dfd1f5a25ce8a32fe65"
    )
    .env(env)
    .add_local_file(source, "/opt/conditional.py")
)


def execute(mode, run_id, limit):
    import datetime, json, subprocess, threading, os, hashlib

    out = Path("/outputs") / run_id
    out.mkdir(parents=True, exist_ok=True)
    if (out / "execution.json").exists() and json.loads(
        (out / "execution.json").read_text()
    ).get("exit_code") == 0:
        return json.loads((out / "execution.json").read_text())
    command = [
        "python",
        "-u",
        "/opt/conditional.py",
        mode,
        "--output",
        "/datasets/lsa64-skresnet-conditional-v1" if mode == "prepare" else str(out),
    ]
    (out / "executed-source.py").write_bytes(Path("/opt/conditional.py").read_bytes())
    (out / "runtime.json").write_text(
        json.dumps(
            {
                "image_id": os.environ.get("MODAL_IMAGE_ID"),
                "source_sha256": hashlib.sha256(
                    Path("/opt/conditional.py").read_bytes()
                ).hexdigest(),
                "visible_cpu_count": os.cpu_count(),
            },
            indent=2,
        )
    )
    (out / "freeze.txt").write_text(
        subprocess.check_output(["python", "-m", "pip", "freeze"], text=True)
    )
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    stop = threading.Event()

    def commit_loop():
        while not stop.wait(120):
            outputs.commit()
            if mode == "prepare":
                data.commit()

    thread = threading.Thread(target=commit_loop, daemon=True)
    thread.start()
    code = 1
    try:
        with open(out / "console.log", "w") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=limit - 120
            )
            code = result.returncode
    except subprocess.TimeoutExpired:
        code = 124
    finally:
        stop.set()
        thread.join()
        record = dict(
            started_at_utc=start,
            finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            command=command,
            exit_code=code,
            app_id=app.app_id,
            function_call_id=modal.current_function_call_id(),
            mode=mode,
        )
        (out / "execution.json").write_text(json.dumps(record, indent=2))
        outputs.commit()
        if mode == "prepare":
            data.commit()
    print(json.dumps(record))
    return record


@app.function(
    image=cpu_image,
    cpu=16,
    memory=32768,
    timeout=7200,
    volumes={"/datasets": data, "/cache/huggingface": cache, "/outputs": outputs},
)
def prepare(run_id: str):
    return execute("prepare", run_id, 7200)


@app.function(
    image=gpu_image,
    gpu="A100-80GB",
    cpu=8,
    memory=65536,
    timeout=18000,
    volumes={
        "/datasets": data.with_mount_options(read_only=True),
        "/cache/huggingface": cache,
        "/outputs": outputs,
    },
)
def train(mode: str, run_id: str):
    return execute(mode, run_id, 1800 if mode == "preflight" else 18000)


@app.local_entrypoint()
def main(mode: str = "preflight", run_id: str = "conditional-preflight-001"):
    result = prepare.remote(run_id) if mode == "prepare" else train.remote(mode, run_id)
    if result["exit_code"] != 0:
        raise SystemExit(result["exit_code"])
