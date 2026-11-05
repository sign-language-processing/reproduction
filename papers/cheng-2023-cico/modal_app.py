"""Run the pinned CiCo release in the repro-sign Modal workspace."""

from pathlib import Path
import modal

HERE = Path(__file__).parent
app = modal.App("repro-cico-18c49909")
data_volume = modal.Volume.from_name("datasets", version=2)
cache_volume = modal.Volume.from_name("huggingface-cache", version=2)
outputs = modal.Volume.from_name(
    "cheng-2023-cico-results", create_if_missing=True, version=2
)
base = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("gdown==5.2.0")
    .env({"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"})
    .add_local_file(HERE / "data.sh", "/repro/data.sh")
)


@app.function(
    image=base,
    timeout=3600,
    cpu=2,
    memory=4096,
    volumes={
        "/datasets": data_volume,
        "/cache/huggingface": cache_volume,
        "/outputs": outputs,
    },
)
def populate(run_id: str):
    import subprocess, json, datetime, hashlib, time

    out = Path("/outputs") / run_id
    out.mkdir(exist_ok=False)
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t = time.monotonic()
    p = subprocess.run(
        ["bash", "/repro/data.sh"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log = out / "console.log"
    log.write_text(p.stdout)
    print(p.stdout[-5000:])
    record = {
        "started_at_utc": start,
        "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "exit_code": p.returncode,
        "wall_time_seconds": time.monotonic() - t,
        "function_call_id": modal.current_function_call_id(),
        "log_sha256": hashlib.file_digest(log.open("rb"), "sha256").hexdigest(),
    }
    (out / "run.json").write_text(json.dumps(record, indent=2))
    data_volume.commit()
    outputs.commit()
    return record


GPU_BASE = "ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291"
REV = "38a4f7b00da7a858d59b7fabe5093876a84db8e0"
gpu_image = (
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


@app.function(
    image=gpu_image,
    gpu="A10G",
    cpu=4,
    memory=16384,
    timeout=7200,
    volumes={
        "/datasets": data_volume.with_mount_options(read_only=True),
        "/cache/huggingface": cache_volume,
        "/outputs": outputs,
    },
)
def evaluate(dataset: str, run_id: str, subset: int = 0, max_seconds: int = 3300):
    import subprocess, json, datetime, hashlib, time, pickle, threading

    out = Path("/outputs") / run_id
    out.mkdir(exist_ok=False)
    (out / "started.json").write_text(
        json.dumps(
            {
                "started_at_utc": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "function_call_id": modal.current_function_call_id(),
            }
        )
    )
    outputs.commit()
    work = Path("/upstream/CiCo/CLCL")
    root = Path("/datasets/cico-features")
    manifests = [str(p) for p in root.glob("*")]
    print("Dataset roots:", manifests)
    weights = Path("/outputs/released-weights")
    mapping = {
        "h2s": ("data_h2", "H2S_sota.pth", "h2s", 0.8),
        "ph": ("data_ph", "ph_sota.pth", "ph", 0.9),
        "csl": ("data_csl", "csl_sota.pth", "csl", 0.8),
    }
    data, weight, prefix, alpha = mapping[dataset]
    weight_paths = list(weights.rglob(weight))
    assert len(weight_paths) == 1, weight_paths
    feature_dirs = [root / "sign_features" / (prefix + "_domain_agnostic")]
    assert feature_dirs[0].is_dir()
    aware = [root / "sign_features" / (prefix + "_domain_aware")]
    assert aware[0].is_dir()
    datapath = work / data
    with (datapath / "test.pkl").open("rb") as f:
        full_labels = pickle.load(f)
    names = [
        v["video_name"]
        for item in full_labels.values()
        for v in (item if isinstance(item, list) else [item])
    ]
    missing = [
        name
        for name in names
        if not all(
            (folder / "test" / (name + ".pkl")).is_file()
            for folder in (feature_dirs[0], aware[0])
        )
    ]
    audit = {
        "query_groups": len(full_labels),
        "video_count": len(names),
        "missing_feature_count": len(missing),
        "label_sha256": hashlib.file_digest(
            (datapath / "test.pkl").open("rb"), "sha256"
        ).hexdigest(),
        "checkpoint_sha256": hashlib.file_digest(
            weight_paths[0].open("rb"), "sha256"
        ).hexdigest(),
    }
    (out / "data-audit.json").write_text(json.dumps(audit, indent=2))
    assert not missing, f"Missing {len(missing)} feature pairs"
    if subset:
        datapath = out / "subset"
        datapath.mkdir(exist_ok=True)
        with (work / data / "test.pkl").open("rb") as f:
            labels = pickle.load(f)
        with (datapath / "test.pkl").open("wb") as f:
            pickle.dump(dict(list(labels.items())[:subset]), f)
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        "main_task_retrieval.py",
        "--do_eval",
        "--init_model",
        str(weight_paths[0]),
        "--data_path",
        str(datapath),
        "--datatype",
        dataset,
        "--features_path",
        str(feature_dirs[0]),
        "--features_path_retrain",
        str(aware[0]),
        "--alpha",
        str(alpha),
        "--output_dir",
        str(out / "native"),
        "--num_thread_reader",
        "2",
        "--batch_size_val",
        "16" if 0 < subset < 256 else "256",
    ]
    (out / "command.json").write_text(json.dumps(command))
    (out / "pip-freeze.txt").write_text(
        subprocess.check_output(["python", "-m", "pip", "freeze"], text=True)
    )
    (out / "hardware.txt").write_text(
        subprocess.check_output(["nvidia-smi"], text=True)
    )
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t = time.monotonic()
    samples = []
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.5):
            samples.append(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
            )

    th = threading.Thread(target=monitor)
    th.start()
    exit_code = 124
    try:
        with (out / "console.log").open("w") as log:
            try:
                exit_code = subprocess.run(
                    command,
                    cwd=work,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=max_seconds,
                ).returncode
            except subprocess.TimeoutExpired:
                log.write("\nDeclared command wall-time ceiling reached.\n")
    finally:
        stop.set()
        th.join()
    log = out / "console.log"
    print(log.read_text()[-9000:])
    record = {
        "dataset": dataset,
        "subset": subset,
        "started_at_utc": start,
        "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "exit_code": exit_code,
        "wall_time_seconds": time.monotonic() - t,
        "function_call_id": modal.current_function_call_id(),
        "peak_device_memory_mib_sampled": max([int(x) for x in samples] or [0]),
        "artifacts": {
            str(f.relative_to(out)): hashlib.file_digest(
                f.open("rb"), "sha256"
            ).hexdigest()
            for f in out.rglob("*")
            if f.is_file()
        },
    }
    (out / "run.json").write_text(json.dumps(record, indent=2))
    outputs.commit()
    return record


@app.local_entrypoint()
def main(
    stage: str = "data",
    run_id: str = "data-1",
    dataset: str = "ph",
    subset: int = 0,
    max_seconds: int = 3300,
):
    import json

    result = (
        populate.remote(run_id)
        if stage == "data"
        else evaluate.remote(dataset, run_id, subset, max_seconds)
    )
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}, indent=2))
    if result["exit_code"]:
        raise SystemExit(result["exit_code"])
