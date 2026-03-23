# Fix the code to latest torch
sed -i \
  's/super(WarmupLambdaLR, self).__init__(optimizer, lr_lambda, last_epoch, verbose)/super(WarmupLambdaLR, self).__init__(optimizer, lr_lambda, last_epoch)/' \
  cosmos_predict1/utils/scheduler.py

# Enable shuffling, optimize dataloader, and read num_workers from NUM_WORKERS env var (default 1)
sed -i \
  -e 's/shuffle=None/shuffle=True/' \
  -e 's/num_workers=num_workers/num_workers=int(os.environ.get("NUM_WORKERS", "1"))/' \
  -e 's/prefetch_factor=2/prefetch_factor=4 if is_train else 2/' \
  -e 's/persistent_workers=False/persistent_workers=is_train/' \
  -e '1s/^/import os\n/' \
  cosmos_predict1/tokenizer/training/configs/base/data.py

# change supported resolutions to lower values
sed -i \
  -e 's/\["1080", "720", "480", "360", "256"\]/["256", "128", "64"]/' \
  cosmos_predict1/tokenizer/training/configs/registry.py

# Support recursive file discovery in video dataset
sed -i \
  's|datasets/hdvila/videos/\*.mp4|datasets/hdvila/videos/**/*.mp4|' \
  training/datasets/dataset_provider.py

# Skip CPU affinity — nvmlDeviceGetCpuAffinity is unsupported on some cloud GPUs
sed -i \
  's/os.sched_setaffinity(0, device.get_cpu_affinity())/pass  # patched: skip CPU affinity/' \
  cosmos_predict1/utils/distributed.py

