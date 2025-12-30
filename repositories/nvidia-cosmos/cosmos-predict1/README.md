# Cosmos-Predict1

GitHub: https://github.com/nvidia-cosmos/cosmos-predict1

## Build

```bash
docker build -t cosmos-predict1:latest -f repositories/nvidia-cosmos/cosmos-predict1/Dockerfile .
```

## Run

Training environment test:
```bash
docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  cosmos-predict1:latest \
  bash -c "python scripts/test_environment.py --training"
```
