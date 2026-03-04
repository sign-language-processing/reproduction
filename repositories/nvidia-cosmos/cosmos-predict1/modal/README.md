# Modal (DRAFT)

## Volumes

```shell
# Create the "videos-256" volume
modal volume create --version=2 videos-256

# Upload videos
modal volume put videos-256 /mnt/nas/transformations/videos/256x256 /
```

## Run

```shell
modal run --detach app.py
```
