# ffmpeg

## apt

Installing `ffmpeg` is simple:
```shell
apt-get install ffmpeg -y
```

To install a specific version (e.g., 4.4.2-0ubuntu1):
```shell
apt-get install ffmpeg=4.4.2-0ubuntu1 -y 
````

If that does not work, try installing from source:
```shell
COPY libraries/ffmpeg/install_from_source.sh /tmp/ffmpeg_install.sh
RUN bash /tmp/ffmpeg_install.sh
```

## Conda

Installing `ffmpeg` is simple:
```shell
conda install -c conda-forge ffmpeg -y
```

To install a specific version (e.g., <5):
```shell
conda install -c conda-forge "ffmpeg<5" -y
````

## Dockerfile

To verify the installation, you can run the following command:

```shell
cd libraries/ffmpeg
docker build -t ffmpeg-test .
docker run --rm ffmpeg-test
```