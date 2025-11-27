# Neural Sign Language Translation (CVPR'18)

Repository: [https://github.com/neccam/nslt](https://github.com/neccam/nslt)

## Reference
```bibtex
@inproceedings{camgoz2018neural,
  author = {Necati Cihan Camgoz and Simon Hadfield and Oscar Koller and Hermann Ney and Richard Bowden},
  title = {Neural Sign Language Translation},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2018}
}
```


## Reproduction
```bash
docker build . -t neccam/nslt
```

```bash
docker run -it --mount type=bind,source="/home/nlp/amit/WWW/datasets/PHOENIX-2014-T-release-v3",target=/nslt/PHOENIX-2014-T-release-v3 neccam/nslt 
```