# Flash Attention

`pip install flash-attn` usually fails on machines without lots of RAM.

The solution is:

```shell
pip install ninja
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```