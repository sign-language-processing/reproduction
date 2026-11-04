"""Verify the existing corpus and create the approved, conditional seed-42 split."""
import hashlib,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
DATA=Path('/datasets/arasl-database-grayscale')
SPLIT=DATA/'splits/alharthi-2023-seed42.json'
SPLIT_SHA256='510adb00b17ec47886f5adf8d0b3ab9d721659430bd9b73c6d9b6f2b579ea428'
PARQUET=DATA/'data/train-00000-of-00001-aa6a48ea2f282316.parquet'
EXPECTED='7c6d9b276f5960bf9fb0efc99c7df3d3854b0690101751f74ab30d68a125d3a3'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
    return h.hexdigest()
def corpus():
    assert sha(PARQUET)==EXPECTED,'Dataset bytes differ from pinned manifest'
    t=pq.read_table(PARQUET).to_pydict()
    y=np.asarray(t['label'],dtype=np.int64)
    assert len(y)==54049 and np.array_equal(np.unique(y),np.arange(32))
    return t,y
if __name__=='__main__':
    t,y=corpus()
    train,val=train_test_split(np.arange(len(y)),test_size=5404,stratify=y,random_state=42)
    split={'conditional':True,'seed':42,'parquet_sha256':EXPECTED,'train':train.tolist(),'validation':val.tolist(),'class_counts':np.bincount(y).tolist(),'train_class_counts':np.bincount(y[train]).tolist(),'validation_class_counts':np.bincount(y[val]).tolist()}
    payload=json.dumps(split,sort_keys=True,separators=(',',':'))+'\n'
    SPLIT.parent.mkdir(parents=True,exist_ok=True)
    if SPLIT.exists():assert SPLIT.read_text()==payload,'Refusing to overwrite a different split'
    else:SPLIT.write_text(payload)
    print(json.dumps({'samples':len(y),'train':len(train),'validation':len(val),'parquet_sha256':EXPECTED,'split_sha256':sha(SPLIT),'split_path':str(SPLIT),'class_counts':split['class_counts']}))
