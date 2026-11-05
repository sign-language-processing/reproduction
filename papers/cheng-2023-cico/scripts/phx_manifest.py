"""Read-only PHOENIX byte/frame inventory for prospective full-run accounting."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib,json,time,sys,pickle
from simple_video_utils.metadata import video_metadata
root=Path('/datasets/rwth-phoenix-2014-t/videos');paths=sorted(root.glob('*/*.mp4'));start=time.monotonic()
def inspect(p):
 with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 meta=video_metadata(str(p))
 return {'path':str(p.relative_to(root)),'sha256':digest,'bytes':p.stat().st_size,'frames':meta.nb_frames,'width':meta.width,'height':meta.height,'fps':meta.fps}
records=[]
with ThreadPoolExecutor(max_workers=8) as pool:
 for i,item in enumerate(pool.map(inspect,paths)):
  records.append(item)
  if (i+1)%500==0:print(f'Inspected {i+1}/{len(paths)}',flush=True)
summary={}
for split in ['train','dev','test']:
 selected=[r for r in records if r['path'].startswith(split+'/')]
 summary[split]={'videos':len(selected),'frames':sum(r['frames'] for r in selected),'windows16_stride1':sum(max(1,r['frames']-15) for r in selected),'bytes':sum(r['bytes'] for r in selected),'shapes':dict(Counter(str((r['height'],r['width'])) for r in selected))}
assert [summary[s]['videos'] for s in ['train','dev','test']]==[7096,519,642]
for split in ['train','test']:
 with open('/upstream/CiCo/CLCL/data_ph/'+split+'.pkl','rb') as f:labels=pickle.load(f)
 names={r['video_name'] for v in labels.values() for r in (v if isinstance(v,list) else [v])}
 assert names=={Path(r['path']).stem for r in records if r['path'].startswith(split+'/')}
report={'dataset_root':'modal://datasets/rwth-phoenix-2014-t/videos','summary':summary,'records':records,'wall_seconds':time.monotonic()-start}
Path(sys.argv[1]).write_text(json.dumps(report,sort_keys=True,indent=2));print(json.dumps(summary),flush=True)
