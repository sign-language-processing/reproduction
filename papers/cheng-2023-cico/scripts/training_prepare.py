"""Bounded CPU inventory and acquisition for the CiCo training continuation."""
from pathlib import Path
import hashlib,json,pickle,urllib.request,subprocess,time
from collections import Counter,defaultdict
import torch
import gdown
out=Path('/outputs/training-inputs');out.mkdir(exist_ok=True)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
weights={
 'bsl5k.pth.tar':'https://www.robots.ox.ac.uk/~vgg/research/bslattend/data/bsl5k.pth.tar',
 'domain_aware_I3D_H2S.pth.tar':'https://drive.google.com/uc?id=1TbX3UjaUvXhsXSQX2UAm81nitt8tsgLq',
}
report={'weights':{},'datasets':{},'source_revision':'38a4f7b00da7a858d59b7fabe5093876a84db8e0'}
def save():
 temp=out/'preparation-report.json.partial';temp.write_text(json.dumps(report,indent=2));temp.replace(out/'preparation-report.json')
for name,url in weights.items():
 p=out/name
 try:
  if not p.exists():
   temp=out/(name+'.partial')
   if 'drive.google.com' in url:
    result=gdown.download(url,str(temp),quiet=False);assert result
   else:urllib.request.urlretrieve(url,temp)
   temp.replace(p)
 except Exception as e:
  report['weights'][name]={'source':url,'acquisition_error':repr(e)};save();continue
 entry={'source':url,'sha256':sha(p),'bytes':p.stat().st_size}
 try:
  state=torch.load(p,map_location='cpu',weights_only=True)
  entry['top_keys']=list(state)
  sd=state.get('state_dict',state)
  entry['tensor_shapes']={k:list(v.shape) for k,v in sd.items() if hasattr(v,'shape')}
 except Exception as e:entry['safe_inspection_error']=str(e)
 report['weights'][name]=entry;save()
 print(json.dumps({'weight':name,**{k:v for k,v in entry.items() if k!='tensor_shapes'}}),flush=True)
for key,slug,folder in [('ph','rwth-phoenix-2014-t','videos'),('csl','csl-daily','videos'),('h2s','how2sign','')]:
 root=Path('/datasets')/slug/folder
 all_paths=sorted(p for p in root.rglob('*') if p.is_file())
 extensions=Counter(p.suffix.lower() for p in all_paths)
 paths=[p for p in all_paths if p.suffix.lower() in {'.mp4','.avi','.mov','.mkv','.webm'}]
 names=defaultdict(list)
 for p in paths:names[p.stem].append(p)
 duplicates={n:[str(p) for p in ps] for n,ps in names.items() if len(ps)>1}
 entry={'extensions':dict(extensions),'duplicate_stem_count':len(duplicates),'duplicate_stem_examples':dict(list(duplicates.items())[:5]),'video_count':len(paths),'example_paths':[str(p) for p in paths[:3]],'splits':{}}
 data={'ph':'data_ph','csl':'data_csl','h2s':'data_h2'}[key]
 for split in ['train','dev','val','test']:
  p=Path('/upstream/CiCo/CLCL')/data/(split+'.pkl')
  if not p.exists():continue
  with p.open('rb') as f:raw=pickle.load(f)
  items=list(raw.values()) if isinstance(raw,dict) else raw
  records=[v for group in items for v in (group if isinstance(group,list) else [group])]
  requested=[r['video_name'] for r in records]
  entry['splits'][split]={'label_sha256':sha(p),'queries':len(items),'videos':len(requested),'missing':len([n for n in requested if n not in names]),'missing_examples':[n for n in requested if n not in names][:5]}
 entry['representative_video_hashes']=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in paths[:3]]
 report['datasets'][key]=entry;save();print(json.dumps({'dataset':key,**entry}),flush=True)
(out/'preparation-report.json').write_text(json.dumps(report,indent=2))
print('Finished CPU preparation.',flush=True)
