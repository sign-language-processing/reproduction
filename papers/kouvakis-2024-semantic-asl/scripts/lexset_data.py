"""Audit the pinned Lexset archive as canonical data; extracted leftovers are unused."""
from pathlib import Path
import collections, hashlib, io, json, shutil, urllib.request, zipfile
from datetime import datetime, timezone
from PIL import Image
import modal
root=Path('/datasets/synthetic-asl-alphabet');root.mkdir(parents=True,exist_ok=True)
url='https://www.kaggle.com/api/v1/datasets/download/lexset/synthetic-asl-alphabet?datasetVersionNumber=3'
expected='fee9a105ef0785ded6c795031fc0b25252263e782a13836645d06f175cd04373'
started=datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
if (root/'manifest.json').exists():
 m=json.loads((root/'manifest.json').read_text());assert m['version']==3 and m['source']==url and m['source_sha256']==expected and m['canonical_format']=='source-v3.zip members'
 print(json.dumps({'status':'already_verified','manifest_sha256':sha(root/'manifest.json'),'summary':m['summary']}));raise SystemExit(0)
archive=root/'source-v3.zip'
if not archive.exists():
 urllib.request.urlretrieve(url,root/'source-v3.zip.partial');(root/'source-v3.zip.partial').rename(archive)
staged=Path('/tmp/lexset-audit-v3.zip');shutil.copyfile(archive,staged)
archive_hash=sha(staged);assert archive_hash==expected
print(json.dumps({'archive_sha256':archive_hash,'bytes':staged.stat().st_size,'canonical_format':'source-v3.zip members'}),flush=True)
counts=collections.Counter();files=[];dims=collections.Counter();hash_splits=collections.defaultdict(set)
with zipfile.ZipFile(staged) as z:
 for entry in z.infolist():
  name=Path(entry.filename);assert not name.is_absolute() and '..' not in name.parts
  if entry.is_dir():continue
  raw=z.read(entry) # ZipFile checks each member's CRC on this complete read.
  h=hashlib.sha256(raw).hexdigest();r={'path':'files/'+entry.filename,'zip_member':entry.filename,'sha256':h,'size_bytes':len(raw),'zip_crc32':entry.CRC}
  if name.suffix.lower() in {'.jpg','.jpeg','.png'}:
   with Image.open(io.BytesIO(raw)) as im:im.load();r['dimensions']=list(im.size);dims[str(im.size)]+=1
   parts=[p.lower() for p in name.parts];split=next((s for s in ['train','test'] if s+'_alphabet' in parts),None)
   if split is None:
    assert str(name)=='alphabet.jpg',('Unexpected unassigned image',name)
    r['role']='alphabet overview, excluded from model samples'
   else:
    label=name.parent.name.upper();counts[f'{split}/{label}']+=1;hash_splits[h].add(split);r.update(split=split,label=label)
  files.append(r)
  if len(files)%3000==0:print('Verified',len(files),flush=True)
letters='ABCDEFGHIKLMNOPQRSTUVWXY'
subset={s:sum(n for k,n in counts.items() if k.startswith(s+'/') and k.split('/')[1] in letters and len(k.split('/')[1])==1) for s in ['train','test']}
assert sum(counts.values())==27000,counts
assert subset=={'train':21600,'test':2400},subset
assert not any(len(s)>1 for s in hash_splits.values()),'Exact duplicate across native splits'
m={'source':url,'version':3,'attribution':'Lexset, Synthetic ASL Alphabet, https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet','license_metadata':'Data files © Original Authors','permission_basis':'Project authorization confirms attribution-based internal acquisition, storage and processing. Source metadata retained verbatim; no broad redistribution license inferred.','canonical_format':'source-v3.zip members','extracted_tree_note':'files/ is an unused noncanonical leftover of earlier acquisition audits; no claim of completed remote-tree CRC audit. Manifest path fields map to ZIP members for deterministic ordering.','source_sha256':archive_hash,'source_bytes':staged.stat().st_size,'files':files,'summary':{'image_count':sum(counts.values()),'auxiliary_image_count':1,'class_split_counts':dict(counts),'paper_subset_counts':subset,'dimensions':dict(dims),'cross_split_exact_duplicates':0},'audit_started_at_utc':started,'audit_finished_at_utc':datetime.now(timezone.utc).isoformat()}
(root/'manifest.partial').write_text(json.dumps(m,indent=2)+'\n');(root/'manifest.partial').replace(root/'manifest.json');modal.Volume.from_name('datasets',version=2).commit()
print(json.dumps({'manifest_sha256':sha(root/'manifest.json'),'summary':m['summary'],'started_at_utc':started,'finished_at_utc':m['audit_finished_at_utc']}),flush=True)
