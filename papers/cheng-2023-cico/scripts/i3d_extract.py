"""Path/output entry point around pinned native I3D dataset and epoch functions."""
from pathlib import Path
import argparse,sys,os,json,hashlib,time,shutil
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--weights',required=True);p.add_argument('--weights-sha',required=True);p.add_argument('--mode',choices=['pseudo','features'],required=True);p.add_argument('--split',choices=['train','dev','test'],required=True);p.add_argument('--probe',action='store_true');args=p.parse_args()
if args.mode=='pseudo' and args.split!='train':raise ValueError('Pseudo labels are training-only')
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(args.manifest)==args.manifest_sha
manifest=json.loads(Path(args.manifest).read_text());selected=[r for r in manifest['records'] if r['path'].startswith(args.split+'/')]
if args.probe:selected=selected[:3]
out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
identity={'mode':args.mode,'split':args.split,'probe':args.probe,'manifest_sha256':args.manifest_sha,'weights_sha256':args.weights_sha,'entrypoint_sha256':sha(__file__)}
identity_path=out/'identity.json'
if identity_path.exists():assert json.loads(identity_path.read_text())==identity
else:identity_path.write_text(json.dumps(identity,indent=2))
local=Path('/tmp/cico-raw')/args.split;local.mkdir(exist_ok=True,parents=True)
for r in selected:
 source=Path('/datasets/rwth-phoenix-2014-t/videos')/r['path'];dest=local/source.name
 if not dest.exists() or sha(dest)!=r['sha256']:shutil.copyfile(source,dest)
 assert sha(dest)==r['sha256']
assert sha(args.weights)==args.weights_sha
sys.path.insert(0,'/upstream/CiCo/I3D_feature_extractor');os.chdir('/upstream/CiCo/I3D_feature_extractor')
from models.i3d import InceptionI3d
from datasets.phoenix2014 import PHOENIX2014
from simple_video_utils.metadata import video_metadata
from simple_video_utils.frames import read_frames_exact
if args.mode=='pseudo':from epoch_pseudo import do_epoch
else:from epoch import do_epoch
torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
state=torch.load(args.weights,map_location='cpu',weights_only=True)['state_dict'];classes=state['module.logits.conv3d.weight'].shape[0]
model=InceptionI3d(num_classes=classes,include_embds=True,num_in_frames=16).cuda();model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True);model.eval()
ranks=list(range(256 if args.split=='train' else 16))
if args.probe:ranks=ranks[-1:]
start=time.monotonic()
for rank in ranks:
 done=out/f'rank-{rank:03d}.json'
 if done.exists():
  saved=json.loads(done.read_text())
  for entry in saved['artifacts']:assert sha(out/entry['path'])==entry['sha256']
  continue
 final=out/f'rank-{rank:03d}'
 # Missing receipt means rank was never durably completed; only its own files
 # may be discarded and regenerated. Completed ranks are verified above.
 if final.exists():shutil.rmtree(final)
 partial=out/f'rank-{rank:03d}.partial'
 if partial.exists():shutil.rmtree(partial)
 partial.mkdir()
 torch.manual_seed(0);np.random.seed(0)
 ds=PHOENIX2014(root_path=str(local.parent),split=args.split,rank=rank,gpu_collation=256,setname='val')
 loader=torch.utils.data.DataLoader(ds,batch_size=18,shuffle=False,num_workers=0,collate_fn=ds.collate_fn)
 t=time.monotonic()
 do_epoch(args.split,loader,model,torch.nn.CrossEntropyLoss().cuda(),epochno=rank,num_classes=classes,feature_dim=1024,save_features=True,save_logits=True,num_figs=0,save_dir=str(partial))
 final=out/f'rank-{rank:03d}';partial.replace(final)
 for clip in sorted(final.rglob('*.mp4')):
  meta=video_metadata(str(clip));assert clip.stat().st_size>0 and meta.nb_frames>0
  assert len(list(read_frames_exact(str(clip),0,0)))==1
 artifacts=[{'path':str(f.relative_to(out)),'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(final.rglob('*')) if f.is_file()]
 receipt={'rank':rank,'videos':len(ds.train),'windows':len(ds),'wall_seconds':time.monotonic()-t,'artifacts':artifacts}
 temp=done.with_suffix('.partial');temp.write_text(json.dumps(receipt,indent=2));temp.replace(done)
 with done.open('rb') as f:os.fsync(f.fileno())
 print(json.dumps({k:v for k,v in receipt.items() if k!='artifacts'}),flush=True)
 print('artifact_count',len(artifacts),flush=True)
 del loader,ds
(out/'complete.json').write_text(json.dumps({'identity':identity,'ranks':ranks,'wall_seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated()},indent=2))
