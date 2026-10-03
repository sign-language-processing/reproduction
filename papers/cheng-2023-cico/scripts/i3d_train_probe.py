"""Native I3D trainer microcase and uninterrupted-versus-resumed recovery proof."""
from pathlib import Path
import os,sys,json,subprocess,time,hashlib
import torch
from simple_video_utils.metadata import video_metadata
out=Path(sys.argv[1]);source=Path('/upstream/CiCo/I3D_trainer');os.chdir(source)
clips=sorted(Path('/outputs/native-pseudo-recovery-v2/pseudo/rank-255').glob('*/*.mp4'));assert len(clips)==2
# Four rows exercise the published batch size with two real generated clips.
# Repeated rows and train-as-validation are diagnostics only, never full data.
rows=clips*2+clips
info={'video_path':[str(p) for p in rows],'class_label':[str(int(p.parent.name)) for p in rows],'class_name':[p.parent.name for p in rows],'frame':[int(video_metadata(str(p)).nb_frames) for p in rows],'split':['train']*4+['val']*2}
info_path=out/'diagnostic-info.json';info_path.write_text(json.dumps(info,indent=2));os.environ['CICO_PSEUDO_INFO']=str(info_path)
base=['python','main.py','--datasetname','phoenix2014','--pretrained','/outputs/training-inputs/bsl5k.pth.tar','--train-batch','4','--test-batch','3','--lr','.01','--coef','1','--workers','0','--num_figs','0','--snapshot','1','--num-classes','5383']
def execute(name,epochs,resume=None):
 command=base+['--checkpoint',str(out/name),'--epochs',str(epochs)]
 if resume:command+=['--resume',str(resume)]
 print('COMMAND',json.dumps(command),flush=True);t=time.monotonic();subprocess.run(command,check=True,timeout=180);return time.monotonic()-t
first=execute('split',1);path=out/'split/checkpoint.pth.tar'
a=torch.load(path,map_location='cpu',weights_only=True);assert a['epoch']==1 and 'rng_state' in a
second=execute('split',2,path);continuous=execute('continuous',2)
a=torch.load(path,map_location='cpu',weights_only=True);b=torch.load(out/'continuous/checkpoint.pth.tar',map_location='cpu',weights_only=True)
assert a['epoch']==b['epoch']==2
assert a['optimizer']['param_groups']==b['optimizer']['param_groups']
assert all(torch.equal(v,b['state_dict'][k]) for k,v in a['state_dict'].items())
for k,state in a['optimizer']['state'].items():
 for n,v in state.items():assert torch.equal(v,b['optimizer']['state'][k][n])
assert all(g['lr']==.01 and g['momentum']==.9 for g in a['optimizer']['param_groups'])
report={'native_microcase':True,'batch_size':4,'diagnostic_distinct_pseudo_clips':2,'diagnostic_repeated_training_rows':4,'diagnostic_train_as_val_rows':2,'epochs':2,'model_and_optimizer_exact_after_resume':True,'rng_restored':True,'first_epoch_wall_seconds':first,'resumed_epoch_wall_seconds':second,'continuous_two_epoch_seconds':continuous,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'label_identity':'Class IDs and frame ranges from native pseudo-label outputs; diagnostic repetitions only.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
