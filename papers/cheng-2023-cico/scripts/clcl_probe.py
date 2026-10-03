"""Native global-batch512 mechanics using real training features; no target score."""
from pathlib import Path
import os,sys,json,hashlib,pickle,shutil,subprocess,time
import numpy as np
import torch
out=Path(sys.argv[1]);source=Path('/upstream/CiCo/CLCL');os.chdir(source)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
clip=Path('/outputs/training-inputs/ViT-B-32.pt');assert sha(clip)=='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af';shutil.copyfile(clip,source/'modules/ViT-B-32.pt')
base=Path('/outputs/phx-agnostic-features-full-v1/train');available={}
for receipt in sorted(base.glob('rank-*.json')):
 for a in json.loads(receipt.read_text())['artifacts']:
  if a['path'].endswith('.pkl'):available[Path(a['path']).stem]=a
labels=pickle.load((source/'data_ph/train.pkl').open('rb'));selected={};evidence=[];features=Path('/tmp/cico-clcl-mechanics-features');(features/'train').mkdir(parents=True,exist_ok=True);(features/'test').mkdir(exist_ok=True)
for key,item in labels.items():
 name=item['video_name']
 if name not in available:continue
 a=available[name];p=base/a['path'];assert sha(p)==a['sha256'];obj=pickle.load(p.open('rb'));assert obj['feature'].ndim==2 and obj['feature'].shape[1]==1024 and np.isfinite(obj['feature']).all()
 shutil.copyfile(p,features/'train'/f'{name}.pkl');selected[key]=item;evidence.append(dict(a,query_id=key,video_name=name))
 if len(selected)==512:break
assert len(selected)==512
subset=out/'subset';subset.mkdir();pickle.dump(selected,(subset/'train.pkl').open('wb'));test=dict(list(selected.items())[:16]);pickle.dump(test,(subset/'test.pkl').open('wb'))
for item in test.values():shutil.copyfile(features/'train'/f"{item['video_name']}.pkl",features/'test'/f"{item['video_name']}.pkl")
(out/'diagnostic-inputs.json').write_text(json.dumps({'training_examples':512,'evaluation_examples':16,'evaluation_is_training_subset':True,'aware_stream':'Same real domain-agnostic features for mechanics only; full-run gate requires independently trained aware features.','artifacts':evidence},indent=2))
basecmd=['python','-m','torch.distributed.run','--standalone','--nproc_per_node=1','main_task_retrieval.py','--do_train','--epochs','200','--batch_size','512','--batch_size_val','16','--gradient_accumulation_steps','1','--num_thread_reader','0','--datatype','ph','--data_path',str(subset),'--features_path',str(features),'--features_path_retrain',str(features),'--alpha','.9']
def execute(name,stop,resume=None):
 env=dict(os.environ,CICO_DIAGNOSTIC_STOP_EPOCH=str(stop));command=basecmd+['--output_dir',str(out/name)]
 if resume:command+=['--resume_model',str(resume)]
 print('COMMAND',json.dumps(command),flush=True);t=time.monotonic();subprocess.run(command,env=env,check=True,timeout=230);return time.monotonic()-t
first=execute('split',1);checkpoint=out/'split/last-full-state.pt';a=torch.load(checkpoint,map_location='cpu',weights_only=True);assert a['epoch']==0 and a['global_step']==1;first_state=a['model_state_dict']
second=execute('split',2,checkpoint);continuous=execute('continuous',2)
a=torch.load(checkpoint,map_location='cpu',weights_only=True);assert a['epoch']==1 and a['global_step']==2
changed=sum(not torch.equal(value,first_state[key]) for key,value in a['model_state_dict'].items());assert changed>0
b=torch.load(out/'continuous/last-full-state.pt',map_location='cpu',weights_only=True)
assert a['epoch']==b['epoch']==1 and a['global_step']==b['global_step']==2
assert a['optimizer_state_dict']['param_groups']==b['optimizer_state_dict']['param_groups']
model_differences={k:float((v.float()-b['model_state_dict'][k].float()).abs().max()) for k,v in a['model_state_dict'].items() if not torch.equal(v,b['model_state_dict'][k])}
optimizer_differences={}
for key,entry in a['optimizer_state_dict']['state'].items():
 for name,value in entry.items():
  expected=b['optimizer_state_dict']['state'][key][name]
  if torch.is_tensor(value):
   if not torch.equal(value,expected):optimizer_differences[f'{key}:{name}']=float((value.float()-expected.float()).abs().max())
  else:assert value==expected
initialization=json.loads((out/'split/initialization-shapes.json').read_text())
assert initialization['retained_random_native_tensors']=={'clip.visual.conv1.weight':{'pretrained':[768,3,32,32],'model':[768,1024,1,1]},'clip.visual.positional_embedding':{'pretrained':[50,768],'model':[65,768]}}
assert initialization['compatible_loaded_tensors_exact'] and initialization['compatible_tensor_count']>100

proof=json.loads((out/'split/resume-proof.json').read_text());assert all(proof[k] for k in ['model_exact','optimizer_exact','rng_exact'])
assert json.loads((out/'split/diagnostic-batch-1.json').read_text())==json.loads((out/'continuous/diagnostic-batch-1.json').read_text())
assert all(np.isfinite(a['loss_record']))
for entry in a['optimizer_state_dict']['state'].values():
 for value in entry.values():
  if torch.is_tensor(value):assert torch.isfinite(value).all()
report={'global_batch':512,'accumulation':1,'optimizer_horizon_epochs':200,'diagnostic_epochs':2,'fresh_resume_state_exact':True,'next_batch_tensors_match':True,'first_epoch_seconds':first,'resumed_epoch_seconds':second,'continuous_two_epoch_seconds':continuous,'losses':a['loss_record'],'changed_tensors_after_second_update':changed,'initialization':initialization,'trajectory_comparison':{'model_different_tensors':len(model_differences),'model_max_absolute_difference':max(model_differences.values(),default=0),'optimizer_different_tensors':len(optimizer_differences),'optimizer_max_absolute_difference':max(optimizer_differences.values(),default=0),'loss_history_split':a['loss_record'],'loss_history_continuous':b['loss_record'],'metric_history_split':a['acc_record'],'metric_history_continuous':b['acc_record']},'checkpoint_sha256':sha(checkpoint),'not_a_target':'Diagnostic duplicated agnostic feature stream and16training-query evaluation; no paper result or final aware-pair preflight.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
