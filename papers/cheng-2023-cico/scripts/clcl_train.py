"""Stage closed native feature bytes, then run the author's full PHOENIX recipe."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,time,zipfile
import torch
p=argparse.ArgumentParser()
for key in ['output','agnostic-manifest','agnostic-sha','aware-manifest','aware-sha']:p.add_argument('--'+key,required=True)
p.add_argument('--deadline',required=True,type=float);p.add_argument('--resume',action='store_true');p.add_argument('--resume-sha');p.add_argument('--preflight',action='store_true');a=p.parse_args()
out=Path(a.output);source=Path('/upstream/CiCo/CLCL');os.chdir(source)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
staged={};input_evidence={};started=time.monotonic()
for stream,manifest_path,expected in [('agnostic',a.agnostic_manifest,a.agnostic_sha),('aware',a.aware_manifest,a.aware_sha)]:
 manifest_path=Path(manifest_path);assert sha(manifest_path)==expected;m=json.loads(manifest_path.read_text());archive=manifest_path.parent/'features.zip'
 assert sha(archive)==m['archive_sha256'] and len(m['features'])==7738
 local=Path('/tmp/cico-clcl-inputs')/stream;local.mkdir(parents=True,exist_ok=True);local_zip=local/'features.zip'
 if not local_zip.exists() or sha(local_zip)!=m['archive_sha256']:
  partial=local_zip.with_suffix('.partial');shutil.copyfile(archive,partial);assert sha(partial)==m['archive_sha256'];partial.replace(local_zip)
 with zipfile.ZipFile(local_zip) as z:
  assert z.namelist()==[r['member'] for r in m['features']]
  for r in m['features']:
   member=Path(r['member']);assert not member.is_absolute() and '..' not in member.parts
   dest=local/member;dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists() or sha(dest)!=r['sha256']:
    payload=z.read(r['member']);assert len(payload)==r['bytes'] and hashlib.sha256(payload).hexdigest()==r['sha256'];partial=dest.with_suffix('.partial');partial.write_bytes(payload);partial.replace(dest)
   assert dest.stat().st_size==r['bytes'] and sha(dest)==r['sha256']
 staged[stream]=local;input_evidence[stream]={'manifest_sha256':expected,'archive_sha256':m['archive_sha256'],'source_run':m['source_run'],'weights_sha256':m['weights_sha256'],'files':len(m['features'])}
assert input_evidence['agnostic']['weights_sha256']=='6430592464a357dfdaa7f31973cb684663237655fdf23f3999608d162167fc6f'
assert input_evidence['aware']['weights_sha256']!=input_evidence['agnostic']['weights_sha256']
clip=Path('/outputs/training-inputs/ViT-B-32.pt');assert sha(clip)=='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af';shutil.copyfile(clip,source/'modules/ViT-B-32.pt')
import pickle
for split,count in [('train',7096),('test',642)]:
 labels=pickle.load((source/'data_ph'/f'{split}.pkl').open('rb'));assert len(labels)==count
 names={item['video_name'] for item in labels.values()};assert len(names)==count
 for path in staged.values():assert {x.stem for x in (path/split).glob('*.pkl')}==names
(out/'staging.json').write_text(json.dumps({'seconds':time.monotonic()-started,'streams':input_evidence,'train_examples':7096,'test_examples':642,'batches_per_epoch':13,'optimizer_steps':2600},indent=2))
command=['python','-m','torch.distributed.run','--standalone','--nproc_per_node=1','main_task_retrieval.py','--do_train','--epochs','200','--batch_size','512','--batch_size_val','256','--gradient_accumulation_steps','1','--num_thread_reader','0','--datatype','ph','--data_path',str(source/'data_ph'),'--features_path',str(staged['agnostic']),'--features_path_retrain',str(staged['aware']),'--alpha','.9','--output_dir',str(out/'native')]
checkpoint=out/'native/last-full-state.pt';state=None
if a.resume:
 assert checkpoint.exists() and sha(checkpoint)==a.resume_sha;state=torch.load(checkpoint,map_location='cpu',weights_only=True);assert 0<=state['epoch']<=199
 command+=['--resume_model',str(checkpoint)]
else:assert not checkpoint.exists()
env=dict(os.environ)
if a.preflight:env['CICO_DIAGNOSTIC_STOP_EPOCH']='2' if a.resume else '1'
else:env.pop('CICO_DIAGNOSTIC_STOP_EPOCH',None)
(out/'native-command.json').write_text(json.dumps(command));completion_only=state is not None and state['epoch']==199 and not a.preflight
native_execution={'native_exit_code':None,'completion_only':completion_only}
if not completion_only:
 process=subprocess.run(command,env=env,timeout=max(1,a.deadline-time.time()-90));native_execution['native_exit_code']=process.returncode
(out/'native-execution.json').write_text(json.dumps(native_execution))
if native_execution['native_exit_code'] not in [None,0]:raise RuntimeError('Native training returned nonzero')
state=torch.load(checkpoint,map_location='cpu',weights_only=True)
expected_epoch=(1 if a.resume else 0) if a.preflight else 199
assert state['epoch']==expected_epoch and state['global_step']==(expected_epoch+1)*13
assert len(state['loss_record'])==len(state['acc_record'])==expected_epoch+1
assert all(torch.isfinite(torch.tensor(state['loss_record'])))
best=state['best_epoch'];metric=out/'native'/f'best-metrics-{best}.json';scores=json.loads(metric.read_text());assert scores['text_to_video']['R1']==state['best_score']
selected=out/'native'/f'best-model-{best}.pt';similarities=out/'native'/f'best-similarities-{best}.npz';assert selected.exists() and similarities.exists()
report={'completed_epochs':expected_epoch+1,'global_step':state['global_step'],'best_epoch_zero_based':best,'selection':'Native maximum test T2V R@1, latest epoch on ties; test evaluated every epoch.','metrics':scores,'last_checkpoint_sha256':sha(checkpoint),'selected_checkpoint_sha256':sha(selected),'selected_metrics_sha256':sha(metric),'selected_similarities_sha256':sha(similarities),'completion_only':completion_only,'preflight':a.preflight,'input_manifests':input_evidence}
(out/('preflight-resumed.json' if a.preflight and a.resume else 'preflight.json' if a.preflight else 'complete.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
