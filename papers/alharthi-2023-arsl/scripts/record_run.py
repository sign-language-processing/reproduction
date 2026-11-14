"""Record a bounded invocation before launch; run from the repository root.
Usage: python scripts/record_run.py RUN_ID GPU_COUNT WALL_SECONDS COMMAND ...
"""
import json,subprocess,sys,datetime,time,re,fcntl,os
from pathlib import Path
root=Path.cwd();p=root/'papers/alharthi-2023-arsl/reproduction.json'
rid=sys.argv[1]; gpu=int(sys.argv[2]);seconds=int(sys.argv[3]);cmd=sys.argv[4:]
assert cmd and seconds>0 and gpu in (0,1)
def persist(record,new=False):
 with Path('/tmp/arsl-ledger.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  doc=json.loads(p.read_text())
  if new:
   if gpu:
    assert not any(g['gate_id']=='cancellation-intent' and g['status']=='open' for g in doc['gates']),'Resolve cancellation intent before GPU execution'
   if rid.startswith('full-') and '--resume-from' in cmd:
    source_id=cmd[cmd.index('--resume-from')+1]
    recovery=doc['conditional_protocol']['recovery_plan'];assert recovery['status']=='authorized','Checkpoint continuation requires recorded authorization'
    previous=next(q for q in doc['runs'] if q['run_id']==source_id)
    assert previous['terminal']['state']=='interrupted' and previous['attempt']['number']==1
    assert previous['attempt']['max_attempts']==1 and previous['terminal']['retry_decision']=='stop'
    proposal=next(x for x in recovery['checkpoint_resumptions'] if x['source_run_id']==source_id)
    assert proposal['proposed_run_id']==rid and seconds<=proposal['max_wall_time_seconds']
    assert previous['checkpoint_inspection']['all_checkpoint_tensors_readable']
    extension={'previous_max_attempts':previous['attempt']['max_attempts'],'authorized_at_utc':recovery['authorized_at_utc'],'basis':'Explicitly authorized checkpoint continuation under unchanged aggregate compute ceiling; no full restart.'}
    previous['attempt'].update(max_attempts=2,ceiling_extension=extension)
    previous['terminal'].update(retry_decision='resume',next_run_id=rid)
    record['attempt']={'group_id':previous['attempt']['group_id'],'number':2,'max_attempts':2,'ceiling_extension':extension}
    record['resume_from_run_id']=source_id
   if rid.startswith('full-') and doc['conditional_protocol']['full_run_plan']['concurrent_runs']>1:
    assert os.environ.get('ARSL_PARALLEL_SCHEDULER')=='1','Use the approved concurrent suite scheduler'
   assert not any(q['run_id']==rid for q in doc['runs']),'Run IDs must be unique'
   doc['runs'].append(record)
  else:doc['runs']=[record if q['run_id']==rid else q for q in doc['runs']]
  temporary=p.with_suffix('.json.tmp');temporary.write_text(json.dumps(doc,indent=2,ensure_ascii=False)+'\n');temporary.replace(p)

now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
d=json.loads(p.read_text());r={'run_id':rid,'agent_ids':['codex-reproduction-agent'],'command':' '.join(cmd),'started_at_utc':None,'finished_at_utc':None,'recording':{'mode':'contemporaneous','source_record_sha256':None,'unknown_fields':[]},'attempt':{'group_id':rid,'number':1,'max_attempts':1},'stop_policy':{'declared_at_utc':now(),'max_wall_time_seconds':seconds,'max_gpu_hours':seconds/3600 if gpu else None,'max_cost_chf':seconds/3600*5 if gpu else None},'compute':{'gpu_count':gpu,'gpu':'A100-40GB' if gpu else None},'modal':{'profile':'repro-sign','environment':'main'},'artifact_ids':[]}
r['purpose']='Real-data training/checkpoint/evaluation preflight' if rid.startswith('preflight') else ('Independent process checkpoint and optimizer resume validation' if rid.startswith('resume-check') else 'Approved conditional full training and evaluation')
r['compute'].update(platform='modal',modal_profile='repro-sign',shared_cache={'modal_volume':'huggingface-cache','mount_path':'/cache/huggingface','environment':{'HF_HOME':'/cache/huggingface','HF_HUB_CACHE':'/cache/huggingface/hub'}})
r['started_at_utc']=now();persist(r,new=True)
log=Path('/tmp')/(rid+'.log')
with log.open('w') as f:
 proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
 try: rc=proc.wait(timeout=seconds)
 except subprocess.TimeoutExpired:proc.terminate();rc=124
r['finished_at_utc']=now();r['exit_code']=rc
ids=re.findall(r'https://modal.com/apps/repro-sign/main/(ap-[A-Za-z0-9]+)',log.read_text())
if ids:r['modal'].update(app_id=ids[0],dashboard_url='https://modal.com/apps/repro-sign/main/'+ids[0])
r['terminal']={'state':'succeeded' if rc==0 else 'failed','reason_code':'completed' if rc==0 else 'command_failed','detail':'Command completed; inspect retained output for scope and any conditional limitations.' if rc==0 else 'Command failed; see evidence and diagnosis.','failure_class':None if rc==0 else 'deterministic_code','retry_decision':'not_needed' if rc==0 else 'stop','gate_ids':[]}
if rc==124:r['terminal'].update(state='stopped',reason_code='wall_time_ceiling_reached',failure_class='budget_protocol',detail='Declared wall-time ceiling reached; inspect remote watchdog and checkpoints before any resumed attempt.')
# Let the already-running legacy writer finish before concurrent final updates.
while True:
 try:doc=json.loads(p.read_text())
 except json.JSONDecodeError:
  time.sleep(.1);continue
 legacy=doc['conditional_protocol']['full_run_plan'].get('transition_wait_for_run')
 if not legacy or any(q['run_id']==legacy and 'terminal' in q for q in doc['runs']):break
 time.sleep(2)
persist(r)
print('EXIT',rc,'LOG',str(log));print(log.read_text()[-14000:])
sys.exit(rc)
