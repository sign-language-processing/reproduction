"""Execute the approved bounded suite; drain active jobs on any failure."""
import json,subprocess,sys,os,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];ledger=root/'reproduction.json'
initial=json.loads(ledger.read_text())
assert not any(g['gate_id']=='cancellation-intent' and g['status']=='open' for g in initial['gates']),'Resolve cancellation intent before GPU execution'
plan=initial['conditional_protocol']['full_run_plan']
def planned_id(item):return item.get('run_id','full-'+item['model'])
selected_ids={planned_id(item) for item in plan['models']}
assert plan['gpu_count_per_run']==1 and plan['concurrent_runs'] in (1,4)
assert plan['total_declared_gpu_hours']<=24
processes={};halt=False
while True:
    try:doc=json.loads(ledger.read_text())
    except json.JSONDecodeError:
        time.sleep(.1);continue
    runs={r['run_id']:r for r in doc['runs']}
    for rid,(proc,log) in list(processes.items()):
        rc=proc.poll()
        if rc is not None:
            print('DONE',rid,'exit',rc,flush=True);log.close();del processes[rid]
            if rc:halt=True
    relevant=[r for rid,r in runs.items() if rid in selected_ids]
    if any(r.get('terminal',{}).get('state') not in (None,'succeeded') for r in relevant):halt=True
    active=sum('terminal' not in r for r in relevant)+sum(rid not in runs for rid in processes)
    pending=[item for item in plan['models'] if planned_id(item) not in runs and planned_id(item) not in processes]
    if not halt:
        for item in pending[:max(0,plan['concurrent_runs']-active)]:
            model=item['model'];rid=planned_id(item)
            print('START',rid,flush=True)
            command=[sys.executable,str(root.relative_to(Path.cwd())/'scripts/record_run.py'),rid,'1',str(item['max_wall_time_seconds']),
                '.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh','run',str(root.relative_to(Path.cwd())/'scripts/modal_app.py'),
                '--mode','full','--model',model,'--run-id',rid,'--max-seconds',str(item['remote_max_seconds']),
                '--preflight-id',item['preflight_run_id']]
            if item.get('resume_from'):command+=['--resume-from',item['resume_from']]
            log=(Path('/tmp')/(rid+'-operator.log')).open('w')
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'ARSL_PARALLEL_SCHEDULER':'1'})
            processes[rid]=(proc,log)
    if not active and not processes and (halt or not pending):break
    time.sleep(2)
sys.exit(1 if halt else 0)
