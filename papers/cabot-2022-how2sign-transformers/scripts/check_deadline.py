"""Local control-flow test of the actual launcher; no model or Modal execution."""
import ast, contextlib, io, json, tempfile, time, types
from pathlib import Path
from unittest.mock import patch
import hashlib, subprocess
source=Path('papers/cabot-2022-how2sign-transformers/scripts/modal_app.py').read_text()
node=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='run_resumable_how2')
real_hash=hashlib.sha256
clock=[100000.0];calls=[];commits=[];call_id=['call-one'];preempt=[True]
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    def mapped(value):
        value=str(value)
        return root/value.lstrip('/') if value.startswith(('/outputs','/opt','/datasets')) else Path(value)
    for name in ['cabot-run.py','collect.py','evaluation.py','parallel_ctc.py','recovery_train.py','recovery_eval.py']:
        p=mapped('/opt')/name;p.parent.mkdir(exist_ok=True);p.write_text(name)
    manifest=mapped('/datasets/how2sign/spot-align-wicv2023/slt-format/manifest.json');manifest.parent.mkdir(parents=True);manifest.write_text('{"splits":{}}')
    class Hash:
        def __init__(self,*args):self.data=b''.join(args);self.h=real_hash(*args)
        def update(self,data):self.data+=data;self.h.update(data)
        def hexdigest(self):return 'fe0d41ea4877a54f2be7b6601e69fa50da1ecec9f93b179d505123f8276ec48b' if self.data==manifest.read_bytes() else self.h.hexdigest()
    def run(command,**kw):
        meta=json.loads((mapped('/outputs')/'proof/execution.json').read_text())
        assert commits and meta['deadline_unix']==186400
        calls.append(kw['timeout'])
        if preempt[0]:
            preempt[0]=False;clock[0]+=4000
            raise SystemExit('simulated lost container')
        return types.SimpleNamespace(returncode=0)
    scope=dict(Path=mapped,app=types.SimpleNamespace(app_id='app-one'),modal=types.SimpleNamespace(current_function_call_id=lambda:call_id[0]),outputs=types.SimpleNamespace(commit=lambda:commits.append(clock[0])),PYTHON='python')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-launcher>','exec'),scope)
    with patch('time.time',side_effect=lambda:clock[0]),patch('hashlib.sha256',Hash),patch('subprocess.check_output',return_value='fixed-environment'),patch('subprocess.run',side_effect=run),contextlib.redirect_stdout(io.StringIO()):
        try:scope['run_resumable_how2']('proof')
        except SystemExit:pass
        state=json.loads((mapped('/outputs')/'proof/execution.json').read_text());assert state['deadline_unix']==186400 and state['exit_code'] is None
        result=scope['run_resumable_how2']('proof')
        assert calls==[85000,82280] and result['exit_code']==0
        assert result['execution_segments'][0]['state']=='interrupted' and result['execution_segments'][0]['exit_code'] is None
        assert scope['run_resumable_how2']('proof')['exit_code']==0 and len(calls)==2
        call_id[0]='call-other'
        try:scope['run_resumable_how2']('proof');raise AssertionError('different writer accepted')
        except AssertionError as e:assert 'different logical call' in str(e)
        call_id[0]='call-one';result['exit_code']=None;clock[0]=186300
        (mapped('/outputs')/'proof/execution.json').write_text(json.dumps(result))
        assert scope['run_resumable_how2']('proof')['exit_code']==124 and len(calls)==2
print(json.dumps({'diagnostic_only':True,'scope':'Actual launcher control flow with local fixture paths, clock, process and volume mocks; no GPU or Modal execution','original_deadline_durable_before_native_call':True,'native_timeouts_seconds':calls,'replay_deducts_original_elapsed':True,'lost_segment_native_exit_preserved_unknown':True,'different_logical_writer_rejected':True,'terminal_replay_does_not_execute':True,'expired_original_deadline_stops_before_execute':True}))
