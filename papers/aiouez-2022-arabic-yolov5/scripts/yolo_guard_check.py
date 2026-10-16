"""Isolated exercise of the actual full-run cancellation/deadline guard."""
import ast,pathlib,tempfile,json,types,time,hashlib
from unittest.mock import patch
source=pathlib.Path(__file__).with_name('yolo_full.py');tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='full');node.decorator_list=[]
with tempfile.TemporaryDirectory() as temp:
 root=pathlib.Path(temp);(root/'work').mkdir();(root/'outputs').mkdir();(root/'work/patches').mkdir()
 for name in ['yolo_stage.py','yolo_train.py','yolo_full.py','Dockerfile.yolo']:(root/'work'/name).write_text(name)
 def mapped(v):
  s=str(v)
  for prefix in ['/work','/outputs']:
   if s==prefix or s.startswith(prefix+'/'):return str(root/s.lstrip('/'))
  return v
 class LocalPath(pathlib.PosixPath):
  def __new__(cls,*args):return super().__new__(cls,*(mapped(a) for a in args))
  def __init__(self,*args):super().__init__(*(mapped(a) for a in args))
  def relative_to(self,*other,**kwargs):return super().relative_to(*(mapped(a) for a in other),**kwargs)
 commits=[]; timers=[];kills=[]
 class Timer:
  def __init__(self,delay,callback):self.delay=delay;self.cancelled=False;timers.append(self)
  def start(self):pass
  def cancel(self):self.cancelled=True
 class Child:
  count=0
  def __init__(self,*a,**k):self.index=Child.count;Child.count+=1;self.pid=42+self.index;self.done=False
  def wait(self,timeout=None):
   if self.index==1:raise KeyboardInterrupt('fixture cancellation during trainer')
   self.done=True;return 0
  def poll(self):return 0 if self.done else None
 ns={'Path':LocalPath,'modal':types.SimpleNamespace(current_function_call_id=lambda:'fixture-call'),'APP':types.SimpleNamespace(app_id='fixture-app'),'OUTPUT':types.SimpleNamespace(commit=lambda:commits.append(1)),'MANIFEST_SHA':'a'*64}
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
 fn=ns['full']
 with patch('shutil.copytree'),patch('subprocess.run'),patch('subprocess.check_output',return_value='fixture'),patch('subprocess.Popen',Child),patch('os.killpg',side_effect=lambda pid,sig:kills.append(pid)),patch('threading.Timer',Timer):
  try:fn('fixture-full','s','b'*64,600)
  except KeyboardInterrupt:pass
  else:raise AssertionError('cancellation swallowed')
 e=json.loads((root/'outputs/fixture-full/execution.json').read_text());claim=json.loads((root/'outputs/fixture-full/claim.json').read_text())
 assert e['exit_code'] is None and e['segments'][0]['state']=='interrupted' and 43 in kills and timers[0].cancelled
 # Same claim cannot be used by a second logical call.
 ns['modal'].current_function_call_id=lambda:'different-call'
 try:fn('fixture-full','s','b'*64,600)
 except AssertionError:pass
 else:raise AssertionError('duplicate accepted')
 ns['modal'].current_function_call_id=lambda:'fixture-call'
 claim['deadline_unix']=time.time()-1;(root/'outputs/fixture-full/claim.json').write_text(json.dumps(claim))
 result=fn('fixture-full','s','b'*64,600);assert result['exit_code']==124
 assert json.loads((root/'outputs/fixture-full/claim.json').read_text())['deadline_unix']==claim['deadline_unix']
 proof={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'method':'Execute actual function AST with isolated local filesystem and mocked Modal, subprocess and Timer; no cloud operation or GPU.','cancelled_training_retains_unknown_exit':True,'cancelled_child_killed':True,'duplicate_call_rejected':True,'expired_original_deadline_refuses_execution':True,'deadline_not_reset':True,'watchdog_delay_seconds':timers[0].delay,'watchdog_cancelled_after_save':True}
 dest=source.parent.parent/'artifacts/yolo-full-guard-proof.json';dest.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
