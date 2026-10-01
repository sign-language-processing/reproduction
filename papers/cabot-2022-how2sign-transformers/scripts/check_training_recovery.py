"""Compare uninterrupted training with a real checkpoint interruption/continuation."""
import gzip,hashlib,json,os,pickle,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/slt')
import torch,yaml

def canonical_state(state):
    optimizer=state['optimizer_state']
    ids=[value for group in optimizer['param_groups'] for value in group['params']]
    mapping={value:index for index,value in enumerate(ids)}
    optimizer['state']={mapping[key]:value for key,value in optimizer['state'].items()}
    for group in optimizer['param_groups']:group['params']=[mapping[key] for key in group['params']]
    if 'repro_state' in state:
        state['repro_state'].pop('best_backup_sha256',None)
        state['repro_state'].pop('best_backup',None)
    return state

out=Path(sys.argv[1])
if len(sys.argv)>2 and sys.argv[2]=='inspect':
    import numpy as np
    from recovery_train import selected_checkpoint
    a=torch.load(selected_checkpoint(out/'reference-model-recovery'),map_location='cpu')
    b=torch.load(selected_checkpoint(out/'continued-model-recovery'),map_location='cpu')
    def differences(a,b,path=''):
        if torch.is_tensor(a):
            if not torch.equal(a,b):print(json.dumps({'path':path,'tensor_difference_max':float((a-b).abs().max()),'shape':list(a.shape)}))
        elif isinstance(a,np.ndarray):
            if not np.array_equal(a,b):print(json.dumps({'path':path,'array_diff':True}))
        elif isinstance(a,dict):
            for k in a:differences(a[k],b[k],path+'/'+str(k))
        elif isinstance(a,(list,tuple)):
            for i,(x,y) in enumerate(zip(a,b)):differences(x,y,path+'/'+str(i))
        elif a!=b:print(json.dumps({'path':path,'left':str(a),'right':str(b)}))
    differences(canonical_state(a),canonical_state(b));print(json.dumps({'optimizer_ids_canonicalized_by_parameter_order':True}));sys.exit(0)
if len(sys.argv)>2:
    variant=sys.argv[2]
    import signjoey.training as training
    import recovery_train
    from parallel_ctc import ParallelCTC
    import tensorflow as tf
    cfg=yaml.safe_load((out/(variant+'.yaml')).read_text())
    torch.set_num_threads(4)
    decoder=ParallelCTC(tf.nn.ctc_beam_search_decoder,4)
    tf.nn.ctc_beam_search_decoder=decoder
    original_batch=training.TrainManager._train_batch
    def traced(self,batch,update=True):
        losses=original_batch(self,batch,update=update)
        with (out/(variant+'-trace.jsonl')).open('a') as stream:
            stream.write(json.dumps({'step':self.steps,'sequences':list(batch.sequence),
                'losses':[float(x.detach().cpu()) for x in losses]})+'\n');stream.flush();os.fsync(stream.fileno())
        return losses
    training.TrainManager._train_batch=traced
    training.test=lambda *args,**kwargs:None  # Training-state diagnostic; full evaluator is verified separately.
    if variant=='interrupted':
        original_save=recovery_train.save_training
        def interrupted(self,iterator,epoch,**kwargs):
            if self.steps==5:kwargs['force']=True
            original_save(self,iterator,epoch,**kwargs)
            if self.steps==5:os._exit(75)
        recovery_train.save_training=interrupted
    training.train(str(out/(variant+'.yaml')))
    decoder.close();sys.exit(0)

source=Path('/datasets/how2sign/spot-align-wicv2023/slt-format/val-preflight.pkl.gz')
assert hashlib.sha256(source.read_bytes()).hexdigest()=='18464e674e8aa4a910c586391ae4782aa2e42433887cf96ed522db589e77cee0'
with gzip.open(source,'rb') as stream:samples=pickle.load(stream)
fixture=out/'fixture';fixture.mkdir()
for split,rows in [('train',samples[1:9]),('dev',samples[2:4]),('test',samples[4:6])]:
    with gzip.GzipFile(filename=str(fixture/(split+'.pkl.gz')),mode='wb',mtime=0) as stream:pickle.dump(rows,stream,protocol=4)
cfg=yaml.safe_load(Path('/outputs/how2sign-preflight-001/config.yaml').read_text())
cfg['data'].update(data_path=str(fixture),train='train.pkl.gz',dev='dev.pkl.gz',test='test.pkl.gz',
    random_train_subset=-1,random_dev_subset=-1,gls_vocab='/outputs/how2sign-preflight-001/model/gls.vocab',
    txt_vocab='/outputs/how2sign-preflight-001/model/txt.vocab')
cfg['training'].update(use_cuda=False,batch_size=2,epochs=3,validation_freq=2,logging_freq=1)
for variant in ['reference','interrupted','resumed']:
    root=out/('reference-model' if variant=='reference' else 'continued-model')
    cfg['training'].update(model_dir=str(root),repro_recovery_dir=str(root.parent/(root.name+'-recovery')))
    if variant=='resumed':
        from recovery_train import selected_checkpoint
        checkpoint=selected_checkpoint(cfg['training']['repro_recovery_dir'])
        cfg['training'].update(load_model=str(checkpoint),repro_resume=True)
    else:
        cfg['training'].pop('load_model',None);cfg['training'].pop('repro_resume',None)
    (out/(variant+'.yaml')).write_text(yaml.safe_dump(cfg))
    with (out/(variant+'.log')).open('w') as stream:
        proc=subprocess.run([sys.executable,__file__,str(out),variant],stdout=stream,stderr=subprocess.STDOUT)
    assert proc.returncode==(75 if variant=='interrupted' else 0),(variant,proc.returncode)

from recovery_train import selected_checkpoint
reference=torch.load(selected_checkpoint(out/'reference-model-recovery'),map_location='cpu')
continued=torch.load(selected_checkpoint(out/'continued-model-recovery'),map_location='cpu')
def equal(a,b):
    import numpy as np
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
# Old optimizer keys are process-local IDs; artifact bytes also differ by process.
best_reference=torch.load(out/'reference-model-recovery'/reference['repro_state']['best_backup'],map_location='cpu')
best_continued=torch.load(out/'continued-model-recovery'/continued['repro_state']['best_backup'],map_location='cpu')
assert equal(canonical_state(best_reference),canonical_state(best_continued)),'Selected native best checkpoint differs'
reference=canonical_state(reference);continued=canonical_state(continued)
assert equal(reference,continued),'Training state differs after continuation'
read=lambda name:[json.loads(x) for x in (out/(name+'-trace.jsonl')).read_text().splitlines()]
ref=read('reference');resumed=read('interrupted')+read('resumed')
assert ref==resumed and len(ref)==12
result=dict(diagnostic_only=True,device='cpu',optimizer_steps=12,epochs=3,interrupted_after_step=5,
    training_optimizer_scheduler_iterator_rng_exact_equal=True,batch_order_and_losses_exact_equal=True,
    final_steps=reference['steps'],best_step=reference['best_ckpt_iteration'],
    train_record_ids=[x['name'] for x in samples[1:9]],
    input_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    note='Real held-out features used only as a discarded correctness fixture; no target training/selection or scores.')
(out/'recovery-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
