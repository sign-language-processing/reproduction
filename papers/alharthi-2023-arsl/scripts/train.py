"""Approved conditional training and evaluation, with resumable epoch checkpoints."""
import argparse,datetime,io,json,os,random,subprocess,time,shutil,sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,roc_auc_score,log_loss
from data import corpus,SPLIT,SPLIT_SHA256,sha
from models import build,NAMES

def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def evaluate(model,ds,y,indices,out,epoch):
    evaluation_start=time.monotonic()
    p=np.concatenate([model(x,training=False).numpy() for x,_ in ds]);p=p.astype(np.float64)
    p/=p.sum(axis=1,keepdims=True)
    truth=y[indices];hot=np.eye(32)[truth];pred=p.argmax(1)
    precision,recall,f1,_=precision_recall_fscore_support(truth,pred,labels=np.arange(32),average='macro',zero_division=0)
    m={'accuracy':float(accuracy_score(truth,pred)),'precision':float(precision),'recall':float(recall),'f1':float(f1),
       'auc':float(roc_auc_score(hot,p,average='macro')),'categorical_crossentropy':float(log_loss(truth,p,labels=np.arange(32))),
       'mse':float(np.mean((p-hot)**2)),'sample_count':len(truth)}
    pm,rm,fm,_=precision_recall_fscore_support(hot.ravel(),(p>.5).ravel(),average='binary',zero_division=0)
    m['diagnostics']={'threshold_0_5_micro_precision':float(pm),'threshold_0_5_micro_recall':float(rm),'threshold_0_5_micro_f1':float(fm),'flattened_roc_auc':float(roc_auc_score(hot.ravel(),p.ravel()))}
    np.savez_compressed(out/f'predictions-epoch{epoch}.npz',indices=indices,labels=truth,probabilities=p)
    m['evaluation_seconds']=time.monotonic()-evaluation_start
    write(out/f'evaluation-epoch{epoch}.json',m)
    return m

def main():
    a=argparse.ArgumentParser();a.add_argument('--model',choices=NAMES,required=True);a.add_argument('--mode',choices=['preflight','full','resume-check'],required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--resume-from',type=Path);a.add_argument('--pins-from',type=Path)
    args=a.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.mode=='full' and (out/'metrics.json').exists():
        completed=json.loads((out/'metrics.json').read_text())
        if completed.get('completed'):
            assert completed['model']==args.model
            print(json.dumps(completed));return
    (out/'source').mkdir(exist_ok=True)
    for source_file in Path('/app').glob('*.py'):shutil.copy2(source_file,out/'source'/source_file.name)
    started=time.monotonic();random.seed(42);np.random.seed(42);tf.random.set_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(8);tf.config.threading.set_inter_op_parallelism_threads(4)
    gpus=tf.config.list_physical_devices('GPU');assert len(gpus)==1,'Expected one GPU'
    for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
    print(json.dumps({'stage':'loading_verified_corpus','model':args.model}),flush=True)
    t,y=corpus();assert sha(SPLIT)==SPLIT_SHA256,'Conditional split changed';split=json.loads(SPLIT.read_text());train=np.array(split['train']);val=np.array(split['validation'])
    assert len(train)==48645 and len(val)==5404 and not set(train)&set(val)
    # Preserve native resolution until final resizing. Normalize RGB source exceptions to grayscale.
    encoded=[];native_sizes={}
    for item in t['image']:
        with Image.open(io.BytesIO(item['bytes'])) as im:
            key=f'{im.width}x{im.height}/{im.mode}';native_sizes[key]=native_sizes.get(key,0)+1
            b=io.BytesIO();im.convert('L').save(b,format='PNG');encoded.append(b.getvalue())
    del t
    pins=None
    if args.mode=='full':
        pinpath=(args.pins_from or Path('/results')/('preflight-'+args.model))/'weights.json'
        assert pinpath.exists(),'Successful preflight weight pins required'
        pins=json.loads(pinpath.read_text())
    print(json.dumps({'stage':'initializing_model','model':args.model}),flush=True)
    model,size,weights=build(args.model,out,pins)
    optimizer=tf.keras.optimizers.Adam(learning_rate=.0001)
    optimizer._create_all_weights(model.trainable_variables)
    epoch=tf.Variable(0,dtype=tf.int64,trainable=False);batch=tf.Variable(0,dtype=tf.int64,trainable=False)
    ckpt=tf.train.Checkpoint(model=model,optimizer=optimizer,epoch=epoch,batch=batch)
    manager=tf.train.CheckpointManager(ckpt,str(out/'checkpoints'),max_to_keep=2)
    source=args.resume_from or out
    latest=tf.train.latest_checkpoint(str(source/'checkpoints'))
    if latest:ckpt.restore(latest).assert_existing_objects_matched()
    def dataset(indices,training=False,epoch_number=0):
        ds=tf.data.Dataset.from_tensor_slices(([encoded[int(i)] for i in indices],y[indices]))
        if training:ds=ds.shuffle(len(indices),seed=42+epoch_number,reshuffle_each_iteration=False)
        def decode(s,label):
            im=tf.io.decode_png(s,channels=1)
            im=tf.image.grayscale_to_rgb(im)
            im=tf.image.resize(im,(size,size),method='bilinear')
            return im,tf.one_hot(label,32)
        return ds.map(decode,num_parallel_calls=8,deterministic=True).batch(32).prefetch(2)
    @tf.function
    def step(x,yhot):
        with tf.GradientTape() as tape:
            p=model(x,training=True)
            loss=tf.reduce_mean(tf.keras.losses.categorical_crossentropy(yhot,p))
        gradients=tape.gradient(loss,model.trainable_variables)
        tf.debugging.assert_all_finite(loss,'Nonfinite loss')
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        return loss
    metadata={'model':args.model,'mode':args.mode,'conditional':True,'seed':42,'batch_size':32,'learning_rate':.0001,'optimizer':'Adam','dtype':'float32','tf32_enabled':tf.config.experimental.tensor_float_32_execution_enabled(),'tensorflow':tf.__version__,'python':sys.version,'train_samples':len(train),'validation_samples':len(val),'split_sha256':sha(SPLIT),'weights':weights,'parameters':model.count_params(),'native_images':native_sizes,'source_hashes':{p.name:sha(p) for p in Path('/app').glob('*.py')},'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if args.mode=='full' and args.resume_from:
        assert source.resolve()!=out.resolve(),'Resume into a fresh output directory'
        previous=json.loads((source/'configuration.json').read_text())
        for field in ['model','seed','batch_size','learning_rate','optimizer','dtype','tf32_enabled','tensorflow','train_samples','validation_samples','split_sha256','weights','parameters']:
            assert metadata[field]==previous[field],f'Resume configuration differs: {field}'
        assert latest,'Resume checkpoint absent'
        assert int(optimizer.iterations.numpy())==int(epoch.numpy())*1521+int(batch.numpy()),'Checkpoint cursor and optimizer disagree'
        metadata['resume']={'source_directory':str(source),'checkpoint':latest,'completed_epochs':int(epoch.numpy()),'batch_offset':int(batch.numpy()),'optimizer_iterations':int(optimizer.iterations.numpy()),'source_configuration_sha256':sha(source/'configuration.json'),'checkpoint_input_sha256':{p.name:sha(p) for p in Path(latest).parent.glob(Path(latest).name+'.*')},'randomness_note':'Parameters, optimizer and recorded epoch/batch cursor restore; bitwise-identical future stochastic/GPU trajectories are not claimed.'}
    write(out/'configuration.json',metadata)
    (out/'environment.txt').write_text(subprocess.check_output(['python','-m','pip','freeze'],text=True)+'\n'+subprocess.check_output(['nvidia-smi'],text=True))
    timings=[]
    if args.mode in ['preflight','resume-check']:
        # Eight examples per class guarantees all 32 labels for tiny-subset AUC.
        tiny=np.concatenate([val[y[val]==c][:8] for c in range(32)])
        ds=dataset(train,True)
        if args.mode=='resume-check':
            assert latest,'Checkpoint absent'
            before=json.loads((source/'metrics.json').read_text())
            assert int(optimizer.iterations.numpy())==before['optimizer_iterations']
            testx,_=next(iter(dataset(tiny[:32])))
            restored_path=ckpt.write(str(out/'restored-state'))
            old_reader=tf.train.load_checkpoint(latest);new_reader=tf.train.load_checkpoint(restored_path)
            keys=[k for k,_ in tf.train.list_variables(latest) if k not in ['_CHECKPOINTABLE_OBJECT_GRAPH','save_counter/.ATTRIBUTES/VARIABLE_VALUE']]
            for key in keys:np.testing.assert_array_equal(old_reader.get_tensor(key),new_reader.get_tensor(key),err_msg=key)
            del old_reader,new_reader
            for file in out.glob('restored-state*'):file.unlink()
            resumed_prob=model(testx,training=False).numpy();reference_prob=np.load(source/'reload-reference.npy')
            max_resume_difference=float(np.max(np.abs(resumed_prob-reference_prob)))
            assert max_resume_difference < .001, 'Restored state matches but inference differs materially'
            print(json.dumps({'restored_tensor_count':len(keys),'all_restored_tensors_exact':True,'max_probability_difference':max_resume_difference}),flush=True)
        count=8 if args.mode=='preflight' else 1
        for i,(x,hot) in enumerate(ds.take(count)):
            s=time.monotonic();loss=float(step(x,hot).numpy());timings.append(time.monotonic()-s)
            print(json.dumps({'step':i,'loss':loss,'seconds':timings[-1]}),flush=True)
        ckpath=manager.save()
        testx,_=next(iter(dataset(tiny[:32])))
        reference=model(testx,training=False).numpy()
        np.save(out/'reload-reference.npy',reference)
        saved=model.trainable_variables[-1].numpy().copy()
        model.trainable_variables[-1].assign_add(tf.ones_like(model.trainable_variables[-1]))
        ckpt.restore(ckpath).assert_existing_objects_matched()
        np.testing.assert_array_equal(model.trainable_variables[-1].numpy(),saved)
        np.testing.assert_allclose(model(testx,training=False).numpy(),reference,rtol=1e-5,atol=1e-6)
        m=evaluate(model,dataset(tiny),y,tiny,out,0)
        m.update(model=args.model,conditional=True,mode=args.mode,step_seconds=timings,steady_step_seconds=float(np.median(timings[3:] or timings)),optimizer_iterations=int(optimizer.iterations.numpy()),checkpoint_reload_verified=True,wall_seconds=time.monotonic()-started,peak_gpu_bytes=tf.config.experimental.get_memory_info('GPU:0')['peak'])
        if args.mode=='resume-check':m.update(all_restored_tensors_exact=True,restored_tensor_count=len(keys),max_probability_difference=max_resume_difference)
        write(out/'metrics.json',m);print(json.dumps(m),flush=True);return
    total_epochs=100 if args.model=='cnn3' else 10
    history=json.loads((source/'metrics.json').read_text()).get('history',[]) if (source/'metrics.json').exists() else []
    history=[row for row in history if row['epoch']<=int(epoch.numpy())]
    if args.resume_from:
        for row in history:
            for filename in [f"predictions-epoch{row['epoch']}.npz",f"evaluation-epoch{row['epoch']}.json"]:
                assert (source/filename).exists(),f'Missing prior epoch evidence: {filename}'
                shutil.copy2(source/filename,out/filename)
    for ep in range(int(epoch.numpy()),total_epochs):
        epstart=time.monotonic();losses=[];offset=int(batch.numpy())
        for i,(x,hot) in enumerate(dataset(train,True,ep).skip(offset),start=offset):
            s=time.monotonic();loss=float(step(x,hot).numpy());timings.append(time.monotonic()-s);losses.append(loss);batch.assign(i+1)
            if (i+1)%200==0:
                manager.save();print(json.dumps({'epoch':ep+1,'batch':i+1,'loss':loss,'seconds_per_batch':float(np.median(timings[-100:]))}),flush=True)
        m=evaluate(model,dataset(val),y,val,out,ep+1)
        history.append({'epoch':ep+1,'training_loss':float(np.mean(losses)) if losses else None,'evaluation':m,'seconds':time.monotonic()-epstart,'training_loss_batches':len(losses),'resumed_batch_offset':offset})
        result={'model':args.model,'conditional':True,'history':history,'wall_seconds':time.monotonic()-started,'optimizer_iterations':int(optimizer.iterations.numpy()),'peak_gpu_bytes':tf.config.experimental.get_memory_info('GPU:0')['peak']}
        write(out/'metrics.json',result)
        epoch.assign(ep+1);batch.assign(0);manager.save()
        print(json.dumps(history[-1]),flush=True)
    result['completed']=True;result['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    write(out/'metrics.json',result)
    # Immutable checksum inventory for retained native artifacts.
    manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['console.log','artifact-hashes.json','modal.json']}
    write(out/'artifact-hashes.json',manifest)
if __name__=='__main__':main()
