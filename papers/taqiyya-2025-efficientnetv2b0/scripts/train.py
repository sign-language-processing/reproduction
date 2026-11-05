"""Reconstruction of the four Table X architectures, seed 42.

This does not claim to recover the unpublished split or augmentation samples.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

p=argparse.ArgumentParser()
p.add_argument('--model',choices=['efficientnetv2b0','mobilenetv2','resnet50v2','convnexttiny'],required=True)
p.add_argument('--preflight',action='store_true')
p.add_argument('--out',required=True)
a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
if (out/'metrics.json').exists():
    print((out/'metrics.json').read_text());raise SystemExit()
started=time.time();start_utc=dt.datetime.now(dt.timezone.utc).isoformat()
tf.keras.utils.set_random_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(8)
for gpu in tf.config.list_physical_devices('GPU'): tf.config.experimental.set_memory_growth(gpu,True)
assert tf.config.list_physical_devices('GPU'), 'TensorFlow GPU unavailable'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
root=Path('/datasets/mavi-27-class')
# One local copy avoids remote Volume page faults during randomized batches.
import shutil
local=Path('/tmp/mavi-X.npy')
shutil.copyfile(root/'X.npy',local)
assert sha(local)==json.loads((root/'manifest.json').read_text())['files']['X.npy']['sha256']
x=np.load(local,mmap_mode='r',allow_pickle=False)
y=np.load(root/'Y.npy',allow_pickle=False).reshape(-1)
classes=np.unique(y)
# Table III adds exactly 551 examples. Only NULL is undersized (314); augment
# NULL to 865. Author-side sample selection and random seed were not released.
null=np.where(y=='NULL')[0]
aug=tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20,width_shift_range=.2,height_shift_range=.2,zoom_range=.2,horizontal_flip=True)
rng=np.random.RandomState(42)
# Figure 1 and Section II.B: resize before stochastic augmentation.
with tf.device('/CPU:0'):
    resized_null=tf.image.resize(np.array(x[rng.choice(null,551,replace=True)]),[224,224],method='bilinear').numpy()
extra=np.stack([aug.random_transform(im,seed=42+j) for j,im in enumerate(resized_null)])
del resized_null
labels=np.searchsorted(classes,np.concatenate([y,np.repeat('NULL',551)]))
ids=np.arange(len(labels))
trainval,test=train_test_split(ids,test_size=2336,random_state=42,stratify=labels)
train,val=train_test_split(trainval,test_size=2335,random_state=42,stratify=labels[trainval])
assert (len(train),len(val),len(test))==(18681,2335,2336)
np.savez(out/'split.npz',train=train,val=val,test=test)
def batch_images(indices):
    indices=np.asarray(indices)
    # Resize original images in a vectorized CPU batch; generated images are
    # already resized before augmentation and must not be resized a second time.
    original=indices<len(x)
    images=np.empty((len(indices),224,224,3),dtype=np.float32)
    if original.any():
        with tf.device('/CPU:0'):
            images[original]=tf.image.resize(np.array(x[indices[original]]),[224,224],method='bilinear').numpy()
    images[~original]=extra[indices[~original]-len(x)]
    return images
def dataset(indices,training=False):
    d=tf.data.Dataset.from_tensor_slices((indices,labels[indices]))
    if training:d=d.shuffle(len(indices),seed=42,reshuffle_each_iteration=True)
    d=d.batch(16)
    def load(i,l):
        images=tf.numpy_function(batch_images,[i],tf.float32)
        images.set_shape([None,224,224,3])
        images=images*255.
        if a.model in ('mobilenetv2','resnet50v2'):images=images/127.5-1.
        return images,tf.one_hot(l,len(classes))
    return d.map(load,num_parallel_calls=2).prefetch(2)
constructors={'efficientnetv2b0':tf.keras.applications.EfficientNetV2B0,'mobilenetv2':tf.keras.applications.MobileNetV2,'resnet50v2':tf.keras.applications.ResNet50V2,'convnexttiny':tf.keras.applications.ConvNeXtTiny}
expected={'efficientnetv2b0':6254187,'mobilenetv2':2592859,'resnet50v2':24096283,'convnexttiny':28023931}
state_path=out/'state.json'; checkpoint=out/'last.keras'
state=json.loads(state_path.read_text()) if state_path.exists() else {'epoch':0,'best':-1.,'wait':0,'history':[]}
if checkpoint.exists():model=tf.keras.models.load_model(checkpoint)
else:
    base=constructors[a.model](include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
    h=tf.keras.layers.Dense(256,activation='relu')(base.output)
    h=tf.keras.layers.Dropout(.2)(h)
    outputs=tf.keras.layers.Dense(27,activation='softmax')(h)
    model=tf.keras.Model(base.input,outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),loss='categorical_crossentropy',metrics=['accuracy'],jit_compile=False)
assert model.count_params()==expected[a.model],(model.count_params(),expected[a.model])
(out/'model.json').write_text(model.to_json())
(out/'freeze.txt').write_text(subprocess.check_output(['/opt/taqiyya/bin/pip','freeze'],text=True))
(out/'hardware.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
weights={str(f.relative_to(Path(os.environ['KERAS_HOME']))):sha(f) for f in (Path(os.environ['KERAS_HOME'])/'models').glob('*') if f.is_file()}
if a.preflight:
    # Include both original and generated examples in every preflight partition.
    def tiny(part,n):
        return np.concatenate([part[part<len(x)][:n-4],part[part>=len(x)][:4]])
    train=tiny(train,64);val=tiny(val,54);test=tiny(test,54)
train_ds=dataset(train,True);val_ds=dataset(val);test_ds=dataset(test)
epoch_times=[]
class Record(tf.keras.callbacks.Callback):
    def on_epoch_begin(self,epoch,logs=None):self.start=time.time()
    def on_epoch_end(self,epoch,logs=None):
        epoch_times.append(time.time()-self.start)
        current=float(logs['val_accuracy'])
        if current>state['best']:
            state['best']=current;state['wait']=0;model.save(out/'best.keras')
        else:state['wait']+=1
        state['epoch']=epoch+1;state['history'].append({k:float(v) for k,v in logs.items()})
        model.save(checkpoint)
        state_path.write_text(json.dumps(state,indent=2)+'\n')
        if state['wait']>=5:model.stop_training=True
rec=Record()
if state['wait']<5:
    model.fit(train_ds,validation_data=val_ds,initial_epoch=state['epoch'],epochs=2 if a.preflight else 50,callbacks=[rec],verbose=2)
model=tf.keras.models.load_model(out/'best.keras')
pred=model.predict(test_ds,verbose=0)
reloaded=tf.keras.models.load_model(out/'best.keras')
reload_pred=reloaded.predict(test_ds.take(1),verbose=0)
assert np.allclose(pred[:len(reload_pred)],reload_pred,rtol=1e-5,atol=1e-6)
resume_verified=False
if a.preflight:
    resumed=tf.keras.models.load_model(checkpoint)
    before=int(resumed.optimizer.iterations.numpy())
    for inputs,targets in train_ds.take(1):resumed.train_on_batch(inputs,targets)
    assert int(resumed.optimizer.iterations.numpy())==before+1
    resume_verified=True
truth=labels[test];prediction=np.argmax(pred,axis=1)
np.savez(out/'predictions.npz',indices=test,truth=truth,prediction=prediction,probabilities=pred)
metric={'model':a.model,'preprocessing_order':'bilinear resize 128 to 224, then augmentation','reconstructed_recipe':True,'preflight':a.preflight,'test_accuracy':float(np.mean(prediction==truth)),'correct':int(np.sum(prediction==truth)),'test_count':len(test),'parameter_count':model.count_params(),'seed':42,'optimizer':'Adam','learning_rate':1e-5,'dropout':.2,'dense_units':256,'batch_size':16,'maximum_epochs':50,'early_stopping_patience':5,'checkpoint_selection':'maximum validation accuracy','epochs_completed':state['epoch'],'validation_best_accuracy':state['best'],'epoch_seconds':epoch_times,'started_at_utc':start_utc,'finished_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'wall_seconds':time.time()-started,'peak_gpu_memory':tf.config.experimental.get_memory_info('GPU:0'),'checkpoint_reload_verified':True,'optimizer_resume_verified':resume_verified,'tensorflow':tf.__version__,'keras':tf.keras.__version__,'weights_sha256':weights,'dataset_manifest_sha256':sha(root/'manifest.json'),'train_script_sha256':sha(__file__),'files':{f.name:sha(f) for f in out.iterdir() if f.is_file()}}
(out/'metrics.json').write_text(json.dumps(metric,indent=2)+'\n');print(json.dumps(metric))
