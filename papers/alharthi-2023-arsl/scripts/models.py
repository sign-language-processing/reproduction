"""Model choices approved as conditional assumptions, not recovered author code."""
import json,os,hashlib
from pathlib import Path
import tensorflow as tf
from data import sha
NAMES=['cnn3','mobilenet','mobilenetv2','xception','inceptionv3','inceptionresnetv2','densenet169','bit-r50x1','bit-r50x3','vit-b16','vit-l32','swinv2-tiny','vgg16','vgg19','resnet50','resnet152']

def build(name,output,pins=None):
    """Hash downloaded weights before training; enforce preflight pins in full runs."""
    observed=[]
    original=tf.keras.utils.get_file
    def get_file(*args,**kwargs):
        kwargs.setdefault('cache_dir',os.environ['KERAS_HOME'])
        path=original(*args,**kwargs)
        if Path(path).is_file():
            item={'url':kwargs.get('origin',args[1] if len(args)>1 else None),'sha256':sha(path),'filename':Path(path).name}
            if pins is not None:assert item in pins, f'Unpinned/changed weights: {item}'
            observed.append(item)
        return path
    tf.keras.utils.get_file=get_file
    import keras.utils.data_utils
    keras.utils.data_utils.get_file=get_file
    L=tf.keras.layers
    size=256 if name=='swinv2-tiny' else 224
    inputs=L.Input((size,size,3),name='rgb_pixels')
    if name=='cnn3':
        x=L.Rescaling(1/255.)(inputs)
        x=L.RandomFlip('horizontal_and_vertical',seed=42)(x)
        x=L.RandomRotation(0.2/360.,seed=43)(x)
        for i,n in enumerate([256,126,64,62,16]):
            x=L.Conv2D(n,3,padding='same',name=f'conv{i+1}')(x)
            if i==4:x=L.BatchNormalization()(x)
            x=L.LeakyReLU(alpha=0.01)(x)
            x=L.MaxPool2D(2)(x)
        x=L.Dropout(.5,seed=44)(x);x=L.Flatten()(x);x=L.Dense(128,activation='relu')(x)
    elif name.startswith('bit-'):
        import tensorflow_hub as hub
        handle='https://tfhub.dev/google/bit/m-'+name[4:]+'/1'
        resolved=Path(hub.resolve(handle))
        files=[{'path':str(p.relative_to(resolved)),'sha256':sha(p)} for p in sorted(resolved.rglob('*')) if p.is_file()]
        item={'url':handle,'sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),'files':files}
        if pins is not None:assert item in pins,'BiT SavedModel bytes changed'
        observed.append(item)
        backbone=hub.KerasLayer(str(resolved),trainable=True,name='bit_backbone')
        x=backbone(inputs/255.)
    elif name.startswith('vit-'):
        from vit_keras import vit
        fn=vit.vit_b16 if name=='vit-b16' else vit.vit_l32
        backbone=fn(image_size=224,pretrained=True,include_top=False,pretrained_top=False,weights='imagenet21k+imagenet2012')
        x=backbone(vit.preprocess_inputs(inputs))
    elif name=='swinv2-tiny':
        from keras_cv_attention_models import swin_transformer_v2,download_and_load
        download_and_load.get_file=get_file
        backbone=swin_transformer_v2.SwinTransformerV2Tiny_window16(input_shape=(256,256,3),num_classes=0,pretrained='imagenet')
        assert observed,'Swin library silently failed to download pretrained weights'
        # kcam 1.3.17 split the archive's positional MLP into two Dense layers.
        # Restore the unchanged tensors explicitly; audit every backbone tensor.
        import h5py,numpy as np
        weight_path=Path(os.environ['KERAS_HOME'])/'models'/observed[-1]['filename']
        verified=0
        with h5py.File(weight_path) as archive:
            groups=archive['model_weights']
            for layer in backbone.layers:
                if not layer.weights:continue
                if '_meta_dense_' in layer.name:
                    prefix,part=layer.name.rsplit('meta_dense_',1)
                    group=groups[prefix+'pos_emb']
                    keys=group.attrs['weight_names']
                    values=[np.array(group[key]) for key in keys]
                    values=values[:2] if part=='1' else values[2:]
                    layer.set_weights(values)
                else:
                    group=groups[layer.name]
                    values=[np.array(group[key]) for key in group.attrs['weight_names']]
                assert len(values)==len(layer.weights),layer.name
                for actual,expected in zip(layer.get_weights(),values):
                    np.testing.assert_array_equal(actual,expected,err_msg=layer.name)
                    verified+=1
        print(json.dumps({'swin_pretrained_tensors_verified':verified}),flush=True)
        x=backbone((inputs/255.-tf.constant([.485,.456,.406]))/tf.constant([.229,.224,.225]))
        if len(x.shape)==4:x=L.GlobalAveragePooling2D()(x)
    else:
        module,cls={'mobilenet':('mobilenet','MobileNet'),'mobilenetv2':('mobilenet_v2','MobileNetV2'),
            'xception':('xception','Xception'),'inceptionv3':('inception_v3','InceptionV3'),
            'inceptionresnetv2':('inception_resnet_v2','InceptionResNetV2'),'densenet169':('densenet','DenseNet169'),
            'vgg16':('vgg16','VGG16'),'vgg19':('vgg19','VGG19'),'resnet50':('resnet','ResNet50'),'resnet152':('resnet','ResNet152')}[name]
        mod=getattr(tf.keras.applications,module)
        backbone=getattr(mod,cls)(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
        x=backbone(mod.preprocess_input(inputs))
    model=tf.keras.Model(inputs,L.Dense(32,activation='softmax',name='arsl_classifier')(x))
    model.trainable=True
    if name!='cnn3':assert observed,'No pretrained provenance captured'
    (output/'weights.json').write_text(json.dumps(observed,indent=2))
    return model,size,observed
