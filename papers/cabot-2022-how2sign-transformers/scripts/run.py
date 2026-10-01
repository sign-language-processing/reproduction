"""Invoke the cited author recipe with path changes and a bounded diagnostic mode."""
import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
sys.path.insert(0,'/slt')
import torch
import yaml
from signjoey.training import train

mode, output = sys.argv[1], Path(sys.argv[2])
segment_dir=output/'segments'/sys.argv[3] if len(sys.argv)>3 else output
config = yaml.safe_load(Path('/slt/configs/sign.yaml').read_text())
config['data']['data_path'] = '/datasets/rwth-phoenix-2014-t/features/author'
config['training']['model_dir'] = str(output/'model')
is_how2=mode.startswith('how2-')
if is_how2:
    mode=mode[len('how2-'):]
    config['data'].update(data_path='/datasets/how2sign/spot-align-wicv2023/slt-format',train='train.pkl.gz',dev='val.pkl.gz',test='test.pkl.gz',version='how2sign')
    config['training'].update(batch_size=16,validation_freq=1000,epochs=13)
    config['model']['encoder'].update(num_layers=3,num_heads=4)
    config['model']['decoder'].update(num_layers=3,num_heads=4)
    config['testing']['recognition_beam_sizes']=[10]
if mode=='preflight':
    config['data'].update(random_train_subset=96,random_dev_subset=16)
    if is_how2:
        config['data'].update(dev='val-preflight.pkl.gz',test='test-preflight.pkl.gz',random_dev_subset=-1)
    config['training'].update(epochs=1,validation_freq=1,logging_freq=1)
    config['testing'].update(recognition_beam_sizes=[10] if is_how2 else [1],translation_beam_sizes=[1],translation_beam_alphas=[-1])
recover=is_how2 and mode=='full'
if recover:
    from recovery_train import selected_checkpoint
    config['training']['repro_recovery_dir']=str(output/'recovery')
    # Preserve the first scientific configuration; replay changes only recovery keys.
    original_path=output/'config.yaml'
    if original_path.exists():
        assert yaml.safe_load(original_path.read_text())==config
    else:original_path.write_text(yaml.safe_dump(config))
    checkpoint=selected_checkpoint(output/'recovery')
    if checkpoint is not None:
        config['training'].update(load_model=str(checkpoint),repro_resume=True)
    elif (output/'model').exists():
        # The initial durable snapshot precedes the first optimizer update.
        # A directory without it therefore contains initialization evidence only.
        (output/'model').rename(segment_dir/'interrupted-initialization-model')
config_path=segment_dir/'config.yaml'
if config_path==output/'config.yaml' and recover:pass
else:config_path.write_text(yaml.safe_dump(config))
start=time.monotonic()
if is_how2:
    import tensorflow as tf
    import signjoey.training as training
    from parallel_ctc import ParallelCTC
    from evaluation import with_recognition_cache
    torch.set_num_threads(4)
    original_decoder=tf.nn.ctc_beam_search_decoder
    original_test=training.test
    decoder=ParallelCTC(original_decoder,workers=4)
    tf.nn.ctc_beam_search_decoder=decoder
    training.test=with_recognition_cache(original_test)
    if recover:
        from recovery_eval import with_evaluation_recovery
        training.test=with_evaluation_recovery(training.test,cache_root=output/'evaluation-cache',
            dataset_identity={'manifest_sha256':'fe0d41ea4877a54f2be7b6601e69fa50da1ecec9f93b179d505123f8276ec48b'})
    try:
        train(str(config_path))
    finally:
        training.test=original_test
        tf.nn.ctc_beam_search_decoder=original_decoder
        decoder.close()
else:
    train(str(config_path))
if mode=='preflight':
    checkpoint=output/'resume.ckpt'
    shutil.copy2(output/'model/best.ckpt',checkpoint)
    config['training']['load_model']=str(checkpoint)
    config['training']['reset_best_ckpt']=True
    config['training']['model_dir']=str(output/'resumed-model')
    config_path=output/'resume-config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    torch.cuda.empty_cache()
    subprocess.run([sys.executable,"-m","signjoey","train",str(config_path)],check=True,env=dict(os.environ,PYTHONPATH="/opt:/slt"))
result=dict(mode=('how2-' if is_how2 else '')+mode,elapsed_seconds=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated(),torch_version=torch.__version__,cuda_version=torch.version.cuda,device=torch.cuda.get_device_name(0),checkpoint_resume_tested=mode=='preflight')
if recover:
    result['segment_id']=segment_dir.name
    result['timing_scope']='Current execution segment only; aggregate wall-time bound is execution.json original deadline.'
    (segment_dir/'runtime.json').write_text(json.dumps(result,indent=2))
(output/'runtime.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
