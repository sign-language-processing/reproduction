"""Exact How2 training continuation at completed optimizer boundaries.

This adds durable state to the author's loop; it does not choose checkpoints,
change losses or alter the iterator's ordering. Only batch_multiplier=1 and the
published validation-stepped scheduler are supported by this reproduction.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temporary=Path(str(path)+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    os.replace(str(temporary),str(path))


def selected_checkpoint(root):
    pointer=Path(root)/'latest.json'
    if not pointer.exists():return None
    record=json.loads(pointer.read_text())['latest']
    path=Path(root)/record['file']
    assert digest(path)==record['sha256'],'Incomplete recovery checkpoint'
    return path


def preserve_native(trainer,path):
    if trainer.repro_recovery_dir and Path(path).exists():
        root=Path(trainer.repro_recovery_dir)/'prior-native';root.mkdir(exist_ok=True)
        shutil.copyfile(path,root/(str(time.time_ns())+'-'+Path(path).name))


def save_training(trainer,iterator,epoch,loop=None,complete=False,force=False):
    root=Path(trainer.repro_recovery_dir);root.mkdir(parents=True,exist_ok=True)
    if not (force or trainer.steps==1 or trainer.steps%250==0 or trainer.stop):return
    assert trainer.batch_multiplier==1 and trainer.ckpt_queue.maxsize==1
    assert trainer.scheduler_step_at in ['validation',None]
    previous=json.loads((root/'latest.json').read_text()) if (root/'latest.json').exists() else {}
    backup=None
    if trainer.best_ckpt_iteration:
        source=Path(trainer.model_dir)/(str(trainer.best_ckpt_iteration)+'.ckpt')
        stamp=(trainer.best_ckpt_iteration,source.stat().st_mtime_ns,source.stat().st_size)
        if getattr(trainer,'_repro_best_stamp',None)!=stamp:
            trainer._repro_best_digest=digest(source);trainer._repro_best_stamp=stamp
        backup='best-{}-{}.ckpt'.format(trainer.best_ckpt_iteration,trainer._repro_best_digest)
        if not (root/backup).exists():
            temporary=root/(backup+'.tmp');shutil.copyfile(source,temporary)
            assert digest(temporary)==trainer._repro_best_digest
            os.replace(str(temporary),str(root/backup))
    state={key:getattr(trainer,key) for key in ['steps','total_txt_tokens','total_gls_tokens','best_ckpt_score','best_all_ckpt_scores','best_ckpt_iteration']}
    state.update(model_state=trainer.model.state_dict(),optimizer_state=trainer.optimizer.state_dict(),
                 scheduler_state=trainer.scheduler.state_dict() if trainer.scheduler is not None else None)
    state['repro_state']=dict(epoch=epoch,iterator=iterator.state_dict(),
        shuffler_state=iterator.random_shuffler.random_state,
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if trainer.use_cuda else None,
        last_best_lr=trainer.last_best_lr,stop=trainer.stop,
        loop=loop or {},complete=complete,best_backup=backup,
        best_backup_sha256=trainer._repro_best_digest if backup else None)
    filename='state-{:08d}-{}.ckpt'.format(trainer.steps,time.time_ns())
    temporary=root/(filename+'.tmp')
    with temporary.open('wb') as stream:
        torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(str(temporary),str(root/filename))
    record=dict(file=filename,sha256=digest(root/filename),steps=trainer.steps,epoch=epoch,
                complete=complete,best_backup=backup)
    pointer=dict(latest=record,previous=previous.get('latest'))
    atomic_json(root/'latest.json',pointer)
    keep={r['file'] for r in pointer.values() if r}
    keep.update(r['best_backup'] for r in pointer.values() if r and r['best_backup'])
    for path in root.glob('*.ckpt'):
        if path.name not in keep:path.unlink()
    print(json.dumps({'durable_training_checkpoint':record}),flush=True)


def restore_training(trainer,iterator):
    assert trainer.batch_multiplier==1 and trainer.ckpt_queue.maxsize==1
    assert trainer.scheduler_step_at in ['validation',None]
    state=getattr(trainer,'repro_state',None)
    if state is None:
        save_training(trainer,iterator,0,force=True)
        return 0,{},False
    if state['iterator']['random_state_this_epoch'] is not None:
        iterator.load_state_dict(state['iterator'])
    else:
        iterator.random_shuffler.random_state=state['shuffler_state']
    trainer.last_best_lr=state['last_best_lr'];trainer.stop=state['stop']
    if state['best_backup']:
        source=Path(trainer.repro_recovery_dir)/state['best_backup']
        assert digest(source)==state['best_backup_sha256']
        destination=Path(trainer.model_dir)/(str(trainer.best_ckpt_iteration)+'.ckpt')
        if destination.exists() and digest(destination)!=state['best_backup_sha256']:
            preserve_native(trainer,destination)
        shutil.copyfile(source,destination)
        link=Path(trainer.model_dir)/'best.ckpt'
        if link.is_symlink() or link.exists():link.unlink()
        link.symlink_to(destination.name)
        trainer.ckpt_queue.put(str(destination))
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'].cpu())
    if trainer.use_cuda:torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    print(json.dumps({'restored_training_cursor':{'epoch':state['epoch'],'steps':trainer.steps,
          'batches':state['iterator']['iterations_this_epoch'],'complete':state['complete']}}),flush=True)
    return state['epoch'],state['loop'],state['complete'] or trainer.stop
