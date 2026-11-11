"""Native Detectron2 training with durable latest/best checkpoint bookkeeping."""
import argparse
import hashlib
import json
import math
import statistics
import time
import sys
from pathlib import Path

CONFIG = 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
WEIGHTS = 'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--weights-sha256', required=True)
    parser.add_argument('--weights-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    execution_path = args.output/'execution.json'
    if execution_path.exists():
        execution = json.loads(execution_path.read_text())
        if len(execution.get('segments', [])) > 1 and not (args.output/'recovery.json').exists():
            raise RuntimeError('Interrupted training has no verified durable recovery checkpoint; refusing a fresh COCO restart.')
    manifest_path = args.data_root/'manifest.json'
    assert sha(manifest_path) == args.manifest_sha256
    manifest = json.loads(manifest_path.read_text())
    for split in ('train','val','test'):
        relative = 'coco/full/'+split+'.json'
        assert sha(args.data_root/relative) == manifest['coco_files'][relative]
    import torch
    import modal
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.data import build_detection_test_loader
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import default_setup, hooks
    from detectron2.evaluation import COCOEvaluator
    from detectron2.model_zoo import get_config_file
    from detectron2.utils.events import EventStorage
    sys.path.insert(0,'/opt/detectron2/tools')
    from train_net import Trainer
    torch.set_num_threads(4)
    volume = modal.Volume.from_name('repro-992e7a-results', version=2)
    for split in ('train','val','test'):
        register_coco_instances('arabic_'+split, {}, str(args.data_root/'coco/full'/f'{split}.json'), str(args.data_root))
    steps_per_epoch = math.ceil(len(manifest['splits']['train'])/24)
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file(CONFIG))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 28
    cfg.MODEL.WEIGHTS = str(args.weights_file)
    assert sha(cfg.MODEL.WEIGHTS) == args.weights_sha256
    cfg.INPUT.MIN_SIZE_TRAIN = (416,)
    cfg.INPUT.MAX_SIZE_TRAIN = 416
    cfg.INPUT.MIN_SIZE_TEST = 416
    cfg.INPUT.MAX_SIZE_TEST = 416
    cfg.SOLVER.IMS_PER_BATCH = 24
    cfg.SOLVER.BASE_LR = .001
    cfg.SOLVER.MAX_ITER = steps_per_epoch*60
    cfg.SOLVER.CHECKPOINT_PERIOD = steps_per_epoch
    cfg.TEST.EVAL_PERIOD = steps_per_epoch
    cfg.DATASETS.TRAIN = ('arabic_train',)
    cfg.DATASETS.TEST = ('arabic_val',)
    cfg.SEED = 42
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.OUTPUT_DIR = str(args.output/'model')
    cfg.freeze()
    default_setup(cfg, argparse.Namespace(config_file='',eval_only=False))
    config_sha = hashlib.sha256(cfg.dump().encode()).hexdigest()
    selection_path = args.output/'selection.json'
    recovery_path = args.output/'recovery.json'
    validation_finished_path = args.output/'validation-finished.json'

    class NativeTrainer(Trainer):
        def __init__(self, config):
            super().__init__(config)
            self.checkpointer.add_checkpointable('optimizer',self.optimizer)

        @classmethod
        def build_evaluator(cls, config, dataset_name, output_folder=None):
            return COCOEvaluator(dataset_name,tasks=('bbox',),distributed=False,use_fast_impl=False,
                                 output_dir=output_folder or str(args.output/dataset_name))

    trainer = NativeTrainer(cfg)
    if recovery_path.exists():
        recovery = json.loads(recovery_path.read_text())
        assert recovery['config_sha256'] == config_sha
        assert recovery['manifest_sha256'] == args.manifest_sha256
        latest = args.output/recovery['checkpoint']
        assert sha(latest) == recovery['sha256']
        # A checkpoint interrupted before durable bookkeeping cannot supersede this one.
        (Path(cfg.OUTPUT_DIR)/'last_checkpoint').write_text(latest.name)
        trainer.resume_or_load(resume=True)
        assert trainer.start_iter == recovery['completed_updates']
    else:
        trainer.resume_or_load(resume=False)
    print(json.dumps({'start_iter':trainer.start_iter,'max_iter':cfg.SOLVER.MAX_ITER,
                      'steps_per_epoch':steps_per_epoch,'config_sha256':config_sha,
                      'resume_limit':'Optimizer/scheduler/iteration restored; native RNG/data cursor restart is disclosed.'}),flush=True)

    class SelectionCheckpointer(DetectionCheckpointer):
        def save(self, name, **kwargs):
            iteration = kwargs['iteration']
            metric = float(trainer.storage.latest()['bbox/AP'][0])
            assert math.isfinite(metric)
            unique = f'{name}_{iteration:07d}'
            super().save(unique, **kwargs)
            path = args.output/'best'/(unique+'.pth')
            history = json.loads(selection_path.read_text())['improvements'] if selection_path.exists() else []
            history.append({'iteration':iteration,'validation_bbox_AP':metric,
                            'checkpoint':str(path.relative_to(args.output)),'sha256':sha(path)})
            atomic_json(selection_path,{'config_sha256':config_sha,'manifest_sha256':args.manifest_sha256,
                                        'improvements':history,'selected':history[-1]})
            volume.commit()

    best = hooks.BestCheckpointer(steps_per_epoch,
                                 SelectionCheckpointer(trainer.model,save_dir=str(args.output/'best')),
                                 'bbox/AP',mode='max')
    if selection_path.exists():
        previous = json.loads(selection_path.read_text())
        assert previous['config_sha256'] == config_sha
        assert previous['manifest_sha256'] == args.manifest_sha256
        selected = previous['selected']
        assert sha(args.output/selected['checkpoint']) == selected['sha256']
        best.best_metric = selected['validation_bbox_AP']
        best.best_iter = selected['iteration']

    def preserve_latest(current_trainer):
        completed = trainer.iter+1
        if completed % steps_per_epoch and completed != trainer.max_iter:
            return
        path = Path(trainer.checkpointer.get_checkpoint_file())
        atomic_json(recovery_path,{'config_sha256':config_sha,'manifest_sha256':args.manifest_sha256,
                                  'completed_updates':completed,'checkpoint':str(path.relative_to(args.output)),
                                  'sha256':sha(path),
                                  'resume_limit':'Native RNG and data-loader cursor are not checkpointed.'})
        volume.commit()

    # Native scheduling stays intact. Bookkeeping occurs after the native latest save,
    # then validation; native best selection observes exactly that validation result.
    for index, hook in enumerate(trainer._hooks):
        if isinstance(hook,hooks.PeriodicCheckpointer):
            hook.max_to_keep = 2
            trainer.register_hooks([hooks.CallbackHook(after_step=preserve_latest)])
            trainer._hooks.insert(index+1,trainer._hooks.pop())
            break
    trainer.register_hooks([best])

    def preserve_validation(current_trainer):
        completed = min(trainer.iter+1,trainer.max_iter)
        if completed % steps_per_epoch and completed != trainer.max_iter:
            return
        metric = trainer.storage.latest().get('bbox/AP')
        if metric is None or metric[1] != trainer.storage.iter:
            return  # Do not mark evaluation complete after a failed training/evaluation hook.
        atomic_json(validation_finished_path,{'completed_updates':completed,'config_sha256':config_sha})
        volume.commit()

    trainer.register_hooks([hooks.CallbackHook(after_step=preserve_validation,after_train=preserve_validation)])
    torch.cuda.reset_peak_memory_stats()
    if trainer.start_iter:
        finished_validation = json.loads(validation_finished_path.read_text()) if validation_finished_path.exists() else {}
        if finished_validation.get('completed_updates',0) < trainer.start_iter:
            # The durable latest save precedes validation. Complete an interrupted
            # validation/selection before taking another optimizer step.
            metric_iter = trainer.start_iter if trainer.start_iter == trainer.max_iter else trainer.start_iter-1
            with EventStorage(metric_iter) as storage:
                trainer.storage = storage
                recovery_val = trainer.test(cfg,trainer.model)
                storage.put_scalar('bbox/AP',float(recovery_val['bbox']['AP']),smoothing_hint=False)
                best._best_checking()
            atomic_json(validation_finished_path,{'completed_updates':trainer.start_iter,'config_sha256':config_sha})
            volume.commit()
    if trainer.start_iter < trainer.max_iter:
        trainer.train()
    selected = json.loads(selection_path.read_text())['selected']
    checkpoint = args.output/selected['checkpoint']
    assert sha(checkpoint) == selected['sha256']
    DetectionCheckpointer(trainer.model).load(str(checkpoint))
    test_cfg = cfg.clone()
    test_cfg.defrost()
    test_cfg.DATASETS.TEST = ('arabic_test',)
    test_cfg.freeze()
    test = trainer.test(test_cfg,trainer.model)
    batches = list(build_detection_test_loader(test_cfg,'arabic_test'))
    assert all(len(batch) == 1 and batch[0]['image'].dtype == torch.uint8
               and tuple(batch[0]['image'].shape[-2:]) == (416,416) for batch in batches)
    trainer.model.eval()
    latency=[]
    with torch.no_grad():
        for batch in batches[:5]:trainer.model(batch)
        for batch in batches:
            torch.cuda.synchronize()
            start=time.perf_counter()
            trainer.model(batch)
            torch.cuda.synchronize()
            latency.append(time.perf_counter()-start)
    result = {'selected':selected,'test':test,'all_parameters':sum(p.numel() for p in trainer.model.parameters()),
              'trainable_parameters':sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
              'completed_updates':cfg.SOLVER.MAX_ITER,'steps_per_epoch':steps_per_epoch,'epochs':60,
              'manifest_sha256':args.manifest_sha256,'weights_sha256':args.weights_sha256,'weights_url':WEIGHTS,
              'config_sha256':config_sha,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),
              'timing':{'image_ids':[batch[0]['image_id'] for batch in batches], 'seconds':latency,
                        'mean_seconds':statistics.mean(latency),'median_seconds':statistics.median(latency),
                        'min_seconds':min(latency),'max_seconds':max(latency),'fps':1/statistics.mean(latency),
                        'scope':'FP32 batch1 decoded/resized CPUuint8 through H2D/native BGR normalization/model/postprocessing;5warmups,synchronizedCUDA;excludesfiledecode/render/metrics.'},
              'normalization':{'input_format':cfg.INPUT.FORMAT,'pixel_mean':list(cfg.MODEL.PIXEL_MEAN),
                               'pixel_std':list(cfg.MODEL.PIXEL_STD)}}
    atomic_json(args.output/'result.json',result)
    volume.commit()
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
