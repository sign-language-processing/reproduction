"""Real-data preflight of native Detectron2 Trainer; no full training entry point."""
import argparse
import copy
import gc
import hashlib
import json
import math
import statistics
import shutil
import sys
import time
from pathlib import Path

CONFIG = 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
WEIGHTS = 'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def subset(source, destination, minimum):
    """Keep class coverage, then deterministic additional images; no split reassignment."""
    data = json.loads(source.read_text())
    selected, covered = set(), set()
    image_ids = {x['id'] for x in data['images']}
    for ann in sorted(data['annotations'], key=lambda a: (a['image_id'], a['id'])):
        if ann['category_id'] not in covered:
            selected.add(ann['image_id'])
            covered.add(ann['category_id'])
    assert covered == (set(range(1,29)) - {25}), 'Observed IDs must preserve absent NOON (zero-based24).'
    assert {x['id'] for x in data['categories']} == set(range(1,29))
    for identity in sorted(image_ids):
        if len(selected) >= minimum: break
        selected.add(identity)
    data['images'] = [x for x in data['images'] if x['id'] in selected]
    data['annotations'] = [x for x in data['annotations'] if x['image_id'] in selected]
    data.setdefault('info', {})
    destination.write_text(json.dumps(data)+'\n')
    return len(data['images'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--train-json', type=Path, required=True)
    parser.add_argument('--val-json', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.manifest) == args.manifest_sha256
    manifest = json.loads(args.manifest.read_text())
    for annotation_path in (args.train_json, args.val_json):
        relative = str(annotation_path.resolve().relative_to(args.data_root.resolve()))
        assert sha(annotation_path) == manifest['coco_files'][relative]
    train_data, val_data = [json.loads(p.read_text()) for p in (args.train_json, args.val_json)]
    assert len(train_data['categories']) == len(val_data['categories']) == 28
    assert train_data['categories'] == val_data['categories']
    assert not ({x['file_name'] for x in train_data['images']} & {x['file_name'] for x in val_data['images']})
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {name: subset(source, args.output/(name+'.json'), minimum)
              for name, source, minimum in [('preflight_train', args.train_json, 112), ('preflight_val', args.val_json, 28)]}
    record_hashes = {r['image']: r['image_sha256'] for r in manifest['records']}
    verified_images = set()
    for name in counts:
        data = json.loads((args.output/(name+'.json')).read_text())
        for image in data['images']:
            image_path = (args.data_root/image['file_name']).resolve()
            image_path.relative_to(args.data_root.resolve())
            if image['file_name'] not in verified_images:
                assert sha(image_path) == record_hashes[image['file_name']]
                verified_images.add(image['file_name'])
    import torch
    torch.set_num_threads(4)
    from detectron2.config import get_cfg
    from detectron2.data import build_detection_test_loader
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import default_setup
    from detectron2.evaluation import COCOEvaluator
    from detectron2.model_zoo import get_config_file
    from detectron2.utils.file_io import PathManager
    sys.path.insert(0, '/opt/detectron2/tools')
    from train_net import Trainer

    class NativeTrainer(Trainer):
        def __init__(self, cfg):
            super().__init__(cfg)
            # v0.6 DefaultTrainer checkpoints iteration/hooks but omits its inner optimizer.
            self.checkpointer.add_checkpointable('optimizer', self.optimizer)

        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            return COCOEvaluator(dataset_name, tasks=('bbox',), distributed=False,
                                 output_dir=output_folder or str(args.output/'validation'), use_fast_impl=False)

    for name in counts:
        register_coco_instances(name, {}, str(args.output/(name+'.json')), str(args.data_root))
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file(CONFIG))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 28
    cfg.MODEL.WEIGHTS = PathManager.get_local_path(WEIGHTS)
    shutil.copyfile(cfg.MODEL.WEIGHTS,args.output/'initialization.pkl')
    cfg.MODEL.WEIGHTS = str(args.output/'initialization.pkl')
    cfg.INPUT.MIN_SIZE_TRAIN = (416,)
    cfg.INPUT.MAX_SIZE_TRAIN = 416
    cfg.INPUT.MIN_SIZE_TEST = 416
    cfg.INPUT.MAX_SIZE_TEST = 416
    cfg.SOLVER.IMS_PER_BATCH = 24
    cfg.SOLVER.BASE_LR = .001
    cfg.SOLVER.MAX_ITER = 10
    cfg.SOLVER.CHECKPOINT_PERIOD = 8
    cfg.TEST.EVAL_PERIOD = 8
    cfg.DATASETS.TRAIN = ('preflight_train',)
    cfg.DATASETS.TEST = ('preflight_val',)
    cfg.SEED = 42
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.OUTPUT_DIR = str(args.output/'model')
    cfg.freeze()
    default_setup(cfg, argparse.Namespace(config_file='', eval_only=False))
    result = {'manifest_sha256': args.manifest_sha256, 'train_annotations_sha256': sha(args.train_json),
              'validation_annotations_sha256': sha(args.val_json), 'counts': counts, 'selected_image_hashes_verified': len(verified_images), 'observed_category_ids': sorted({a['category_id'] for a in train_data['annotations']}), 'absent_category_id': 25,
              'weights_url': WEIGHTS, 'weights_sha256': sha(cfg.MODEL.WEIGHTS),
              'normalization': {'input_format': cfg.INPUT.FORMAT, 'pixel_mean': list(cfg.MODEL.PIXEL_MEAN), 'pixel_std': list(cfg.MODEL.PIXEL_STD)},
              'resume_scope': 'Native checkpointer with explicit optimizer registration because v0.6 DefaultTrainer omits it; model/optimizer/scheduler/iteration restore tested. RNG and data cursor are not preserved.',
              'native_source_revision': 'd1e04565d3bec8719335b88be9e9b961bf3ec464'}
    torch.cuda.reset_peak_memory_stats()
    trainer = NativeTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.max_iter = 8  # Stop the preflight at a checkpoint; keep the scheduler horizon at ten.
    result['all_parameters'] = sum(p.numel() for p in trainer.model.parameters())
    result['trainable_parameters'] = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    start = time.monotonic()
    trainer.train()
    result['initial_training_and_validation_seconds'] = time.monotonic()-start
    initial_checkpoint = args.output/'model/model_0000007.pth'
    result['initial_checkpoint_path'] = str(initial_checkpoint.relative_to(args.output))
    result['initial_checkpoint_sha256'] = sha(initial_checkpoint)
    saved = torch.load(initial_checkpoint, map_location='cpu', weights_only=False)
    state = trainer.model.state_dict()
    assert all(torch.equal(value.cpu(), saved['model'][key]) for key, value in state.items())
    result['saved_model_exact'] = True
    initial_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    initial_scheduler = copy.deepcopy(trainer.scheduler.state_dict())
    del state, trainer
    gc.collect()
    torch.cuda.empty_cache()
    resumed_cfg = cfg.clone()
    resumed_cfg.defrost()
    resumed_cfg.SOLVER.MAX_ITER = 10
    resumed_cfg.freeze()
    resumed = NativeTrainer(resumed_cfg)
    resumed.resume_or_load(resume=True)
    assert resumed.start_iter == 8
    assert all(torch.equal(value.cpu(), saved['model'][key]) for key, value in resumed.model.state_dict().items())
    restored_optimizer = resumed.optimizer.state_dict()
    assert initial_optimizer['param_groups'] == restored_optimizer['param_groups']
    assert initial_optimizer['state'].keys() == restored_optimizer['state'].keys()
    for identity, values in initial_optimizer['state'].items():
        for key, value in values.items():
            other = restored_optimizer['state'][identity][key]
            assert torch.equal(value.cpu(), other.cpu()) if torch.is_tensor(value) else value == other
    assert initial_scheduler == resumed.scheduler.state_dict()
    result['resume_model_optimizer_scheduler_exact'] = True
    result['resume_start_iter'] = resumed.start_iter
    del saved, initial_optimizer, restored_optimizer
    resumed.train()
    result['completed_updates'] = 10
    result['validation_metrics'] = resumed._last_eval_results
    result['peak_gpu_memory_bytes'] = torch.cuda.max_memory_allocated()
    loader = build_detection_test_loader(resumed_cfg, 'preflight_val')
    batches = list(loader)
    assert all(len(batch) == 1 for batch in batches)
    assert all(tuple(batch[0]['image'].shape[-2:]) == (416,416) for batch in batches)
    assert all(batch[0]['image'].dtype == torch.uint8 for batch in batches)
    resumed.model.eval()
    samples = []
    with torch.no_grad():
        for batch in batches[:5]: resumed.model(batch)
        for batch in batches:
            torch.cuda.synchronize()
            start = time.perf_counter()
            resumed.model(batch)
            torch.cuda.synchronize()
            samples.append(time.perf_counter()-start)
    result['timing'] = {'scope':'CPU uint8 CHW decoded/resized tensor through H2D/native normalization/model/postprocessing; excludes file decode and evaluation metrics. FP32 batch1, five warmups, synchronized CUDA.',
                        'seconds':samples, 'mean_seconds':statistics.mean(samples),
                        'median_seconds':statistics.median(samples), 'min_seconds':min(samples),
                        'max_seconds':max(samples), 'fps':1/statistics.mean(samples)}
    native_metrics = [json.loads(line) for line in (args.output/'model/metrics.json').read_text().splitlines()]
    result['native_training_step_seconds'] = [x['time'] for x in native_metrics if 'time' in x]
    assert all(math.isfinite(x['total_loss']) for x in native_metrics if 'total_loss' in x)
    result['native_loss_finite'] = True
    result['final_checkpoint_sha256'] = sha(args.output/'model/model_final.pth')
    (args.output/'preflight-result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
