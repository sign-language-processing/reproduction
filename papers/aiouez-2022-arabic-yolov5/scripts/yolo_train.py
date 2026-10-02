"""Invoke pinned native YOLOv5 training, resume and held-out evaluation."""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics
import sys
import time
import urllib.request

import numpy as np
from PIL import Image
import torch
import yaml

sys.path.insert(0, '/opt/yolov5')
import train
import val
from models.experimental import attempt_load
from utils.callbacks import Callbacks
from utils.general import non_max_suppression

RECIPES = {'s': (16, 60, .015), 'm': (16, 50, .01), 'l': (24, 50, .01)}


class PreflightCheckpointReady(Exception):
    """Intentional boundary after native epoch1 checkpoint save, before epoch2."""


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=RECIPES, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume-check', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--weights-sha256')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'metrics.json').exists():
        print((args.output / 'metrics.json').read_text())
        return
    if not args.preflight:
        assert args.weights_sha256, 'Full training requires the preflight-pinned COCO weight checksum'
        if (args.output / 'train').exists() and not args.resume:
            raise RuntimeError('Incomplete prior training exists; explicit checkpoint resume required')
    batch, epochs, lr = RECIPES[args.model]
    epochs = 3 if args.preflight else epochs
    dataset = yaml.safe_load(args.data.read_text())
    assert dataset['nc'] == len(dataset['names']) == 28
    for split in ['train', 'val', 'test']:
        assert Path(dataset[split]).is_file()
    train_count = len(Path(dataset['train']).read_text().splitlines())
    batch_count = math.ceil(train_count / batch)
    weight = Path('/cache/huggingface/yolov5-v6.0') / f'yolov5{args.model}.pt'
    weight.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://github.com/ultralytics/yolov5/releases/download/v6.0/{weight.name}'
    if not weight.exists():
        temporary = weight.with_suffix('.part')
        urllib.request.urlretrieve(url, temporary)
        assert temporary.stat().st_size > 1_000_000
        temporary.replace(weight)
    weight_hash = sha(weight)
    if args.weights_sha256:
        assert weight_hash == args.weights_sha256, 'COCO weight bytes changed'
    hyp = yaml.safe_load(Path('/opt/yolov5/data/hyps/hyp.scratch.yaml').read_text())
    hyp['lr0'] = lr
    hyp_path = args.output / 'hyp.yaml'
    hyp_path.write_text(yaml.safe_dump(hyp))
    opt = train.parse_opt(known=True)
    opt.data, opt.weights, opt.hyp = str(args.data), str(weight), str(hyp_path)
    opt.cfg, opt.batch_size, opt.epochs, opt.imgsz = '', batch, epochs, 416
    opt.project, opt.name, opt.exist_ok = str(args.output), 'train', True
    opt.device, opt.workers, opt.freeze = '0', 4, 0
    opt.save_period = 1 if args.preflight else -1
    opt.resume = False
    callbacks = Callbacks()
    # Native Callbacks uses class-level storage; this process has only one training invocation.
    warm_times = []
    observed_shapes = set()
    epoch_counts = {}
    validation_started = {}
    validation_seconds = {}
    previous = None
    expected_resume = None
    resume_proof = {}

    def batch_end(ni, model, images, targets, paths, plots, sync_bn):
        nonlocal previous
        torch.cuda.synchronize()
        now = time.monotonic()
        observed_shapes.add(tuple(images.shape))
        epoch = ni // batch_count
        epoch_counts[epoch] = epoch_counts.get(epoch, 0) + len(images)
        if expected_resume is not None and 'first_resumed_batch_ni' not in resume_proof:
            assert ni == (expected_resume['epoch'] + 1) * batch_count
            resume_proof['first_resumed_batch_ni'] = ni
            resume_proof['first_resumed_epoch'] = epoch
            (args.output / 'resume-proof.json').write_text(json.dumps(resume_proof, indent=2) + '\n')
        if previous is not None and ni % batch_count != 0:
            warm_times.append(now - previous)
        previous = now

    callbacks.register_action('on_train_batch_end', callback=batch_end)

    def epoch_end(epoch):
        assert epoch_counts[epoch] == train_count, 'Native loader sample count changed'
        torch.cuda.synchronize()
        validation_started[epoch] = time.monotonic()

    callbacks.register_action('on_train_epoch_end', callback=epoch_end)

    def validation_end(log_vals, epoch, best_fitness, fi):
        torch.cuda.synchronize()
        validation_seconds[epoch] = time.monotonic() - validation_started[epoch]

    callbacks.register_action('on_fit_epoch_end', callback=validation_end)

    def equal_state(actual, expected):
        if isinstance(expected, torch.Tensor):
            assert torch.equal(actual.detach().cpu(), expected.detach().cpu().to(actual.dtype))
        elif isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected:
                equal_state(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            assert len(actual) == len(expected)
            for left, right in zip(actual, expected):
                equal_state(left, right)
        else:
            assert actual == expected

    def verify_native_restore():
        if expected_resume is None:
            return
        # The published hook has no arguments. Inspect its native caller read-only;
        # no upstream training function or optimizer is replaced.
        frame = inspect.currentframe().f_back
        while frame is not None and not (frame.f_code.co_name == 'train' and
                                          frame.f_code.co_filename.endswith('/yolov5/train.py')):
            frame = frame.f_back
        assert frame is not None, 'Pinned native training caller not found'
        native = frame.f_locals
        assert native['start_epoch'] == expected_resume['epoch'] + 1
        equal_state(native['model'].state_dict(), expected_resume['model'].state_dict())
        equal_state(native['optimizer'].state_dict(), expected_resume['optimizer'])
        equal_state(native['ema'].ema.state_dict(), expected_resume['ema'].state_dict())
        assert native['ema'].updates == expected_resume['updates']
        resume_proof.update({'loaded_model_state_equal': True, 'loaded_optimizer_state_equal': True,
                             'loaded_ema_state_equal': True, 'loaded_ema_updates': native['ema'].updates,
                             'native_start_epoch': native['start_epoch']})
        del frame, native

    callbacks.register_action('on_pretrain_routine_end', callback=verify_native_restore)

    def stop_at_resume_boundary(last, epoch, final_epoch, best_fitness, fi):
        if args.preflight and args.model == 's' and not args.resume_check and epoch == 1:
            raise PreflightCheckpointReady()

    callbacks.register_action('on_model_save', callback=stop_at_resume_boundary)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    if args.resume_check or args.resume:
        checkpoint = args.output / ('train/weights/epoch1.pt' if args.resume_check else 'train/weights/last.pt')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['epoch'] >= 0 and state['optimizer']['state'] and state['ema'] is not None
        if args.resume_check:
            assert state['epoch'] == 1
        (args.output / 'resume-input.json').write_text(json.dumps({
            'checkpoint_sha256': sha(checkpoint), 'epoch': state['epoch'],
            'optimizer_state_entries': len(state['optimizer']['state']), 'ema_updates': state['updates'],
        }, indent=2) + '\n')
        expected_resume = state
        opt.resume = str(checkpoint)
    try:
        train.main(opt, callbacks=callbacks)
    except PreflightCheckpointReady:
        assert args.preflight and args.model == 's' and not args.resume_check
    if expected_resume is not None:
        assert resume_proof['first_resumed_epoch'] == expected_resume['epoch'] + 1
    torch.cuda.synchronize()
    train_seconds = time.monotonic() - started
    if args.preflight and args.model == 's' and not args.resume_check:
        # The caller launches a fresh process to exercise native --resume from epoch1.pt.
        result = {'phase': 'pre-resume', 'training_seconds': train_seconds,
                  'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
                  'mean_warm_batch_seconds': statistics.mean(warm_times),
                  'epoch_validation_seconds': validation_seconds,
                  'observed_training_shapes': sorted(observed_shapes),
                  'weights_url': url, 'weights_sha256': weight_hash}
        (args.output / 'pre-resume.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return
    checkpoint = args.output / 'train/weights/best.pt'
    evaluation = args.output / 'test'
    evaluation.mkdir(exist_ok=True)
    test_shapes = set()

    def record_test_shape(pred, predn, path, names, image):
        test_shapes.add(tuple(image.shape))

    callbacks.register_action('on_val_image_end', callback=record_test_shape)
    eval_started = time.monotonic()
    values, maps, timings = val.run(str(args.data), weights=str(checkpoint), batch_size=batch,
                                   imgsz=416, task='test', device='0', project=str(args.output),
                                   name='test', exist_ok=True,
                                   save_txt=True, save_conf=True, save_json=True, plots=True,
                                   half=True, verbose=True, callbacks=callbacks)
    torch.cuda.synchronize()
    eval_seconds = time.monotonic() - eval_started
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    ground_truth = COCO(str(args.data.parent / 'coco-test.json'))
    native_prediction_path = evaluation / 'best_predictions.json'
    native_predictions = json.loads(native_prediction_path.read_text()) if native_prediction_path.exists() else []
    image_ids = set(ground_truth.getImgIds())
    common_predictions = []
    for prediction in native_predictions:
        assert prediction['image_id'] in image_ids and 0 <= prediction['category_id'] < 28
        common_predictions.append({**prediction, 'category_id': prediction['category_id'] + 1})
    (evaluation / 'common-coco-predictions.json').write_text(json.dumps(common_predictions) + '\n')
    if common_predictions:
        detections = ground_truth.loadRes(common_predictions)
    else:
        detections = COCO()
        detections.dataset = {'images': ground_truth.dataset['images'],
                              'categories': ground_truth.dataset['categories'], 'annotations': []}
        detections.createIndex()
    coco_evaluator = COCOeval(ground_truth, detections, 'bbox')
    coco_evaluator.params.imgIds = sorted(image_ids)
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    np.savez_compressed(evaluation / 'common-coco-curves.npz',
                        precision=coco_evaluator.eval['precision'], recall=coco_evaluator.eval['recall'],
                        scores=coco_evaluator.eval['scores'])
    model = attempt_load(str(checkpoint), map_location=torch.device('cuda')).float().eval()
    # Unfused native architecture parameter count, before attempt_load's default fusion.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    original_model = saved['ema'] if saved.get('ema') is not None else saved['model']
    parameter_count = sum(p.numel() for p in original_model.parameters())
    timing_split = 'val' if args.preflight else 'test'
    test_paths = Path(dataset[timing_split]).read_text().splitlines()
    cpu_tensors = []
    for path in test_paths:
        with Image.open(path) as image:
            assert image.size == (416, 416)
            cpu_tensors.append(torch.from_numpy(np.asarray(image.convert('RGB')).copy())
                               .permute(2, 0, 1).contiguous().unsqueeze(0))
    latency = []
    with torch.inference_mode():
        for index in range(5 + len(cpu_tensors)):
            tensor = cpu_tensors[index if index < 5 else index - 5]
            torch.cuda.synchronize()
            clock = time.perf_counter()
            tensor = tensor.to('cuda').float() / 255.0
            prediction = model(tensor)[0]
            non_max_suppression(prediction, conf_thres=.25, iou_thres=.45, max_det=300)
            torch.cuda.synchronize()
            if index >= 5:
                latency.append(time.perf_counter() - clock)
    result = {'model': f'yolov5{args.model}', 'preflight': args.preflight,
              'native_training_seconds_this_process': train_seconds,
              'evaluation_seconds': eval_seconds, 'native_test_metrics': list(values),
              'raw_native_maps_including_absent_class_fallback': maps.tolist(),
              'per_class_ap': [float(value) if ground_truth.getAnnIds(catIds=[i + 1]) else None
                               for i, value in enumerate(maps)],
              'per_class_note': 'Native maps fills unsupported classes with global mAP; NOON has no ground truth and is reported null rather than as a measured per-class score.',
              'native_mean_timing_ms': list(timings),
              'observed_test_image_shapes': sorted(test_shapes),
              'common_pycocotools_stats': coco_evaluator.stats.tolist(),
              'common_evaluator_note': 'pycocotools 2.0.11 bbox COCOeval with default maxDets=100; category IDs shifted from 0..27 to 1..28; identical held-out image IDs and boxes as Faster R-CNN. Native YOLO JSON rounds boxes to 3 decimals and scores to 5 decimals; native AP remains separate. Category 25 (NOON) has no ground truth and standard COCO averaging excludes it.',
              'parameters_unfused': parameter_count,
              'matched_batch1_inference_seconds': {'split': timing_split, 'image_paths': test_paths, 'samples': latency, 'min': min(latency), 'max': max(latency),
                                                 'mean': statistics.mean(latency),
                                                 'median': statistics.median(latency),
                                                 'fps': 1 / statistics.mean(latency)},
              'timing_boundary': 'FP32 fused evaluation model, batch 1 with actual 416x416 decoded CPU uint8 input; H2D transfer, float/255 normalization, forward and native NMS (confidence .25, IoU .45, max 300); 5 warmups and one synchronized sample per timing image (shared validation subset in preflight, test split in full runs). Excludes file decoding, rendering and metric computation. Native half-precision held-out validation timing is separate.',
              'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
              'mean_warm_batch_seconds': statistics.mean(warm_times),
                  'epoch_validation_seconds': validation_seconds,
              'observed_training_shapes': sorted(observed_shapes),
              'observed_epoch_sample_counts': epoch_counts,
              'split_seed': 42, 'native_training_seed': 0,
              'weights_url': url, 'weights_sha256': weight_hash,
              'best_checkpoint_sha256': sha(checkpoint), 'resume_verified': bool(args.resume_check or args.resume)}
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
