"""Dataset/config glue around pinned Detectron2; no local model or optimizer."""
import argparse
import json
import platform
from pathlib import Path

PIN = 'd1e04565d3bec8719335b88be9e9b961bf3ec464'
CONFIG = 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import detectron2
    from detectron2.config import get_cfg
    from detectron2.model_zoo import get_config_file, get_checkpoint_url
    from detectron2.modeling import build_model
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file(CONFIG))
    cfg.MODEL.DEVICE = 'cpu' if args.probe else 'cuda'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 28
    cfg.INPUT.MIN_SIZE_TRAIN = (416,)
    cfg.INPUT.MAX_SIZE_TRAIN = 416
    cfg.INPUT.MIN_SIZE_TEST = 416
    cfg.INPUT.MAX_SIZE_TEST = 416
    cfg.SOLVER.IMS_PER_BATCH = 24
    cfg.SOLVER.BASE_LR = .001
    cfg.SEED = 42
    cfg.OUTPUT_DIR = str(args.output)
    cfg.freeze()
    model = build_model(cfg)
    counts = {'all_parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    result = {'source_revision': PIN, 'config': CONFIG, 'class_count': 28,
              'python': platform.python_version(), 'torch': torch.__version__,
              'detectron2': detectron2.__version__, 'parameter_counts': counts,
              'proposed_coco_weights_url': get_checkpoint_url(CONFIG),
              'scope': 'CPU model construction only; no training, weights download or target metric claim.'}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'source-probe.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.output / 'config.yaml').write_text(cfg.dump())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
