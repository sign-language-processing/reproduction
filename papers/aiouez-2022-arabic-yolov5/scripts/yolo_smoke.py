"""CPU-only execution probe of pinned native imports, real loaders and loss."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

import torch
import numpy as np
import yaml

sys.path.insert(0, '/opt/yolov5')
import train  # exercise native import tree
from models.yolo import Model
from utils.datasets import create_dataloader
from utils.torch_utils import intersect_dicts
from utils.general import labels_to_class_weights, labels_to_image_weights
from utils.loss import ComputeLoss

torch.set_num_threads(4)
data = yaml.safe_load(Path('/tmp/yolo-data/data.yaml').read_text())
hyp = yaml.safe_load(Path('/opt/yolov5/data/hyps/hyp.scratch.yaml').read_text())
loader, dataset = create_dataloader(data['train'], 416, 16, 32, hyp=hyp, augment=True, workers=0)
images, targets, paths, shapes = next(iter(loader))
assert len(labels_to_class_weights(dataset.labels, 28)) == 28
assert len(labels_to_image_weights(dataset.labels, 28, np.ones(28))) == len(dataset.labels)
validation, _ = create_dataloader(data['val'], 416, 16, 32, hyp=hyp, rect=True, pad=.5, workers=0)
validation_images = next(iter(validation))[0]
weight = Path('/cache/huggingface/yolov5-v6.0/yolov5s.pt')
weight.parent.mkdir(parents=True, exist_ok=True)
url = 'https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt'
if not weight.exists():
    temporary = weight.with_suffix('.part')
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(weight)
checkpoint = torch.load(weight, map_location='cpu', weights_only=False)
model = Model(checkpoint['model'].yaml, ch=3, nc=28)
state = intersect_dicts(checkpoint['model'].float().state_dict(), model.state_dict())
model.load_state_dict(state, strict=False)
model.hyp = hyp
model.nc = 28
model.gr = 1.0
model.names = data['names']
model.train()
prediction = model(images[:1].float() / 255)
loss, items = ComputeLoss(model)(prediction, targets[targets[:, 0] == 0])
assert torch.isfinite(loss).all()
loss.backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
result = {'train_batch_shape': list(images.shape), 'validation_batch_shape': list(validation_images.shape),
          'loss': float(loss.detach()), 'loss_items': items.detach().tolist(),
          'weights_url': url, 'weights_sha256': hashlib.sha256(weight.read_bytes()).hexdigest(),
          'real_forward_loss_backward': True, 'native_source': train.__file__}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
