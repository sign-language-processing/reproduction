"""Architecture-only size audit; no data, training or accuracy selection."""
import json
import sys
from pathlib import Path
import torch
from torch import nn
import timm
sys.path.insert(0, "/opt/TCN/TCN")
from tcn import TemporalBlock

torch.set_num_threads(4)
results = []
for name, layers in [("skresnext50_32x4d", [1,1,1,1]), ("skresnext50_32x4d", [1,2,2,1]), ("skresnext50_32x4d", [2,2,2,2]), ("skresnet18", [2,2,2,2])]:
    spatial = timm.create_model(name, layers=layers, pretrained=False, num_classes=0, global_pool="max", act_layer=nn.Mish)
    for width in [128,224,232,256]:
        incoming = spatial.num_features
        blocks = []
        for dilation in [1,2,5]:
            blocks.append(TemporalBlock(incoming,width,3,1,dilation,2*dilation,dropout=0.2))
            incoming = width
        temporal = nn.Sequential(*blocks)
        classifier = nn.Linear(width,64)
        counts = {"spatial":sum(p.numel() for p in spatial.parameters()), "temporal":sum(p.numel() for p in temporal.parameters()), "classifier":sum(p.numel() for p in classifier.parameters())}
        results.append(dict(backbone=name,layers=layers,temporal_width=width,counts=counts,total=sum(counts.values())))
print(json.dumps(results,indent=2),flush=True)
Path(sys.argv[1]).write_text(json.dumps(results,indent=2))
