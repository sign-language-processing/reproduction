"""Exercise author loader, training/evaluation, and checkpoint I/O on real records.
This unrectified subset is diagnostic evidence, not either Table 6 experiment.
"""
import hashlib, json, os, random, sys, time
from pathlib import Path
sys.path.insert(0, '/opt/siformer')
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets.czech_slr_dataset import CzechSLRDataset
from siformer.model import SiFormer
from siformer.utils import train_epoch, evaluate

source, output = Path(sys.argv[1]), Path(sys.argv[2])
random.seed(379)
np.random.seed(379)
torch.manual_seed(379)
torch.cuda.manual_seed_all(379)
torch.backends.cudnn.deterministic = True
frame = pd.read_csv(source)
# No augmentation or score claim; preserve real author CSV input and exact loader.
subset = output / 'temporary-subset.csv'
frame.iloc[:72].to_csv(subset,index=False)
train_set = CzechSLRDataset(str(subset))
frame.iloc[72:80].to_csv(subset,index=False)
eval_set = CzechSLRDataset(str(subset))
subset.unlink()
loader = DataLoader(train_set,batch_size=24,shuffle=False,num_workers=0)
eval_loader = DataLoader(eval_set,batch_size=4,shuffle=False,num_workers=0)
device = torch.device('cuda')
model = SiFormer(num_classes=64,device=device).to(device)
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(0.9,0.999),weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[60,80],gamma=0.1)
start=time.monotonic()
model.train()
train = train_epoch(model,loader,torch.nn.CrossEntropyLoss(),optimizer,device,scheduler)
torch.cuda.synchronize()
training_seconds=time.monotonic()-start
checkpoint=output/'checkpoint.pth'
torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()},checkpoint)
state=torch.load(checkpoint,weights_only=True)
model.load_state_dict(state['model'])
optimizer.load_state_dict(state['optimizer'])
scheduler.load_state_dict(state['scheduler'])
model.eval()
correct,total,accuracy=evaluate(model,eval_loader,device)
result=dict(seed=379,training_samples=72,training_steps=3,evaluation_samples=total,correct=correct,accuracy_percent=accuracy*100,training_seconds=training_seconds,peak_memory_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),torch_version=torch.__version__,cuda_version=torch.version.cuda,device=torch.cuda.get_device_name(),comparable_to_table6=False)
(output/'metrics.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
