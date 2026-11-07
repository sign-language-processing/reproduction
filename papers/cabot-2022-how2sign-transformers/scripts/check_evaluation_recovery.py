"""Real legacy-GPU interrupted-batch and CUDA RNG recovery verification."""
import hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/slt')
import numpy as np
import tensorflow as tf
import torch,yaml
from signjoey.data import load_data
from signjoey.model import build_model
from signjoey.prediction import validate_on_data
from signjoey.loss import XentLoss
from signjoey.vocabulary import PAD_TOKEN,SIL_TOKEN
from parallel_ctc import ParallelCTC
from recovery_eval import with_evaluation_recovery

out=Path(sys.argv[1]);source=Path('/outputs/how2sign-preflight-001/model')
cfg=yaml.safe_load((source.parent/'config.yaml').read_text())
cfg['data'].update(train='val-preflight.pkl.gz',dev='val-preflight.pkl.gz',test='test-preflight.pkl.gz',
    random_train_subset=-1,random_dev_subset=-1,gls_vocab=str(source/'gls.vocab'),txt_vocab=str(source/'txt.vocab'))
_,dev,_,gv,tv=load_data(cfg['data']);dev.examples=[dev.examples[i] for i in [2,4,6,8]]
model=build_model(cfg=cfg['model'],gls_vocab=gv,txt_vocab=tv,sgn_dim=1024,do_recognition=True,do_translation=True)
checkpoint=torch.load(source/'best.ckpt',map_location='cpu');model.load_state_dict(checkpoint['model_state']);del checkpoint
model.cuda().eval();torch.set_num_threads(4)
original_decoder=tf.nn.ctc_beam_search_decoder;decoder=ParallelCTC(original_decoder,4);tf.nn.ctc_beam_search_decoder=decoder
kwargs=dict(model=model,data=dev,batch_size=2,use_cuda=True,sgn_dim=1024,do_recognition=True,
    recognition_loss_function=torch.nn.CTCLoss(blank=gv.stoi[SIL_TOKEN],zero_infinity=True),recognition_loss_weight=1,
    do_translation=True,translation_loss_function=XentLoss(pad_index=tv.stoi[PAD_TOKEN],smoothing=0),translation_loss_weight=1,
    translation_max_output_length=30,level=cfg['data']['level'],txt_pad_index=tv.stoi[PAD_TOKEN],batch_type='sentence',
    dataset_version='how2sign',frame_subsampling_ratio=None,recognition_beam_size=10,translation_beam_size=2,translation_beam_alpha=2)
def evaluate(cfg_file,ckpt=None):return validate_on_data(**kwargs)
def normalized(value):
    if isinstance(value,dict):return {str(k):normalized(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [normalized(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if torch.is_tensor(value):return value.detach().cpu().tolist()
    return value
config=out/'config.yaml';config.write_text(yaml.safe_dump(cfg))
def digest_state():
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
state=digest_state();native=normalized(evaluate(str(config),str(source/'best.ckpt')))
identity={'manifest_sha256':'fe0d41ea4877a54f2be7b6601e69fa50da1ecec9f93b179d505123f8276ec48b'}
recovered=with_evaluation_recovery(evaluate,out/'batch-cache',identity)
original_run=model.run_batch;calls=[0]
class SimulatedPreemption(Exception):pass
def interrupted(**arguments):
    calls[0]+=1
    if calls[0]==2:raise SimulatedPreemption()
    return original_run(**arguments)
model.run_batch=interrupted
try:
    recovered(str(config),str(source/'best.ckpt'))
    raise AssertionError('Interruption was not reached')
except SimulatedPreemption:pass
assert calls[0]==2
calls[0]=0
def counted(**arguments):calls[0]+=1;return original_run(**arguments)
model.run_batch=counted
resumed=normalized(recovered(str(config),str(source/'best.ckpt')))
assert resumed==native and calls[0]==1
cfg['training'].update(load_model='operational-recovery.ckpt',repro_resume=True,repro_recovery_dir='operational-recovery')
resume_config=out/'resume-config.yaml';resume_config.write_text(yaml.safe_dump(cfg));calls[0]=0
assert normalized(recovered(str(resume_config),str(source/'best.ckpt')))==native and calls[0]==0
model.run_batch=original_run
# CUDA generator state is checked across a genuinely fresh Python process.
rng=torch.cuda.get_rng_state_all();torch.save(rng,out/'cuda-rng.pt')
expected=hashlib.sha256(torch.rand(1024,device='cuda').cpu().numpy().tobytes()).hexdigest()
program="import torch,hashlib,sys; torch.cuda.set_rng_state_all(torch.load(sys.argv[1],map_location='cpu')); print(hashlib.sha256(torch.rand(1024,device='cuda').cpu().numpy().tobytes()).hexdigest())"
actual=subprocess.check_output([sys.executable,'-c',program,str(out/'cuda-rng.pt')],text=True).strip()
assert actual==expected
torch.cuda.set_rng_state_all(rng)
assert digest_state()==state and model.do_recognition is True
tf.nn.ctc_beam_search_decoder=original_decoder;decoder.close()
result=dict(diagnostic_only=True,native_and_resumed_complete_outputs_exact=True,completed_batch_reused=True,
    missing_batch_recomputed_once=True,operational_config_replay_all_batches_cached=True,
    cuda_rng_fresh_process_exact=True,model_state_unchanged=True,model_state_sha256=state,
    real_record_lengths=[len(x.sgn) for x in dev.examples],gloss_vocabulary=len(gv),text_vocabulary=len(tv),
    peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),torch_version=torch.__version__,torchtext_version=__import__('torchtext').__version__)
(out/'evaluation-recovery-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
