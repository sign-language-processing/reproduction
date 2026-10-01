"""Real-checkpoint equivalence and capacity diagnostics; no target scores."""
from pathlib import Path
import hashlib, json, sys, time
sys.path.insert(0, '/slt')
import numpy as np
import tensorflow as tf
from parallel_ctc import ParallelCTC
import torch
import yaml
from signjoey.data import load_data
from torchtext.data import Dataset
from signjoey.model import build_model
from signjoey.prediction import validate_on_data
from signjoey.loss import XentLoss
from signjoey.vocabulary import PAD_TOKEN, SIL_TOKEN
from evaluation import reuse_recognition

out = Path(sys.argv[1])
remaining_mode = len(sys.argv)>2 and sys.argv[2]=='remaining'
capacity_mode = len(sys.argv)>2 and sys.argv[2].startswith('capacity')
normal_capacity = capacity_mode and sys.argv[2]=='capacity-normal'
gpu_mode = capacity_mode or (len(sys.argv)>2 and sys.argv[2]=='gpu')
source = Path('/outputs/how2sign-preflight-001/model')
cfg = yaml.safe_load((source.parent/'config.yaml').read_text())
cfg['data'].update(train='val-preflight.pkl.gz', random_train_subset=-1,
                   gls_vocab=str(source/'gls.vocab'), txt_vocab=str(source/'txt.vocab'),
                   dev='val-preflight.pkl.gz' if capacity_mode and not normal_capacity else 'val.pkl.gz', test='test-preflight.pkl.gz' if capacity_mode else 'test.pkl.gz')
_, dev, test, gv, tv = load_data(cfg['data'])
normal = Dataset(list(dev.examples), dev.fields)
indices = np.random.RandomState(42).choice(len(dev.examples),16,replace=False)
normal.examples = [dev.examples[int(i)] for i in indices]
if normal_capacity:
    indices = np.random.RandomState(42).choice(len(dev.examples),96,replace=False)
    dev.examples = [dev.examples[int(i)] for i in indices]
for data in (dev, test):
    ordered = sorted(data.examples, key=lambda x: len(x.sgn))
    if not capacity_mode:data.examples = [ordered[len(ordered)//2], ordered[3*len(ordered)//4] if gpu_mode else ordered[-1]]
model = build_model(cfg=cfg['model'], gls_vocab=gv, txt_vocab=tv, sgn_dim=1024, do_recognition=True, do_translation=True)
checkpoint = torch.load(source/'best.ckpt', map_location='cpu')
model.load_state_dict(checkpoint['model_state']); del checkpoint
model.eval(); torch.set_num_threads(4 if gpu_mode else 8)
if gpu_mode:model.cuda()

def digest_state():
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def normalized(value):
    if isinstance(value, dict): return {str(k): normalized(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [normalized(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if torch.is_tensor(value): return value.detach().cpu().tolist()
    return value

kwargs=dict(model=model, batch_size=2, use_cuda=gpu_mode, sgn_dim=1024,
    do_recognition=True, recognition_loss_function=torch.nn.CTCLoss(blank=gv.stoi[SIL_TOKEN],zero_infinity=True),
    recognition_loss_weight=1, do_translation=True,
    translation_loss_function=XentLoss(pad_index=tv.stoi[PAD_TOKEN],smoothing=0),
    translation_loss_weight=1, translation_max_output_length=30,
    level=cfg['data']['level'],txt_pad_index=tv.stoi[PAD_TOKEN],
    batch_type='sentence',dataset_version='how2sign',frame_subsampling_ratio=None)
if capacity_mode:
    state=digest_state();rows=[];previous=model.do_recognition
    try:
        model.do_recognition=False
        for beam in [1,10]:
            args=dict(kwargs,data=dev,batch_size=16,do_recognition=False,
                      recognition_loss_function=None,recognition_loss_weight=None,
                      recognition_beam_size=None,translation_beam_size=beam,translation_beam_alpha=5)
            torch.cuda.synchronize();begin=time.monotonic();result=validate_on_data(**args);torch.cuda.synchronize()
            row={'beam':beam,'alpha':5,'seconds':time.monotonic()-begin,
                 'input_lengths':[len(x.sgn) for x in dev.examples],
                 'hypothesis_lengths':[len(x.split()) for x in result['txt_hyp']],
                 'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),
                 'result_sha256':hashlib.sha256(json.dumps(normalized(result),sort_keys=True).encode()).hexdigest()}
            rows.append(row);print(json.dumps(row),flush=True)
    finally:model.do_recognition=previous
    assert state==digest_state()
    (out/'capacity.json').write_text(json.dumps({'diagnostic_only':True,'model_state_unchanged':True,'rows':rows},indent=2))
    sys.exit(0)

if not gpu_mode and not remaining_mode:
    # Capture exact real-model logits for one ordinary16-record validation batch.
    original_decoder = tf.nn.ctc_beam_search_decoder
    captured = []
    def capture(inputs, sequence_length, **options):
        captured.append((np.array(inputs,copy=True),np.array(sequence_length,copy=True)))
        return original_decoder(inputs=inputs,sequence_length=sequence_length,**options)
    tf.nn.ctc_beam_search_decoder = capture
    try:
        validate_on_data(**dict(kwargs,data=normal,batch_size=16,recognition_beam_size=1,translation_beam_size=1,translation_beam_alpha=-1))
    finally:
        tf.nn.ctc_beam_search_decoder = original_decoder
    assert len(captured)==1
    values,lengths = captured[0]

    def decoder_equal(left,right):
        assert np.array_equal(left[1].numpy(),right[1].numpy())
        for a,b in zip(left[0],right[0]):
            for attr in ('indices','values','dense_shape'):
                assert np.array_equal(getattr(a,attr).numpy(),getattr(b,attr).numpy())

    benchmarks=[]
    for width in [1,10]:
        begin=time.monotonic();expected=original_decoder(inputs=values,sequence_length=lengths,beam_width=width,top_paths=1);native_seconds=time.monotonic()-begin
        for workers in [1,2,4]:
            decoder=ParallelCTC(original_decoder,workers)
            begin=time.monotonic();actual=decoder(inputs=values,sequence_length=lengths,beam_width=width,top_paths=1);seconds=time.monotonic()-begin
            decoder_equal(expected,actual);decoder.close()
            row=dict(beam=width,workers=workers,native_seconds=native_seconds,parallel_seconds=seconds,raw_sparse_and_log_probability_exact=True)
            benchmarks.append(row);print(json.dumps({'ctc_benchmark':row}),flush=True)
    parallel=ParallelCTC(original_decoder,4)
    # Partial final batch plus all-blank, repeated-label, ties and length-one cases.
    for width in [1,10]:
        for x,l in [(values[:,:3,:],lengths[:3]),(np.zeros((8,4,5),dtype=np.float32),np.array([1,3,5,8],dtype=np.int32)),(np.tile(np.array([0,0,0,0,20],dtype=np.float32),(8,4,1)),np.array([1,3,5,8],dtype=np.int32)),(np.tile(np.array([20,0,0,0,0],dtype=np.float32),(8,4,1)),np.array([1,3,5,8],dtype=np.int32))]:
            decoder_equal(original_decoder(inputs=x,sequence_length=l,beam_width=width,top_paths=1),parallel(inputs=x,sequence_length=l,beam_width=width,top_paths=1))
    (out/'decoder-equivalence.json').write_text(json.dumps({'benchmarks':benchmarks,'normal_record_ids':[x.sequence for x in normal.examples],'normal_lengths':lengths.tolist(),'input_sha256':hashlib.sha256(values.tobytes()).hexdigest(),'exact_edge_cases':True},indent=2))
    del captured,values,lengths,expected,actual
    # Each decoder output above is exactly native; now verify recognition reuse over
    # the entire native grid without paying repeated serial batch scheduling.
    tf.nn.ctc_beam_search_decoder = parallel
else:
    original_decoder=tf.nn.ctc_beam_search_decoder
    parallel=ParallelCTC(original_decoder,4)
if remaining_mode:tf.nn.ctc_beam_search_decoder=parallel
cached = reuse_recognition(validate_on_data)
state = digest_state(); records=[]; start=time.monotonic()
# Native order: requested recognition width, then all70 translation candidates.
candidates=[('dev-recognition',dev,10,1,-1)]
candidates += [('dev-translation',dev,1,b,a) for b,a in ([(1,-1),(10,5)] if gpu_mode else [(b,a) for b in range(1,11) for a in [-1,0,1,2,3,4,5]])]
best = None
continuation = None
if remaining_mode:
    prefix_path=Path('/outputs/how2sign-evaluation-equivalence-004/stdout.log')
    prefix=[]
    for line in prefix_path.read_text().splitlines():
        try: row=json.loads(line)
        except ValueError: continue
        if row.get('phase') in ['dev-recognition','dev-translation']:prefix.append(row)
    previous=[r for r in prefix if r['phase']=='dev-translation']
    expected=[(b,a) for b in range(1,11) for a in [-1,0,1,2,3,4,5]]
    assert [(r['translation_beam'],r['alpha']) for r in previous]==expected[:35]
    assert all(r['exact_equal'] for r in prefix)
    for r in previous:
        if best is None or r['bleu4']>best[0]:best=(r['bleu4'],r['translation_beam'],r['alpha'])
    records=prefix
    candidates=[c for c in candidates if c[0]=='dev-translation'][35:]
    continuation=dict(run_id='how2sign-evaluation-equivalence-004',
                      prefix_sha256=hashlib.sha256(prefix_path.read_bytes()).hexdigest(),
                      previously_verified_translation_candidates=len(previous),
                      newly_verified_translation_candidates=len(candidates))
for phase,data,rb,tb,alpha in candidates:
    args=dict(kwargs,data=data,recognition_beam_size=rb,translation_beam_size=tb,translation_beam_alpha=alpha)
    if gpu_mode:tf.nn.ctc_beam_search_decoder=original_decoder
    t=time.monotonic();native=validate_on_data(**args);native_seconds=time.monotonic()-t
    if gpu_mode:tf.nn.ctc_beam_search_decoder=parallel
    t=time.monotonic();reused=cached(**args);reused_seconds=time.monotonic()-t
    a,b=normalized(native),normalized(reused)
    assert a==b,('Non-equivalent evaluator',phase,rb,tb,alpha)
    assert model.do_recognition is True
    row=dict(phase=phase,recognition_beam=rb,translation_beam=tb,alpha=alpha,
             native_seconds=native_seconds,reused_seconds=reused_seconds,
             full_result_sha256=hashlib.sha256(json.dumps(a,sort_keys=True).encode()).hexdigest(),
             exact_equal=True,bleu4=a['valid_scores']['bleu'])
    records.append(row);print(json.dumps(row),flush=True)
    if phase=='dev-translation' and (best is None or a['valid_scores']['bleu']>best[0]):best=(a['valid_scores']['bleu'],tb,alpha)
# Fixed test uses the same validation-selected settings in both paths.
args=dict(kwargs,data=test,recognition_beam_size=10,translation_beam_size=best[1],translation_beam_alpha=best[2])
if gpu_mode:tf.nn.ctc_beam_search_decoder=original_decoder
a=normalized(validate_on_data(**args))
if gpu_mode:tf.nn.ctc_beam_search_decoder=parallel
b=normalized(cached(**args));assert a==b
assert state==digest_state() and model.do_recognition is True
result=dict(source_checkpoint=str(source/'best.ckpt'),source_state_sha256=state,
            full_model_vocabulary={'gloss':len(gv),'text':len(tv)},
            dev_lengths=[len(x.sgn) for x in dev.examples],test_lengths=[len(x.sgn) for x in test.examples],
            candidates=records,selected_translation={'beam':best[1],'alpha':best[2]},
            final_test_exact_equal=True,model_state_unchanged=True,cache_stats=cached.stats,
            elapsed_seconds=time.monotonic()-start,diagnostic_only=True,continuation=continuation,gpu_smoke=gpu_mode,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if gpu_mode else None)
tf.nn.ctc_beam_search_decoder = original_decoder; parallel.close()
(out/'equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
