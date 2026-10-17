"""Summarize retained native author results without recomputing or tuning metrics."""
from pathlib import Path
import hashlib,io,json,pickle,sys
import torch
from concurrent.futures import ThreadPoolExecutor
class CPUUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='torch.storage' and name=='_load_from_bytes':
            return lambda b:torch.load(io.BytesIO(b),map_location='cpu')
        return super().find_class(module,name)
out=Path(sys.argv[1]);metrics={};legacy_diagnostics={}
def preserve_how2_decoding(key,result):
    if not out.name.startswith('how2sign-'):return
    # Evaluation-only secondary diagnostic on identical retained decoded text.
    # This collector uses the original unpatched source; no checkpoint/grid choice changes.
    sys.path.insert(0,'/slt')
    from signjoey.metrics import wer_list
    text={'references':result['gls_ref'],'hypotheses':result['gls_hyp']}
    text_path=out/(key+'-recognition-text.json')
    text_path.write_text(json.dumps(text,ensure_ascii=False))
    legacy_diagnostics[key]={'legacy_uint8_wer':float(wer_list(text['references'],text['hypotheses'])['wer']),
        'corrected_primary_wer':float(result['valid_scores']['wer']),
        'purpose':'Secondary overflow diagnostic on identical decoded text; no model rerun or selection',
        'text_artifact':text_path.name}

for path in sorted(out.rglob('*.dev_results.pkl')):
    with path.open('rb') as f:d=CPUUnpickler(f).load()
    r=min(d['recognition_results'].items(),key=lambda item:item[1]['valid_scores']['wer'])
    preserve_how2_decoding('dev',r[1])
    candidates=[(beam,alpha,result) for beam,alphas in d['translation_results'].items() for alpha,result in alphas.items()]
    beam,alpha,best=max(candidates,key=lambda item:item[2]['valid_scores']['bleu'])
    metrics[str(path.relative_to(out))]={'wer':float(r[1]['valid_scores']['wer']),**{k:float(v) for k,v in best['valid_scores']['bleu_scores'].items()},'recognition_beam':r[0],'translation_beam':beam,'translation_alpha':alpha}
for path in sorted(out.rglob('*.test_results.pkl')):
    with path.open('rb') as f:d=CPUUnpickler(f).load()
    preserve_how2_decoding('test',d)
    metrics[str(path.relative_to(out))]={'wer':float(d['valid_scores']['wer']),**{k:float(v) for k,v in d['valid_scores']['bleu_scores'].items()}}
if legacy_diagnostics:(out/'legacy-wer-diagnostic.json').write_text(json.dumps(legacy_diagnostics,indent=2))
if metrics:(out/'raw-metrics.json').write_text(json.dumps(metrics,indent=2))
def hash_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return dict(path=str(p.relative_to(out)),sha256=h.hexdigest(),size_bytes=p.stat().st_size)
# Remote per-file reads dominate the large recovery cache. Preserve sorted
# manifest order and exact bytes while bounding independent I/O concurrency.
paths=[p for p in sorted(out.rglob('*')) if p.is_file() and p.name!='evidence.json']
print('Hashing {} closed evidence files'.format(len(paths)),flush=True)
with ThreadPoolExecutor(max_workers=8) as pool:
    files=list(pool.map(hash_file,paths))
result=dict(files=files,metrics=metrics)
(out/'evidence.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(file_count=len(files),metrics=metrics)))
