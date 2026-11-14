"""Ingest the hashed native artifacts collected by modal_app.py --mode report."""
import argparse,json,re,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODEL_TARGET={'cnn3':'cnn-3','mobilenet':'mobilenet','mobilenetv2':'mobilenetv2','xception':'xception','inceptionv3':'inceptionv3','inceptionresnetv2':'inceptionresnetv2','densenet169':'densenet169','bit-r50x1':'bit-m-r50x1','bit-r50x3':'bit-m-r50x3','vit-b16':'vit-b16','vit-l32':'vit-132','swinv2-tiny':'swinv2tiny256','vgg16':'vgg16-39','vgg19':'vgg19','resnet50':'resnet50','resnet152':'resnet152-40'}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('evidence',type=Path);args=parser.parse_args()
    lock=Path('/tmp/arsl-ledger.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    evidence=json.loads(args.evidence.read_text());path=ROOT/'reproduction.json';doc=json.loads(path.read_text())
    artifacts={a['artifact_id']:a for a in doc['artifacts']};models={};partials={}
    for run in doc['runs']:
        rid=run['run_id'];files=evidence.get(rid)
        if not files:continue
        run['artifact_ids']=[]
        for name,data in files.items():
            if name=='_checkpoint_inspection':
                run['checkpoint_inspection']=data;continue
            aid=rid+'-'+name.replace('.','-')
            artifacts[aid]={'artifact_id':aid,'kind':'native_run_evidence','uri':f'modal://arsl-4982792c-results/{rid}/{name}','sha256':data['sha256'],'size_bytes':data['size_bytes']}
            run['artifact_ids'].append(aid)
        if files.get('console.log',{}).get('content'):run['failure_evidence_excerpt']=files['console.log']['content']
        metadata=files.get('modal.json',{}).get('content',{})
        run['modal'].update({k:metadata[k] for k in ['image_id','function_call_id'] if k in metadata})
        run['compute'].update(platform='modal',modal_profile='repro-sign',shared_cache={'modal_volume':'huggingface-cache','mount_path':'/cache/huggingface','environment':{'HF_HOME':'/cache/huggingface','HF_HUB_CACHE':'/cache/huggingface/hub'}})
        if 'wall_seconds' in metadata:
            run['compute']['process_wall_seconds']=metadata['wall_seconds'];run['compute']['process_gpu_hours']=metadata['wall_seconds']/3600
            run['compute']['estimated_process_cost_usd']=metadata['wall_seconds']*(.000694+8*.0000131+32*.00000222)
            run['compute']['cost_basis']='Conservative A100-80GB public rate plus 8 CPU cores and 32 GiB RAM; process duration only, excludes startup/storage/network. Actual bill not available.'
        config=files.get('configuration.json',{}).get('content',{})
        if config:run['configuration']=config
        environment=files.get('environment.txt',{}).get('content','')
        gpu=re.search(r'NVIDIA A100[^|\n]*',environment)
        if gpu:run['compute']['gpu']=gpu.group(0).strip()
        metrics=files.get('metrics.json',{}).get('content',{})
        if metrics:
            if 'history' not in metrics:
                run['observed_result']=metrics
            else:
                run['conditional_result']={'completed':metrics.get('completed',False),'epochs':len(metrics['history']),'final':metrics['history'][-1] if metrics['history'] else None}
                if metrics.get('completed'):models[metrics['model']]=(rid,metrics)
                else:partials[metrics['model']]=(rid,metrics)
            run['compute']['peak_gpu_bytes']=metrics.get('peak_gpu_bytes')
    partials={k:v for k,v in partials.items() if k not in models}
    for model,(rid,metrics) in {**partials,**models}.items():
        slug=MODEL_TARGET[model]
        for phase,epoch in [('one-cycle',1),('final',100 if model=='cnn3' else 10)]:
            if phase=='final' and not metrics.get('completed'):continue
            matching=[x for x in metrics['history'] if x['epoch']==epoch]
            if not matching:continue
            observation=matching[0]['evaluation']
            for target in doc['targets']:
                if target['experiment_id']!=slug+'-'+phase:continue
                metric=target['metric_id'];original=target['original_value']
                if metric=='loss':
                    value={k:observation[k] for k in ['categorical_crossentropy','mse']};delta={k:v-original for k,v in value.items()}
                else:value=observation[metric];delta=value-original
                target['conditional_observation']={'run_id':rid,'artifact_id':rid+'-metrics-json','epoch':epoch,'value':value,'difference_from_published':delta,'comparable':target['result']['status']=='produced','full_training_completed':metrics.get('completed',False),'basis':target['result'].get('comparison_basis',target['result'].get('reason',{}).get('detail','Approved reconstruction; metric comparability is assessed per target.'))}
                if target['result']['status']=='produced':
                    target['result'].update(reproduced_value=value,difference=delta,artifact_ids=[rid+'-metrics-json'])
                target['result']['run_ids']=[rid]
                target['result']['artifact_ids']=list(set(target['result']['artifact_ids']+[rid+'-metrics-json']))
    doc['artifacts']=list(artifacts.values())
    doc['conditional_protocol']['completed_models']=list(models)
    doc['conditional_protocol']['interrupted_models']=[{'model':k,'run_id':v[0],'completed_epochs':len(v[1]['history']),'artifact_id':v[0]+'-metrics-json'} for k,v in partials.items()]
    doc['conditional_protocol']['status']='complete' if len(models)==16 else doc['conditional_protocol'].get('status','in_progress')
    path.write_text(json.dumps(doc,indent=2,ensure_ascii=False)+'\n')
    report=ROOT/'README.md'
    start='## Conditional full-run observations';end='Comparability is assessed per target; unresolved metric definitions remain documented limitations.'
    lines=['## Conditional full-run observations','',f'{len(models)} of 16 full model runs completed. These values use the approved assumptions; target eligibility and numerical agreement are recorded in `reproduction.json`. Available conditional values and mechanical differences are linked to their matching cells in the 128-target ledger in `reproduction.json`. CE is categorical cross-entropy; MSE is mean probability-space squared error.','',
        '| Model | Epoch-1 accuracy | Epoch-1 CE | Epoch-1 MSE | Final accuracy | Published final accuracy | Difference (percentage points) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for model,(rid,metrics) in models.items():
        first=metrics['history'][0]['evaluation'];last=metrics['history'][-1]['evaluation']
        target=next(t for t in doc['targets'] if t['experiment_id']==MODEL_TARGET[model]+'-final' and t['metric_id']=='accuracy')
        original=target['original_value']
        lines.append(f"| {model} | {first['accuracy']:.6f} | {first['categorical_crossentropy']:.6f} | {first['mse']:.8f} | {last['accuracy']:.6f} | {original:.4f} | {(last['accuracy']-original)*100:+.4f} |")
    lines+=['','| Model | Final macro precision | Final macro recall | Final macro F1 | Final macro OVR AUC | Final CE | Final MSE |','|---|---:|---:|---:|---:|---:|---:|']
    for model,(rid,metrics) in models.items():
        m=metrics['history'][-1]['evaluation']
        lines.append(f"| {model} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['auc']:.6f} | {m['categorical_crossentropy']:.6f} | {m['mse']:.8f} |")
    if partials:
        lines+=['','Interrupted training snapshots:','','| Model | Completed epochs | Planned epochs | Latest validation accuracy |','|---|---:|---:|---:|']
        for model,(rid,metrics) in partials.items():
            last=metrics['history'][-1]
            lines.append(f"| {model} | {last['epoch']} | {100 if model=='cnn3' else 10} | {last['evaluation']['accuracy']:.6f} |")
    body=start+'\n\n'+'\n'.join(lines[2:])+'\n\n'+end
    text=report.read_text().replace('<!-- conditional-results:start -->\n','').replace('\n<!-- conditional-results:end -->','\n'+end)
    if start in text:text=text[:text.index(start)]+body+text[text.index(end)+len(end):]
    else:text+='\n'+body+'\n'
    report.write_text(text)
    print('Collected',len(evidence),'run directories;',len(models),'completed conditional model runs.')
if __name__=='__main__':main()
