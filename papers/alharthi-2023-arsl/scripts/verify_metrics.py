"""Independently verify epoch-one/final metrics from checksum-verified predictions.
Usage: python verify_metrics.py /tmp/arsl-evidence.json /tmp/arsl-predictions
Requires NumPy; files are RUN_ID-predictions-epochN.npz in the second directory.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np

def auc_binary(truth,score):
    order=np.argsort(score,kind='stable');score=score[order];truth=truth[order]
    _,start,count=np.unique(score,return_index=True,return_counts=True)
    positive=np.add.reduceat(truth.astype(np.int64),start);negative=count-positive
    preceding=np.cumsum(negative)-negative
    return float(np.sum(positive*(preceding+.5*negative))/(truth.sum()*(len(truth)-truth.sum())))

def verify(evidence,directory):
    findings={};reference=None
    for rid,files in evidence.items():
        metrics=files.get('metrics.json',{}).get('content',{})
        if not rid.startswith('full-') or not metrics.get('completed'):continue
        expected_epochs=100 if metrics['model']=='cnn3' else 10
        assert len(metrics['history'])==expected_epochs
        assert metrics['optimizer_iterations']==1521*expected_epochs
        hashes=files['artifact-hashes.json']['content'];findings[rid]={}
        for row in [metrics['history'][0],metrics['history'][-1]]:
            epoch=row['epoch'];name=f'predictions-epoch{epoch}.npz';path=directory/(rid+'-'+name)
            assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[name]
            with np.load(path) as arrays:
                p=arrays['probabilities'];y=arrays['labels'];indices=arrays['indices']
            assert p.shape==(5404,32) and np.isfinite(p).all() and (p>=0).all()
            np.testing.assert_allclose(p.sum(1),1,rtol=0,atol=1e-12)
            assert len(np.unique(indices))==5404 and len(np.unique(y))==32
            if reference is None:reference=(indices.copy(),y.copy())
            else:
                np.testing.assert_array_equal(indices,reference[0]);np.testing.assert_array_equal(y,reference[1])
            predicted=p.argmax(1);confusion=np.bincount(y*32+predicted,minlength=1024).reshape(32,32)
            true_positive=confusion.diagonal();precision=np.divide(true_positive,confusion.sum(0),out=np.zeros(32),where=confusion.sum(0)!=0)
            recall=true_positive/confusion.sum(1);f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(32),where=precision+recall!=0)
            hot=np.eye(32)[y];clipped=np.clip(p,np.finfo(p.dtype).eps,1-np.finfo(p.dtype).eps);clipped/=clipped.sum(1,keepdims=True)
            calculated={'accuracy':float((predicted==y).mean()),'precision':float(precision.mean()),'recall':float(recall.mean()),'f1':float(f1.mean()),'auc':float(np.mean([auc_binary(y==c,p[:,c]) for c in range(32)])),'categorical_crossentropy':float(-np.log(clipped[np.arange(len(y)),y]).mean()),'mse':float(((p-hot)**2).mean())}
            differences={k:abs(value-row['evaluation'][k]) for k,value in calculated.items()}
            assert max(differences.values())<1e-12,(rid,epoch,differences)
            assert row['evaluation']['sample_count']==5404
            findings[rid][str(epoch)]={'prediction_sha256':hashes[name],'max_metric_absolute_difference':max(differences.values()),'verified_metrics':list(calculated),'sample_count':5404,'predicted_class_counts':confusion.sum(0).tolist(),'probability_variation_max':float(np.ptp(p,axis=0).max())}
    return findings
if __name__=='__main__':
    print(json.dumps(verify(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2])),indent=2))
