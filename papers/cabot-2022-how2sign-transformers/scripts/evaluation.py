"""Reuse identical recognition calculations within one native checkpoint test.

The author still enumerates every translation candidate and selects with its
original order/ties. Training validation and final test data remain independent.
"""
from functools import wraps


def reuse_recognition(validate):
    cache = {}
    stats = {'native_joint_calls': 0, 'recognition_cache_hits': 0}

    @wraps(validate)
    def evaluate(**kwargs):
        if not kwargs.get('do_recognition'):
            return validate(**kwargs)
        model = kwargs['model']
        key = tuple(kwargs.get(k) for k in (
            'model', 'data', 'recognition_beam_size', 'batch_size', 'batch_type',
            'use_cuda', 'sgn_dim', 'dataset_version', 'frame_subsampling_ratio',
            'recognition_loss_function', 'recognition_loss_weight', 'level'))
        if key not in cache:
            result = validate(**kwargs)
            cache[key] = {k: result[k] for k in (
                'valid_recognition_loss', 'decoded_gls', 'gls_ref', 'gls_hyp')}
            cache[key]['scores'] = {k: result['valid_scores'][k] for k in ('wer', 'wer_scores')}
            stats['native_joint_calls'] += 1
            return result
        arguments = dict(kwargs, do_recognition=False,
                         recognition_loss_function=None,
                         recognition_loss_weight=None, recognition_beam_size=None)
        previous = model.do_recognition
        try:
            model.do_recognition = False
            result = validate(**arguments)
        finally:
            model.do_recognition = previous
        recognition = cache[key]
        result.update({k: v for k, v in recognition.items() if k != 'scores'})
        result['valid_scores'].update(recognition['scores'])
        stats['recognition_cache_hits'] += 1
        return result

    evaluate.stats = stats
    return evaluate


def with_recognition_cache(test):
    @wraps(test)
    def run(*args, **kwargs):
        import json
        import signjoey.prediction as prediction
        original = prediction.validate_on_data
        cached = reuse_recognition(original)
        prediction.validate_on_data = cached
        try:
            return test(*args, **kwargs)
        finally:
            prediction.validate_on_data = original
            print(json.dumps({'checkpoint_test_recognition_reuse': cached.stats}), flush=True)
    return run
