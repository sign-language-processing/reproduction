"""Bounded CPU scheduling of the unchanged TF2.1 per-sentence beam decoder.

The upstream kernel serializes independent batch entries. This adapter preserves
input values, class/blank order, decoder arguments, and sparse-output batch order.
"""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import tensorflow as tf


class ParallelCTC:
    def __init__(self, decoder, workers=4):
        self.decoder = decoder
        self.pool = ThreadPoolExecutor(max_workers=workers)
        with tf.device('/CPU:0'):
            tf.constant(0)  # Initialize eager context before worker calls.

    def __call__(self, inputs, sequence_length, **kwargs):
        values = np.asarray(inputs)
        lengths = np.asarray(sequence_length)
        if len(lengths) <= 1:
            return self.decoder(inputs=inputs, sequence_length=sequence_length, **kwargs)

        def decode(index):
            with tf.device('/CPU:0'):
                return self.decoder(inputs=values[:,index:index+1,:],
                                    sequence_length=lengths[index:index+1], **kwargs)

        results = list(self.pool.map(decode, range(len(lengths))))
        paths = []
        for path in range(len(results[0][0])):
            indices, decoded, width = [], [], 0
            for batch_index, (sample_paths, _) in enumerate(results):
                sample = sample_paths[path]
                index = sample.indices.numpy().copy()
                index[:,0] = batch_index
                indices.append(index)
                decoded.append(sample.values.numpy())
                width = max(width, int(sample.dense_shape.numpy()[1]))
            paths.append(tf.SparseTensor(np.concatenate(indices,axis=0),
                                         np.concatenate(decoded,axis=0),
                                         np.array([len(lengths),width],dtype=np.int64)))
        probabilities = tf.convert_to_tensor(np.concatenate([r[1].numpy() for r in results],axis=0))
        return paths, probabilities

    def close(self):
        self.pool.shutdown(wait=True)
