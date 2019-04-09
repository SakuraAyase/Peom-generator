# coding: utf-8


import numpy as np
import tensorflow as tf


# 2 * 3 * 4
# 可以看成 batch_size = 2, max_time = 3, len(char2id) + 1 = 4
logits_np = np.array([
    [[0.6, 0.5, 0.3, 0.2], [0.9, 0.5, 0.3, 0.2], [1.0, 0.5, 0.3, 0.2]],
    [[0.2, 0.5, 0.3, 0.2], [0.3, 0.5, 0.3, 0.2], [0.4, 0.5, 0.3, 0.2]]
])

targets_np = np.array([
    [0, 1, 2],
    [3, 0, 1]
], dtype=np.int32)

logits = tf.convert_to_tensor(logits_np)
targets = tf.convert_to_tensor(targets_np)

cost = tf.contrib.seq2seq.sequence_loss(logits=logits,
                     targets=targets,
                     weights=tf.ones_like(targets, dtype=tf.float64))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    r = sess.run(cost)
    print(r)

print((np.log(np.exp(0.6) / (np.exp(0.6) + np.exp(0.5) + np.exp(0.3) + np.exp(0.2))) +
       np.log(np.exp(0.5) / (np.exp(0.9) + np.exp(0.5) + np.exp(0.3) + np.exp(0.2))) +
       np.log(np.exp(0.3) / (np.exp(1.0) + np.exp(0.5) + np.exp(0.3) + np.exp(0.2))) +
       np.log(np.exp(0.2) / (np.exp(0.2) + np.exp(0.5) + np.exp(0.3) + np.exp(0.2))) +
       np.log(np.exp(0.3) / (np.exp(0.3) + np.exp(0.5) + np.exp(0.3) + np.exp(0.2))) +
       np.log(np.exp(0.5) / (np.exp(0.4) + np.exp(0.5) + np.exp(0.3) + np.exp(0.2)))) / 6)