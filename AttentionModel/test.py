import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    # g1 = tf.Graph()
    # for i in range(3):
    #     with g1.as_default():
    #         a = tf.Variable(initial_value=1, name="a_in_g1")
    #         b = tf.Variable(initial_value=2, name="b_in_g1")

    a = [[1, 2, 3, 4],
         [2, 8, 9, 0]]

    a1 = tf.concat([a for i in range(3)], axis=1)
    a2 = tf.reshape(a1, shape=[2, 3, 4])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(a2[1, 1, 3]))
