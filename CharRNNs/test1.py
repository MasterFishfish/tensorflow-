import numpy as np
import tensorflow as tf
import codecs
#from readfile import TextConverter, batch_generator
def get_a_cell():
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=16)
    dropcell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
    return dropcell

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    return np.random.choice(vocab_size, 1, p)[0]
if __name__ == "__main__":
    # arr = np.array([[1, 2, 3, 4, 5, 6],
    #                 [2, 3, 4, 5, 6, 7]])
    # a = [1, 2, 3, 4, 5, 6]
    # for n in range(0, arr.shape[1], 3):
    #     x = arr[:, n:n+3]
    #     print(x)
    num_steps = 26
    num_seqs = 32
    with codecs.open("./test.txt", encoding="utf-8") as f:
        text = f.read()
    # converter = TextConverter(text)
    # g = batch_generator(converter.text_to_arr(text), num_seqs, num_steps)
    # c = tf.get_variable("embedding", [converter.vocab_size(), 128],
    #                     initializer=tf.ones_initializer())
    # b = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='inputs')
    # x = 0
    # for i, j in g:
    #     with tf.Session() as sess:
    #         sess.run(tf.initialize_all_variables())
    #         d = tf.nn.embedding_lookup(c, b)
    #         if x == 0:
    #             print(d)
    #             print(sess.run(fetches=c, feed_dict={b: i}))
    #             print(converter.arr_to_text(i[0]))
    #     x += 1
    a = [[[1, 2, 3, 4, 5],
          [2, 3, 4, 5, 6],
          [3, 4, 5, 6, 8]],

        [[4, 7, 9, 0, 1],
         [2, 6, 8, 2, 5],
         [5, 2, 4, 7, 8]]]


    cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for i in range(3)])
    inputs = tf.placeholder(np.float32, shape=(2, 3, 5))
    h0 = cell.zero_state(2, tf.float32)
    print(h0)
    outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, initial_state=h0)
    print(outputs)
    x = tf.Variable(initial_value=0, name="init_x")
    y = x ** 2

    embeddings = tf.get_variable("embeddings", [3, 4])
    # embedding_slookup函数用于将input矩阵里面的每一数字对应的embedding找出来
    # 生成的lstm_inputs为(num_seqs, num_steps, embedding_size)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(outputs, feed_dict={inputs: a}))
        #print(sess.run(y))
