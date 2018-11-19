import tensorflow as tf
import numpy as np
import time
import os

def pick_top_n(preds, vocab_size, top_n=1):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    p = list(p)
    c = np.random.choice(vocab_size, 1, replace=False, p=p)
    return c[0]

class CharRNN():
    # 设每一个字母转换为的向量维度为embedding_size, 即inputs_size
    # 输入的每一句字母长度为num_steps
    # 每一次输入包含num_seqs句
    # lstm内部的weights是BasicLstmCell函数中自己自带生成的, 不需要显示定义
    # 每一层神经网络的隐藏层包含的神经元数量为 lstm_size, 即state_size
    # 包含神经网络的隐藏层的数量为 num_layers
    # 神经网络的学习率为 learn_rate
    # 神经网络的clip为该神经网络的梯度下降的最大的范数
    # 神经网络的dropout率 使用train_keep_prob表示
    # 输入的是英文字母则不需要embedding转化, 输入的如果是中文那么就需要embedding,
    # #此处用 use_embedding boolen向量来表示
    # num_class 表示的是 输入神经网络的字母或者字有多少种类
    # 鸟事该模型的状态 是否处于sample状态, sampling

    def __init__(self, num_class, num_steps=26, num_seqs=32, lstm_size=128,
                        num_layers=3, learn_rate=0.001, grad_clip=5,
                        train_keep_prob=0.5, sampling=False, use_embedding=False, embedding_size=64):
        if sampling == True:
            this_num_steps = 1
            this_num_seqs = 1
        else:
            this_num_seqs = num_seqs
            this_num_steps = num_steps

        self.vocab_num = num_class
        self.num_steps = this_num_steps
        self.num_seqs = this_num_seqs
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.init_keep_prob = train_keep_prob
        self.embedding_size = embedding_size
        self.learn_rate = learn_rate
        self.grad_clip = grad_clip
        self.usr_embedding = use_embedding

        tf.reset_default_graph()
        self.built_input()
        self.built_lstm()
        self.built_loss()
        self.built_optimizer()
        self.saver = tf.train.Saver()

    def built_input(self):
        with tf.name_scope("inputs"):
            self.inputs = tf.placeholder(tf.int32,
                                         shape=(self.num_seqs, self.num_steps), name="inputs")
            self.targets = tf.placeholder(tf.int32,
                                          shape=(self.num_seqs, self.num_steps), name="targets")
            self.keep_probs = tf.placeholder(tf.float32, name="keep_probs")

            # 如果是中文的话就引入embedding层
            # 英文字母没必要使用embedding层
            if self.usr_embedding == False:
                # 使用的self.inputs 的维度为(num_seqs, num_steps)
                # 转化成的self.lstm_inputs 的维度为(num_seqs, num_steps, vocab_num)
                self.lstm_inputs = tf.one_hot(self.inputs, self.vocab_num)
            else:
                # 生成embeddding矩阵(vocab_num, embedding_size)
                # 表示每一个单词都有对应的一个embedding向量
                embeddings = tf.get_variable("embeddings", [self.vocab_num, self.embedding_size])
                # embedding_slookup函数用于将input矩阵里面的每一数字对应的embedding找出来
                # 生成的lstm_inputs为(num_seqs, num_steps, embedding_size)
                self.lstm_inputs = tf.nn.embedding_lookup(embeddings, self.inputs)

    def built_lstm(self):
        # 从这个函数里面获得一层lstm,
        # 其中存在神经元的dropout, dropout率为self.keep_probs
        def get_a_nnCell():
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
            dropcell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_probs)
            return dropcell

        # lstm网络结构
        lstmcells = tf.nn.rnn_cell.MultiRNNCell([get_a_nnCell() for _ in range(self.num_layers)])
        # 传入lstm的初始化状态
        self.initial_state = lstmcells.zero_state(self.num_seqs, tf.float32)
        # 前向传播，展开时间维度
        # LSTM_outputs 为展开时间维度后的结果，
        # 其维度为(num_seqs, num_steps, lstm_size)
        # Lstm_final_state为最后一个时间段结束后的状态
        # 其维度为(num_seqs, lstm_size)
        self.lstm_outputs, self.lstm_final_state = tf.nn.dynamic_rnn(cell=lstmcells, inputs=self.lstm_inputs, initial_state=self.initial_state)

        # 对lstm_outputs进行一定处理，
        # 使其变成二维的,
        # 即维度为(num_seqs * num _steps, lstm_size)
        x = tf.reshape(tf.concat(self.lstm_outputs, 1), [-1, self.lstm_size])
        # 定义输出层的参数
        with tf.variable_scope("softmax"):
            # 最后的输出结果为下一个字（字母）的独热表示，
            # 所以输出层的权重的维度为(lstm_size, vocab_num)
            softmax_x = tf.Variable(tf.truncated_normal([self.lstm_size, self.vocab_num], stddev=0.1), "softmax_x")
            softmax_b = tf.Variable(tf.zeros(self.vocab_num), "softmax_b")

        # logit_x 维度为(num_seqs * num_steps, vocab_num)
        self.logit_x = tf.matmul(x, softmax_x) + softmax_b
        self.prod_result = tf.nn.softmax(logits=self.logit_x, name="predict_result")

    def built_loss(self):
        with tf.name_scope("loss"):
            # self.target的维度为(num_seqs, num_steps)
            # 使用one_hot之后 y_one_hot维度为(num_seqs, num_steps, vocab_num)
            y_one_hot = tf.one_hot(self.targets, self.vocab_num)
            # y_shaped (num_seqs * num_steps, vocab_num)
            y_shaped = tf.reshape(y_one_hot, self.logit_x.shape)
            # 这个函数可以直接算出损失函数
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logit_x, labels=y_shaped)
            # loss函数, 将其平均下来
            self.loss = tf.reduce_mean(loss)

    def built_optimizer(self):
        # clipping gradicent函数，对梯度进行clip梯度下降
        # tf.train_varibles
        train_var = tf.trainable_variables()
        # clip_by_global_norm是梯度缩放输入是所有trainable向量的梯度，和所有trainable向量，
        # 返回值 第一个为clip好的梯度，第二个为globalnorm
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, train_var), self.grad_clip)
        # 构建一个优化器，此时学习率为learn_rate
        train_op = tf.train.AdamOptimizer(self.learn_rate)
        # 输入格式为grads, train_var, 来执行梯度的正式更新
        self.optimizer = train_op.apply_gradients(zip(grads, train_var))

    def train(self, batch_generate, max_steps, save_path, save_per_n, print_per_n):
        self.sesstion = tf.Session()
        with self.sesstion as sess:
            steps = 0
            sess.run(tf.global_variables_initializer())
            new_state = sess.run(self.initial_state)
            for x, y in batch_generate:
                steps += 1
                start_time = time.time()
                feed = {self.inputs: x,
                        self.targets: y,
                        self.initial_state: new_state,
                        self.keep_probs: self.init_keep_prob}
                new_loss, new_state, _ = sess.run([self.loss,
                                                   self.lstm_final_state,
                                                   self.optimizer], feed_dict=feed)
                end_time = time.time()
                if steps % print_per_n == 0:
                    print('step: {}/{}... '.format(steps, max_steps),
                          'loss: {:.4f}... '.format(new_loss),
                          '{:.4f} sec/batch'.format((end_time - start_time)))
                if (steps % save_per_n == 0):
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=steps)
                # 注意此处 控制的是随机梯度下降的次数
                if steps >= max_steps:
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=steps)

    def sample(self, sample_num, prime, vaocab_size):
        samples = [c for c in prime]
        sess = self.sesstion
        new_state = sess.run(self.initial_state)
        preds = np.ones((vaocab_size,))
        for c in samples:
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {
                self.inputs:x,
                self.initial_state: new_state,
                self.keep_probs: 1
            }
            preds, new_state = sess.run([self.prod_result, self.lstm_final_state],
                                            feed_dict=feed)

        thenewvocab = pick_top_n(preds, self.vocab_num)
        samples.append(thenewvocab)

        for i in range(sample_num):
            x = np.zeros((1, 1))
            x[0, 0] = thenewvocab
            feed = {
                self.inputs: x,
                self.initial_state: new_state,
                self.keep_probs: 1
            }
            preds, new_state = sess.run([self.prod_result, self.lstm_final_state], feed_dict=feed)
            thenewvocab = pick_top_n(preds, self.vocab_num)
            samples.append(thenewvocab)

        return np.array(samples)

    def load(self, checkpoint):
        self.sesstion = tf.Session()
        self.saver.restore(self.sesstion, checkpoint)
        print('Restored from: {}'.format(checkpoint))





