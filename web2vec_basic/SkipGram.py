import random

import numpy as np
import tensorflow as tf
import collections

def built_dataset(words, n_words):
    count = [("UNK", -1)]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    # dictionary 储存的是词语及其出现频率的排名
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    # 储存的是每一个词语对应的的排名， 按照给定的词语list顺序一一对应
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    # reversed_dictionary 储存的是出现频率的排名及其对应的词语
    reversed_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return count, data, dictionary, reversed_dictionary

def batch_generate(batch_size, skip_window, num_skip, data):
    # 全局变量的data_index 使得在之后的训练之中，该函数被重复调用的时候，
    # 能够将data_index记忆下来
    global data_index
    # 使用滑动窗口来检测语料库中的相邻单词
    # skip_window为窗口的最中间的值, 作为选中的单词
    # num_skip则为将要从滑动窗口中取出的skip_window对应的词的相关词的个数
    # 所以必须满足下面的关系
    assert batch_size % num_skip == 0
    assert num_skip <= skip_window

    # batch_size用于存储词语
    # labels用于存储该词语的相关的词
    batch = np.ndarray(shape=(batch_size), dtype=tf.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=tf.int32)
    span = skip_window * 2 + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skip):
        target = skip_window
        target_avoid_list = [skip_window]
        for j in range(num_skip):
            while target in target_avoid_list:
                target = random.randint(0, span-1)
            target_avoid_list.append(target)
            batch[i*num_skip + j] = skip_window
            labels[i*num_skip + j, 0] = target
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

def main(vocabulary):
    # embedding_size每个词语转换成的嵌入向量的维度数
    embedding_size = 128
    batch_size = 128
    skip_window = 1
    num_skip = 2

    vocabulary_size = len(vocabulary)
    # 在训练的过程中, 会对模型进行验证
    test_words = 16
    test_field = 100
    # 在一百个数字中随机抽取16个数字
    test_example = np.random.choice(test_field, test_words, replace=False)

    # 损失噪声词的数量
    num_samples = 64

    train_inputs = tf.placeholder(tf.int32, shape=(batch_size))
    train_targets = tf.placeholder(tf.int32, shape=(batch_size, 1))
    valid_samples = tf.constant(test_example, dtype=tf.int32)

    #embeddings = tf.get_variable("embeddings", shape=(vocabulary_size, embedding_size))
    embeddings = tf.Variable(initial_value=tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),
                             dtype=tf.float32)
    # embeds 的维度为 (batch_size, embeddings_size)
    embeds = tf.nn.embedding_lookup(embeddings, train_inputs)

    weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/embedding_size))
    bias = tf.Variable(tf.zeros([vocabulary_size]))

    # tf.nn.nec_loss会自动的选取噪声词， 自动的生成损失
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=weights,
                       biases=bias,
                       num_sampled=num_samples,
                       inputs=embeds,
                       labels=train_targets,
                       num_classes=vocabulary_size)
    )
    # 构建优化器，梯度下降这个吧loss
    optimzer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 对embedding层做一次归一化
    # square会对每个元素进行乘方
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    nd_embeddings = embeddings / norm
    # 找出验证词的embedding计算他们对于所有词的相似度
    sample_embedding = tf.nn.embedding_lookup(embeddings, valid_samples)
    samiliar = tf.matmul(valid_samples, nd_embeddings, transpose_b=True)

    # 初始化
    init = tf.global_variables_initializer()

    num_steps = 10001
    with tf.Sesstion() as sess:
        sess.run(init)

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = batch_generate(batch_size, skip_window=1, num_skip=2)
            feed = {train_inputs: batch_inputs, train_targets: batch_labels}

            _, loss_val = sess.run([optimzer, loss_val], feed_dict=feed)
            average_loss += loss_val






