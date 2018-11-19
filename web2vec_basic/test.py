import collections
import tensorflow as tf

def isPrime(n):
    """This function return a number is a prime or not"""
    assert n >= 2
    from math import sqrt
    for i in range(2, int(sqrt(n))+1):
        if n % i == 0:
            return False
    return True

if __name__ == '__main__':
    # count = [['UNK', -1]]
    # word = ["two", "three", "two", "one", "three", "three"]
    # #print(collections.Counter(word))
    # n_words = len(word)
    # count.extend(collections.Counter(word).most_common(n_words - 1))
    # dictionary = dict()
    # print(count)
    # for word, _ in count:
    #     dictionary[word] = len(dictionary)
    # print(dictionary)
    # data = list()

    # input = [[0, 1],
    #          [2, 3]]
    # embeddings = tf.get_variable("embeddings", [4, 3])
    # a = tf.placeholder(tf.int32, shape=(2, 2), name="input")
    # result = tf.nn.embedding_lookup(embeddings, a)
    # c = tf.trainable_variables()
    # with tf.Session() as sess:
    #     init = tf.global_variables_initializer()
    #     sess.run(init)
    #     re, emb, cr = sess.run([result, embeddings, c], feed_dict={a: input})
    #     print(re)
    #     print(emb)
    #     print(cr)

    buffer = collections.deque(maxlen=3)
    testArray = [1, 2, 3, 4, 5]
    for i in testArray:
        buffer.append(i)
        print(buffer)

    embedding = [[1, 2, 3, 4],
                 [2, 3, 4, 5],
                 [3, 4, 5, 6]]
    squre = tf.square(embedding)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(squre))