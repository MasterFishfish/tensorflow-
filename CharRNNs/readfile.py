import numpy as np
import pickle
import copy

# arr为一串数字数组，每一个元素为对应字的使用频率排名的数字
# n_seqs为序列的长度， 表示的是一个batch里面有多少个句子
# n_steps为句子的长度，表示的是一个句子的长度
def batch_generator(arr, n_seqs, n_steps):
    arr = copy.copy(arr)
    batch_size = n_seqs * n_steps
    batch_num = int(len(arr) / batch_size)
    arr = arr[:batch_size * batch_num]
    arr = np.array(arr)
    arr = arr.reshape((n_seqs, -1))
    # 每一次随机梯度训练完所有的数据之后，重新开始训练，
    # 在RNN类中的train方法来控制训练的次数
    endnum = (np.shape(arr))[1]
    while True:
        np.random.shuffle(arr)
        # range(start, end) 左闭右开
        for n in range(0, endnum, n_steps):
            x = arr[:, n:n + n_steps]
            y = np.zeros_like(x)

            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y

# text为一段文本字符串
# filename 意思为可以直接导入的vocab文本
# max_vocab意思为最大的字类的总数
# 该类的主要作用是用字母出现的频率的序号来表示字母
# 并通过字母和序号之间的转换，实现字符串序列和数字向量之间的转换
class TextConverter():
    def __init__(self, text=None, filename=None, max_vocab=5000):
        if filename is not None:
            with open(filename, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            wordlist = set(text)
            # 定义一个字典储存字及其出现的频率
            vocab_count = {}
            for word in wordlist:
                vocab_count[word] = 0
            for word in text:
                vocab_count[word] += 1
            vocab_list = []
            for word in wordlist:
                vocab_list.append((word, vocab_count[word]))
            # 以出现频率为倒序 排列所有的词和词语频率
            vocab_list.sort(key=lambda x:x[1], reverse=True)
            if len(vocab_list) > max_vocab:
                vocab_list = vocab_list[:max_vocab]
            vocab = [x[0] for x in vocab_list]
            # vocab 是一个按照字的出现频率排序的list, 长度<=max_vocab
            self.vocab = vocab

        # 每个字母为key, 每一个value为该字母的出现频率倒序排列后的序号
        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        # 每个序号为key, 每个value为字母
        self.int_to_word_table = dict(enumerate(self.vocab))

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, Index):
        if Index < len(self.vocab):
            return self.int_to_word_table[Index]
        elif Index == len(self.vocab):
            return '<unk>'
        else:
            raise Exception('Unknown index!')

    def vocab_size(self):
        return len(self.vocab) + 1

    # 文本转数组
    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)

    # 数组转文本
    def arr_to_text(self, arr):
        words = []
        for data in arr:
            words.append(self.int_to_word(data))
        # 按照""来链接每一个list元素，使所有的字母排列成为单词组成的句子
        return "".join(words)

    def save_to_file(self,filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)

