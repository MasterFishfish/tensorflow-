import tensorflow as tf
import codecs
import numpy as np
from readfile import batch_generator, TextConverter
from model import CharRNN
import os

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('name', 'default', 'name of the model')
tf.flags.DEFINE_integer('num_seqs', 100, 'number of seqs in one batch')
tf.flags.DEFINE_integer('num_steps', 100, 'length of one seq')
tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.flags.DEFINE_string('input_file', '', 'utf8 encoded text file')
tf.flags.DEFINE_integer('max_steps', 100000, 'max steps to train')
tf.flags.DEFINE_integer('save_per_n', 1000, 'save the model every n steps')
tf.flags.DEFINE_integer('print_per_n', 10, 'log to the screen every n steps')
tf.flags.DEFINE_integer('max_vocab', 3500, 'max char number')

def main(_):
    model_path = os.path.join("model", FLAGS.name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    with open(FLAGS.input_file, encoding='utf-8') as f:
        text = f.read()
    converter = TextConverter(text=text, filename=None, max_vocab=FLAGS.max_vocab)
    converter.save_to_file(os.path.join(model_path, 'converter.pkl'))

    arr = converter.text_to_arr(text=text)
    print(converter.vocab_size())
    g = batch_generator(arr, FLAGS.num_seqs, FLAGS.num_steps)
    model = CharRNN(num_class=converter.vocab_size(),
                    num_seqs=FLAGS.num_seqs,
                    num_steps=FLAGS.num_steps,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    learn_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size
                    )
    model.train(batch_generate=g,
                max_steps=FLAGS.max_steps,
                save_path=model_path,
                save_per_n=FLAGS.save_per_n,
                print_per_n=FLAGS.print_per_n,
                )

if __name__ == '__main__':
    tf.app.run()