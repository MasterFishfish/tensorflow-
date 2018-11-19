import tensorflow as tf
import numpy as np

class Attention():
    def __init__(self, num_seqs, input_steps, num_classes,
                 encoder_state_size, decoder_state_size, embedding_size, output_steps,
                 encoder_keep_probs=0.5, decoder_keep_probs=0.5,
                 grad_clip=5, learning_rate=0.01, decoder_layers=2, encoder_layers=2):

        self.num_seqs = num_seqs
        self.input_steps = input_steps
        self.vocab_size = num_classes
        self.encoder_state_size = encoder_state_size
        self.decoder_state_size = decoder_state_size
        self.embedding_size = embedding_size
        self.output_steps = output_steps
        self.encoder_keep_probs = encoder_keep_probs
        self.decoder_keep_probs = decoder_keep_probs
        self.grad_clip = grad_clip
        self.learning_rate = learning_rate
        self.decoder_layer = decoder_layers
        self.encoder_layer = encoder_layers

        tf.reset_default_graph()
        self.build_encoder()
        self.build_decoder()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()

    def get_a_cell(self, size, keep_prob):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=size)
        dropoutnn = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return dropoutnn

    def build_encoder(self):
        with tf.name_scope("encoder_input"):
            self.input = tf.placeholder(dtype=tf.int32,
                                        shape=[self.num_seqs, self.input_steps], name="input")
            self.keep_probs = tf.placeholder(dtype=tf.float32, name="keep_probs")
            self.target = tf.placeholder(dtype=tf.int32,
                                         shape=[self.num_seqs, self.output_steps], name="output")

            embeddings = tf.get_variable("embeddings", shape=[self.vocab_size, self.embedding_size])
            self.encoder_input = tf.nn.embedding_lookup(embeddings, self.input)

        with tf.name_scope("encoder_layers"):
            encoder = tf.nn.rnn_cell.MultiRNNCell([self.get_a_cell(self.encoder_state_size, self.encoder_keep_probs)
                                                   for i in range(self.encoder_layer)])
            self.encoder_init_state = encoder.zero_state(self.num_seqs, dtype=tf.float32)
            self.encoder_output, self.encoder_final_state = tf.nn.dynamic_rnn(cell=encoder, inputs=self.input,
                                                              initial_state=self.encoder_init_state)

    def build_decoder(self):
        with tf.name_scope("decoder_layers"):
            decoder = tf.nn.rnn_cell.MultiRNNCell([self.get_a_cell(self.decoder_state_size, self.decoder_keep_probs)
                                                  for i in range(self.decoder_layer)])
            self.decoder_init_state = decoder.zero_state(self.num_seqs, dtype=tf.float32)

            decoder_input = tf.concat([self.encoder_final_state for i in range(self.output_steps)], axis=1)
            self.decoder_input = tf.reshape(decoder_input,
                                            shape=[self.num_seqs, self.output_steps, self.encoder_state_size])
            self.decoder_output, _ = tf.nn.dynamic_rnn(cell=decoder, inputs=self.decoder_input,
                                                       initial_state=self.decoder_init_state)
        # 输出的结果为 ( num_seqs, output_steps, decoder_state_size )
        with tf.name_scope("decoder_result"):
            x = tf.reshape(tf.concat(self.decoder_output, axis=1), (-1, self.decoder_state_size))
            output_weights = tf.Variable(initial_value=tf.truncated_normal(shape=(self.decoder_state_size, self.vocab_size),
                                                                           stddev=0.1), name="logit_weights")
            output_bias = tf.Variable(initial_value=tf.random_uniform(shape=(self.vocab_size), minval=-1.0, maxval=1.0, dtype=tf.float32),
                                      name="logit_bias")
            self.logit_result = tf.matmul(x, output_weights) + output_bias
            self.soft_max_result = tf.nn.softmax(logits=self.logit_result, name="predict_result")

    def build_loss(self):
        with tf.name_scope("loss"):
            y_shaped = tf.reshape(tf.one_hot(self.target, self.vocab_size), shape=(-1, self.vocab_size))
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logit_result, labels=y_shaped)

    def build_optimizer(self):
        with tf.name_scope("optimizer"):
            train_variable = tf.trainable_variables()
            gradients, _ = tf.clip_by_global_norm(tf.gradients(ys=self.loss, xs=train_variable), clip_norm=self.grad_clip)
            train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.optimizer = train_op.apply_gradients(grads_and_vars=zip(gradients, train_variable))

    def train(self):
        self.session = tf.Session()
        with self.session as sess:
            init = tf.global_variables_initializer()

    def sample(self):
        pass

    def do_save(self):
        pass

if __name__ == "__main__":
    pass