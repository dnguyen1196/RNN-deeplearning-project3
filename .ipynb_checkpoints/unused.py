import getpass
import math
import sys
import time

import numpy as np
from copy import deepcopy

from utils import calculate_perplexity, get_ptb_dataset, Vocab
from utils import ptb_iterator, sample

import tensorflow as tf
# from tensorflow.python.ops.seq2seq import sequence_loss
# from tensorflow.contrib.legacy_seq2seq.python.ops import sequence_loss
# from tensorflow.contrib import seq2seq
# import tensorflow.contrib.seq2seq.sequence_loss as sequence_loss
# import seq2seq.sequence_loss

from tensorflow.contrib.legacy_seq2seq import sequence_loss
from model import LanguageModel

# Let's set the parameters of our model
# http://arxiv.org/pdf/1409.2329v4.pdf shows parameters that would achieve near
# SotA numbers

class Config(object):
  """Holds model hyperparams and data information.
  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  batch_size = 64
  embed_size = 50
  hidden_size = 100
  num_steps = 10
  max_epochs = 16
  early_stopping = 2
  dropout = 0.9
  lr = 0.001

class RNNLM_Model(LanguageModel):

  def load_data(self, debug=False):
    """Loads starter word-vectors and train/dev/test data."""
    self.vocab = Vocab()
    self.vocab.construct(get_ptb_dataset('train'))
    self.encoded_train = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('train')],
        dtype=np.int32)
    self.encoded_valid = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('valid')],
        dtype=np.int32)
    self.encoded_test = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('test')],
        dtype=np.int32)
    if debug:
      num_debug = 1024
      self.encoded_train = self.encoded_train[:num_debug]
      self.encoded_valid = self.encoded_valid[:num_debug]
      self.encoded_test = self.encoded_test[:num_debug]

  def add_placeholders(self):
    """Generate placeholder variables to represent the input tensors
    These placeholders are used as inputs by the rest of the model building
    code and will be fed data during training.  Note that when "None" is in a
    placeholder's shape, it's flexible
    Adds following nodes to the computational graph.
    (When None is in a placeholder's shape, it's flexible)
    input_placeholder: Input placeholder tensor of shape
                       (None, num_steps), type tf.int32
    labels_placeholder: Labels placeholder tensor of shape
                        (None, num_steps), type tf.float32
    dropout_placeholder: Dropout value placeholder (scalar),
                         type tf.float32
    Add these placeholders to self as the instance variables
  
      self.input_placeholder
      self.labels_placeholder
      self.dropout_placeholder
    (Don't change the variable names)
    """
    self.input_placeholder = tf.placeholder(tf.int32, [None, self.config.num_steps], "input_placeholder")
    self.labels_placeholder = tf.placeholder(tf.float32, [None, self.config.num_steps], "labels_placeholder")
    self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout_placeholder")

  def add_embedding(self):
    """Add embedding layer.
    Hint: This layer should use the input_placeholder to index into the
          embedding.
    Hint: You might find tf.nn.embedding_lookup useful.
    Hint: You might find tf.split, tf.squeeze useful in constructing tensor inputs
    Hint: Check the last slide from the TensorFlow lecture.
    Hint: Here are the dimensions of the variables you will need to create:
      L: (len(self.vocab), embed_size)
    Returns:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    """
    # self.embeddings = tf.get_variable("embeddings",
    #                                   [len(self.vocab), self.config.embed_size])
    self.embeddings = tf.Variable(
      tf.random_uniform([len(self.vocab), self.config.embed_size], -1.0, 1.0))
    # The embedding lookup is currently only implemented for the CPU
    with tf.device('/cpu:0'):
      all_embeddings = tf.nn.embedding_lookup(self.embeddings, self.input_placeholder) # batch x step x embed
      embeddings_list = tf.split(all_embeddings, self.config.num_steps, 1)
      embeddings = [tf.squeeze(item, squeeze_dims=(1,)) for item in embeddings_list]
      return embeddings

  def add_projection(self, rnn_outputs):
    """Adds a projection layer.
    The projection layer transforms the hidden representation to a distribution
    over the vocabulary.
    Hint: Here are the dimensions of the variables you will need to
          create
          U:   (hidden_size, len(vocab))
          b_2: (len(vocab),)
    Args:
      rnn_outputs: List of length num_steps, each of whose elements should be
                   a tensor of shape (batch_size, embed_size).
    Returns:
      outputs: List of length num_steps, each a tensor of shape
               (batch_size, len(vocab)
    """
    # self.U = tf.get_variable("U", [self.config.hidden_size, len(self.vocab)])
    # self.b_2 = tf.get_variable("b_2", [len(self.vocab)])
    self.U = tf.Variable([self.config.hidden_size, len(self.vocab)], name="U", dtype=tf.float32)
    self.b_2 = tf.Variable([len(self.vocab)], name="b_2", dtype=tf.float32)

    outputs = [tf.matmul(o, self.U) + b_2 for o in rnn_outputs]

    return outputs

  def add_loss_op(self, output):
    """Adds loss ops to the computational graph.
    Hint: Use tensorflow.python.ops.seq2seq.sequence_loss to implement sequence loss. 
    Args:
      output: A tensor of shape (None, self.vocab)
    Returns:
      loss: A 0-d tensor (scalar)
    """
    return sequence_loss([output],
                         [tf.reshape(
                             self.labels_placeholder,
                             [self.config.batch_size * self.config.num_steps, -1])],
                         [tf.constant(1.0)])

  def add_training_op(self, loss):
    """Sets up the training Ops.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train. See 
    https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer
    for more information.
    Hint: Use tf.train.AdamOptimizer for this model.
          Calling optimizer.minimize() will return a train_op object.
    Args:
      loss: Loss tensor, from cross_entropy_loss.
    Returns:
      train_op: The Op for training.
    """
    return tf.train.AdamOptimizer(self.config.lr).minimize(loss)

  def __init__(self, config):
    self.config = config
    self.load_data(debug=False)
    self.add_placeholders()
    self.inputs = self.add_embedding()
    self.rnn_outputs = self.add_model(self.inputs)
    self.outputs = self.add_projection(self.rnn_outputs)
    # We want to check how well we correctly predict the next word
    # We cast o to float64 as there are numerical issues at hand
    # (i.e. sum(output of softmax) = 1.00000298179 and not 1)
    self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
    # Reshape the output into len(vocab) sized chunks - the -1 says as many as
    # needed to evenly divide
    output = tf.stack(self.outputs, 1)
    self.calculate_loss = self.add_loss_op(output)
    self.train_step = self.add_training_op(self.calculate_loss)


  def add_model(self, inputs):
    """Creates the RNN LM model.
    In the space provided below, you need to implement the equations for the
    RNN LM model. Note that you may NOT use built in rnn_cell functions from
    tensorflow.
    Hint: Use a zeros tensor of shape (batch_size, hidden_size) as
          initial state for the RNN. Add this to self as instance variable
          self.initial_state
  
          (Don't change variable name)
    Hint: Add the last RNN output to self as instance variable
          self.final_state
          (Don't change variable name)
    Hint: Make sure to apply dropout to the inputs and the outputs.
    Hint: Use a variable scope (e.g. "RNN") to define RNN variables.
    Hint: Perform an explicit for-loop over inputs. You can use
          scope.reuse_variables() to ensure that the weights used at each
          iteration (each time-step) are the same. (Make sure you don't call
          this for iteration 0 though or nothing will be initialized!)
    Hint: Here are the dimensions of the various variables you will need to
          create:
          H: (hidden_size, hidden_size)
          I: (embed_size, hidden_size)
          b_1: (hidden_size,)
    Args:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    Returns:
      outputs: List of length num_steps, each of whose elements should be
               a tensor of shape (batch_size, hidden_size)
    """
    # self.H = tf.get_variable("H", [self.config.hidden_size, self.config.hidden_size])
    # self.I = tf.get_variable("I", [self.config.embed_size, self.config.hidden_size])
    # self.b_1 = tf.get_variable("b_1", [self.config.hidden_size])

    # self.initial_state = tf.zeros((self.config.batch_size, self.config.hidden_size))

    # # inputs.shape == (num_steps, batch_size, embed_size)
    # # Can convert it to (batch_size, num_steps, embed_size) # so that will work on
    # # RNN code


    # rnn_cell = tf.keras.layers.SimpleRNNCell(self.config.hidden_size)

    # # inputs
    # rnn_outputs, rnn_final_state = tf.nn.dynamic_rnn(rnn_cell, inputs,
    #                                initial_state=self.initial_state,
    #                                dtype=tf.float32)

    # print(rnn_outputs.shape)
    # rnn_outputs = []
    # for time_step in range(self.config.num_steps):
    #   state = tf.nn.sigmoid(tf.matmul(state, self.H) + tf.matmul(inputs[time_step], self.I) + self.b_1)
    #   rnn_outputs.append(state)
    

    # # final state should have shape = (batch_size, hidden_size)

    # self.final_state = state

    # return rnn_outputs
    rnn_outputs = []
    self.initial_state = tf.zeros([self.config.batch_size,self.config.hidden_size])
    with tf.variable_scope("RNN",reuse = tf.AUTO_REUSE):
        self.H = tf.get_variable("H",shape=[self.config.hidden_size,self.config.hidden_size],\
                                        initializer=tf.contrib.layers.xavier_initializer())
        self.I = tf.get_variable("I",shape=[self.config.embed_size,self.config.hidden_size],\
                                        initializer=tf.contrib.layers.xavier_initializer())
    
        self.b1 = tf.get_variable("b1",shape=[self.config.hidden_size],\
                                        initializer=tf.contrib.layers.xavier_initializer())
    
    h = self.initial_state
    for i in range(self.config.num_steps):
        curr_input = inputs[i]
        curr_input = tf.nn.dropout(curr_input,keep_prob=self.dropout_placeholder)
        h = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(h, self.H) + tf.matmul(curr_input, self.I) + self.b1)\
                            ,keep_prob=self.dropout_placeholder)
        rnn_outputs.append(h)
    self.final_state = h        
    ### END YOUR CODE
    return rnn_outputs

  def run_epoch(self, session, data, train_op=None, verbose=10):
    config = self.config
    dp = config.dropout
    if not train_op:
      train_op = tf.no_op()
      dp = 1
    total_steps = sum(1 for x in ptb_iterator(data, config.batch_size, config.num_steps))
    total_loss = []
    state = self.initial_state.eval()
    for step, (x, y) in enumerate(
      ptb_iterator(data, config.batch_size, config.num_steps)):
      # We need to pass in the initial state and retrieve the final state to give
      # the RNN proper history
      feed = {self.input_placeholder: x,
              self.labels_placeholder: y,
              self.initial_state: state,
              self.dropout_placeholder: dp}
      loss, state, _ = session.run(
          [self.calculate_loss, self.final_state, train_op], feed_dict=feed)
      total_loss.append(loss)
      if verbose and step % verbose == 0:
          sys.stdout.write('\r{} / {} : pp = {}'.format(
              step, total_steps, np.exp(np.mean(total_loss))))
          sys.stdout.flush()
    if verbose:
      sys.stdout.write('\r')
    return np.exp(np.mean(total_loss))

def feed_token(session, model, state, token):
  state, prediction = session.run(
      [model.final_state, model.predictions[-1]],
      feed_dict={model.initial_state: state,
                 model.input_placeholder: [[token]],
                 model.dropout_placeholder: 1.0})
  return state, prediction

def generate_text(session, model, config, starting_text='<eos>',
                  stop_length=100, stop_tokens=None, temp=1.0):
  """Generate text from the model.
  Hint: Create a feed-dictionary and use sess.run() to execute the model. Note
        that you will need to use model.initial_state as a key to feed_dict
  Hint: Fetch model.final_state and model.predictions[-1]. (You set
        model.final_state in add_model() and model.predictions is set in
        __init__)
  Hint: Store the outputs of running the model in local variables state and
        y_pred (used in the pre-implemented parts of this function.)
  Args:
    session: tf.Session() object
    model: Object of type RNNLM_Model
    config: A Config() object
    starting_text: Initial text passed to model.
  Returns:
    output: List of word idxs
  """
  state = model.initial_state.eval()
  # Imagine tokens as a batch size of one, length of len(tokens[0])
  tokens = [model.vocab.encode(word) for word in starting_text.split()]
  for i in xrange(len(tokens) - 1):
    state, _ = feed_token(session, model, state, tokens[i])
  for i in xrange(stop_length):
    state, prediction = feed_token(session, model, state, tokens[-1])
    next_word_idx = sample(prediction.flatten(), temperature=temp)
    tokens.append(next_word_idx)
    if stop_tokens and model.vocab.decode(tokens[-1]) in stop_tokens:
      break
  output = [model.vocab.decode(word_idx) for word_idx in tokens]
  return output

def generate_sentence(session, model, config, *args, **kwargs):
  """Convenice to generate a sentence from the model."""
  return generate_text(session, model, config, *args, stop_tokens=['<eos>'], **kwargs)

def test_RNNLM():
  config = Config()
  gen_config = deepcopy(config)
  gen_config.batch_size = gen_config.num_steps = 1

  # We create the training model and generative model
  with tf.variable_scope('RNNLM', reuse=tf.AUTO_REUSE) as scope:
    model = RNNLM_Model(config)
    # This instructs gen_model to reuse the same variables as the model above
    scope.reuse_variables()
    gen_model = RNNLM_Model(gen_config)

  init = tf.initialize_all_variables()
  saver = tf.train.Saver()

  with tf.Session() as session:
    best_val_pp = float('inf')
    best_val_epoch = 0
  
    session.run(init)
    for epoch in xrange(config.max_epochs):
      print('Epoch {}'.format(epoch))
      start = time.time()
      ###
      train_pp = model.run_epoch(
          session, model.encoded_train,
          train_op=model.train_step)
      valid_pp = model.run_epoch(session, model.encoded_valid)
      print('Training perplexity: {}'.format(train_pp))
      print('Validation perplexity: {}'.format(valid_pp))
      if valid_pp < best_val_pp:
        best_val_pp = valid_pp
        best_val_epoch = epoch
        saver.save(session, './ptb_rnnlm.weights')
      if epoch - best_val_epoch > config.early_stopping:
        break
      print('Total time: {}'.format(time.time() - start))
      
    saver.restore(session, 'ptb_rnnlm.weights')
    test_pp = model.run_epoch(session, model.encoded_test)
    print('=-=' * 5)
    print('Test perplexity: {}'.format(test_pp))
    print('=-=' * 5)

    starting_snippets = ['in boston', 'they have', 'please', 'today', 'the president']
    for starting_text in starting_snippets:
      print('\n')
      print(' '.join(generate_sentence(
          session, gen_model, gen_config, starting_text=starting_text, temp=1.0)))

if __name__ == "__main__":
    test_RNNLM()
