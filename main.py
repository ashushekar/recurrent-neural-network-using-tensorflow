"""
To Introduce basic RNNs and how to work with sequences, we take a simple
sequential view of images: we look at each image in our data as a sequence of rows (or
columns). In our MNIST data, this just means that each 28×28-pixel image can be
viewed as a sequence of length 28, each element in the sequence a vector of 28 pixels
(see Figure 5-3). Then, the temporal dependencies in the RNN can be imaged as a scanner
head, scanning the image from top to bottom (rows) or left to right (columns).
"""
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
print(tf.__version__)

MNIST = input_data.read_data_sets('/tmp/data/', one_hot=True)

# Define Some Parameters
SIZE = 28
TIMESTEPS = 28
NUMCLASSES = 10
BATCHSIZE = 128
HIDDENLAYERSIZE = 128

# Where to save tensorboard model summaries
LOGDIR = 'logs/RNN_with_summaries'

# create placeholders for inputs and labels
_inputs = tf.placeholder(tf.float32, shape=[None, TIMESTEPS, SIZE], name='inputs')
y = tf.placeholder(tf.float32, shape=[None, NUMCLASSES], name='labels')

def variable_summaries(var):
    """
    This helper function, taken from the official TensorFlow documentation,
    simply adds some ops that take care of logging summaries.

    :param var: tensor variable
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# Weight and Bias for input layer and hidden layer
with tf.name_scope('rnn_weights'):
    with tf.name_scope('W_x'):
        Wx = tf.Variable(tf.zeros(shape=[SIZE, HIDDENLAYERSIZE]))
        variable_summaries(Wx)
    with tf.name_scope('bias'):
        b_rnn = tf.Variable(tf.zeros(shape=[HIDDENLAYERSIZE]))
        variable_summaries(b_rnn)
    with tf.name_scope('W_h'):
        Wh = tf.Variable(tf.zeros(shape=[HIDDENLAYERSIZE, HIDDENLAYERSIZE]))
        variable_summaries(Wh)


# Weights for the output layers
with tf.name_scope('linear_layer_weights') as scope:
    with tf.name_scope('W_linear'):
        Wl = tf.Variable(tf.truncated_normal([HIDDENLAYERSIZE, NUMCLASSES],
                                             mean=0, stddev=0.01))
        variable_summaries(Wl)
    with tf.name_scope('Bias_linear'):
        bl = tf.Variable(tf.truncated_normal([NUMCLASSES],
                                             mean=0, stddev=0.01))
        variable_summaries(bl)


def rnn_step(previous_hidden_state, x):
    """
    We now create a function that implements the vanilla RNN step.
    :param previous_hidden_state: Previous hidden state input in tensor
    :param x: Current input
    :return: Current hidden state output
    """
    current_hidden_state = tf.tanh(tf.matmul(previous_hidden_state, Wh) +
                                   tf.matmul(x, Wx) + b_rnn)

    return current_hidden_state


# Processing inputs to work with scan function
# Current input shape: (BATCHSIZE, TIMESTEPS, SIZE)
# New input shape: (TIMESTEPS, BATCHSIZE, SIZE)
processed_input = tf.transpose(_inputs, perm=[1, 0, 2])

initial_hidden= tf.zeros([BATCHSIZE, HIDDENLAYERSIZE])
# Getting all state vectors across time
all_hidden_states = tf.scan(rnn_step, processed_input,
                            initializer=initial_hidden, name='states')


def get_linear_layer(hidden_state):
    """
    Apply linear layer to state vector
    :param hidden_state: Hidden state tensor output
    :return: last output
    """
    return tf.matmul(hidden_state, Wl) + bl

with tf.name_scope('linear_layer_weights') as scope:
    # Iterate across time, apply linear layer to all RNN outputs
    all_outputs = tf.map_fn(get_linear_layer, all_hidden_states)
    # Get Last Output
    output = all_outputs[-1]
    tf.summary.histogram('outputs', output)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output,
                                                                              labels=y))
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction= tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100
    tf.summary.scalar('accuracy', accuracy)

# Merge all summaries
merged = tf.summary.merge_all()

# Get a small test
test_data = MNIST.test.images[:BATCHSIZE].reshape((-1, TIMESTEPS, SIZE))
test_label = MNIST.test.labels[:BATCHSIZE]

with tf.Session() as sess:
    # Write summaries to LOG_DIR -- used by TensorBoard
    train_writer = tf.summary.FileWriter(LOGDIR + '/train',
                                         graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter(LOGDIR + '/test',
                                        graph=tf.get_default_graph())

    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        batch_x, batch_y = MNIST.train.next_batch(BATCHSIZE)
        # Reshape the data to get 28 sequences of 28 pixels
        batch_x = batch_x.reshape((BATCHSIZE, TIMESTEPS, SIZE))

        summary, _ = sess.run([merged, train_step], feed_dict={
            _inputs: batch_x, y: batch_y
        })

        # add the summary to logs
        train_writer.add_summary(summary, i)

        if i % 1000:
            acc, loss = sess.run([accuracy, cross_entropy], feed_dict={
                _inputs: batch_x, y: batch_y
            })
            print("Iter " + str(i) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

        if i % 10:
            summary, acc = sess.run([merged, accuracy], feed_dict={
                _inputs: test_data, y: test_label
            })
            # add the summary to logs
            test_writer.add_summary(summary, i)

    test_acc = sess.run(accuracy, feed_dict={_inputs: test_data, y: test_label})
    print("Test Accuracy:", test_acc)
