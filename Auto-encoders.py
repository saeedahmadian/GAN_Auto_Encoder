import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

num_nodes = 3
num_features = 24*4
num_deep_feature = 24*4*32 #768
learning_rate = 0.0002
epochs = 25
num_examples = 1000
batch_size = 16
display_freq=10
mean=20
std=5

def next_batch(X, batch_size):
    """ get next batch of samples"""
    for i in range(0, X.shape[0], batch_size):
        yield X[i: i + batch_size, :, :]


def randomize(X):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(X.shape[0])
    X = X[permutation, :, :, :]
    return X

def encod(x,num_deep_feature):
    with tf.variable_scope('encod_var'):
        h1 = tf.layers.conv2d(x, 32*4, kernel_size=[1, 5], strides=[1, 2], padding='same', activation=tf.nn.relu)
        h2 = tf.layers.conv2d(h1, 64*4, kernel_size=[1, 5], strides=[1, 2], padding='same', activation=tf.nn.relu)
        h3 = tf.layers.conv2d(h2, 128*4, kernel_size=[1, 5], strides=[1, 2], padding='same', activation=tf.nn.relu)

        h3 = tf.layers.average_pooling2d(h3, 1, 2)
        h3 = tf.contrib.layers.flatten(h3)
        h4 = tf.layers.dense(h3, num_deep_feature)
        h4 = tf.nn.relu(h4)
        return h4


def decod(x, num_deep_feature):
    with tf.variable_scope('decod_var'):
        h1 = tf.layers.dense(x, units=num_deep_feature, use_bias=False)
        h1 = tf.reshape(h1, shape=[-1, 1, 3, 64 * 4*4], name='h1_reshape')
        h2 = tf.layers.conv2d_transpose(inputs=h1, filters=64 * 4*4, kernel_size=(1, 5), strides=(1, 2), padding='same',
                                        use_bias=False, activation=tf.nn.relu)
        h3 = tf.layers.conv2d_transpose(inputs=h2, filters=32 * 2*4, kernel_size=(1, 5), strides=(1, 2), padding='same',
                                        use_bias=False, activation=tf.nn.relu)
        h4 = tf.layers.conv2d_transpose(inputs=h3, filters=1*4, kernel_size=(1, 5), strides=(1, 2), padding='same',
                                        use_bias=False, activation=tf.nn.relu)
        #h5 = tf.layers.conv2d_transpose(inputs=h4, filters=1, kernel_size=(1, 5), strides=(1, 2), padding='same',
                                       # use_bias=False, activation=tf.nn.relu)
        return h4



auto_input = tf.placeholder(tf.float32, shape=[None, num_nodes, num_features, 1],name='Auto_input')
dataSet=np.zeros([num_examples,1,num_features,1],'float32')


deep_feature = encod(auto_input,num_deep_feature)
new_represent = decod(deep_feature,num_deep_feature)

loss = tf.reduce_mean(tf.losses.mean_squared_error(auto_input, new_represent), name='loss')
tf.summary.scalar('loss', loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
merged = tf.summary.merge_all()

for i in range(num_examples):
    for j in range(num_nodes):
        dataSet[i, j, :, 0] = np.random.normal(loc=mean, scale=std, size=num_features)

with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter('logs_auto', sess.graph)
    num_tr_iter = int(num_examples / batch_size)
    global_step = 0
    for epoch in range(epochs):
        print('Training epoch: {}'.format(epoch + 1))
        dataSet=randomize(dataSet)
        for iteration in range(num_tr_iter):
            batch_x=dataSet[iteration: iteration + batch_size, :, :,:]

            global_step += 1
            # Run optimization op (backprop)
            feed_dict_batch = {auto_input: batch_x}
            test=sess.run(new_represent,feed_dict=feed_dict_batch)

            _, summary_tr = sess.run([optimizer, merged], feed_dict=feed_dict_batch)
            train_writer.add_summary(summary_tr, global_step)

            if iteration % display_freq == 0:
                # Calculate and display the batch loss and accuracy
                loss_batch = sess.run(loss,
                                      feed_dict=feed_dict_batch)
                print("iter {0:3d}:\t Reconstruction loss={1:.3f}".
                      format(iteration, loss_batch))
                fig = plt.figure()
                ax = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                ax.plot(np.arange(1, 25), np.squeeze(test[0, :, :, 0]))
                ax2.plot(np.arange(1, 25), np.squeeze(batch_x[0, :, :, 0]))
                fig.savefig(os.path.join('figs_auto', '{}_{}.png'.format(iteration, global_step)), bbox_inches='tight')


                # Run validation after every epoch
        #x_valid_original  = mnist.validation.images
        #x_valid_noisy = x_valid_original + noise_level * np.random.normal(loc=0.0, scale=1.0, size=x_valid_original.shape)

        #feed_dict_valid = {x_original: x_valid_original, x_noisy: x_valid_noisy}
        #loss_valid = sess.run(loss, feed_dict=feed_dict_valid)
        #print('---------------------------------------------------------')
        #print("Epoch: {0}, validation loss: {1:.3f}".
        #      format(epoch + 1, loss_valid))
        #print('---------------------------------------------------------')


a=1