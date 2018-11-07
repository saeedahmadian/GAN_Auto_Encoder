import os
import tensorflow as tf
import numpy as np
import time
from sklearn import decomposition
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('fast')


class CyberGAN(object):
    def __init__(self, config):

        self.num_samples = config.num_samples
        self.num_features = config.num_features
        self.num_nodes = config.num_nodes
        self.num_class = config.num_class
        self.hid_layer1 = config.hid_layer1
        self.hid_layer2 = config.hid_layer2
        self.learning_rate=config.learning_rate
        self.model_name=config.model_name
        self.checkpoint_dir=config.checkpoint_dir

        self.batch_size = config.batch_size
        self.epoch = config.epoch
        self.num_deep_feature=config.num_deep_feature

        self.alpha = config.alpha
        self.beta = config.beta

        self.use_autoencoder = config.use_autoencoder

        # Inputs
        self.gen_input = tf.placeholder(tf.float32, shape=[None, self.num_nodes], name='gen_input')
        self.disc_input = tf.placeholder(tf.float32, shape=[None, self.num_nodes, self.num_features, 1],
                                         name='disc_input')
        if self.use_autoencoder:
            # Auto-Encoders input
            self.auto_input = tf.placeholder(tf.float32, shape=[None, self.num_nodes, self.num_features, 1],
                                             name='Auto_input')


        # Targets (Real input: 1, Fake input: 0)
        self.disc_target = tf.placeholder(tf.float32, shape=[None, 1], name='disc_target')
        self.gen_target = tf.placeholder(tf.float32, shape=[None, 1], name='gen_target')



        self.build_model()

    def build_model(self):

        if self.use_autoencoder:
            # Build Auto
            self.deep_feature = self.encod(self.auto_input)
            self.new_represent = self.decod(self.deep_feature)

            # Auto-encoder Loss
            self.auto_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.disc_input, self.new_represent),
                                            name='Auto_loss')
            tf.summary.scalar('Auto_encoder_loss', self.auto_loss)

        # Build generator
        gen_out = self.generator(self.gen_input)
        self.gen_out = gen_out

        # Build Discriminator Networks (one from noise input, one from generated samples)
        disc_out_real = self.discriminator(self.disc_input)
        disc_out_fake = self.discriminator(gen_out, reuse=True)

        # Build the stacked generator/discriminator
        stacked_out = self.discriminator(gen_out, reuse=True)

        # Build Loss 1
        # Discriminator tries to discriminate real or fake input
        self.disc_real_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
            logits=disc_out_real, multi_class_labels=tf.ones_like(disc_out_real)))
        self.dist_fake_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
            logits=disc_out_fake, multi_class_labels=tf.zeros_like(disc_out_fake)))

        self.disc_loss = self.disc_real_loss + self.dist_fake_loss
        tf.summary.scalar('discriminator_real_loss',self.disc_real_loss)
        tf.summary.scalar('discriminator_fake_loss',self.dist_fake_loss)
        tf.summary.scalar('discriminator_loss', self.disc_loss)

        # Accuracy
        correct_prediction_real = tf.equal(tf.round(tf.nn.sigmoid(disc_out_real)), tf.ones_like(disc_out_real),
                                           name='correct_pred_real')

        correct_prediction_fake = tf.equal(tf.round(tf.nn.sigmoid(disc_out_fake)), tf.zeros_like(disc_out_fake),
                                           name='correct_pred_fake')
        self.correct_prediction = tf.concat((correct_prediction_real, correct_prediction_fake), axis=0)


        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name='accuracy')

        tf.summary.scalar('accuracy', self.accuracy)

        if self.use_autoencoder:
            # Generator tries to fool discriminator => label=1
            gen_loss1 = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=stacked_out,
                                                                       multi_class_labels=tf.ones_like(stacked_out)))
            gen_loss2 = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.new_represent,
                                                                    predictions=self.gen_out))
            self.gen_loss = self.alpha * gen_loss1 + self.beta * gen_loss2
        else:
            self.gen_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=stacked_out,
                                                                        multi_class_labels=tf.ones_like(stacked_out)))

        tf.summary.scalar('generator_loss', self.gen_loss)


        # Build Optimizers
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        optimizer_disc = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        if self.use_autoencoder:
            optimizer_auto = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # Training Variables for each optimizer
        # By default in TensorFlow, all variables are updated by each optimizer, so we
        # need to precise for each one of them the specific variables to update.
        # Generator Network Variables
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        # Discriminator Network Variables
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        # Create training operations
        if self.use_autoencoder:
            auto_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='auto_encoder')
            self.train_auto = optimizer_auto.minimize(self.auto_loss, var_list=auto_vars)

        self.train_gen = optimizer_gen.minimize(self.gen_loss, var_list=gen_vars)
        self.train_disc = optimizer_disc.minimize(self.disc_loss, var_list=disc_vars)

        # Initialize the variables (i.e. assign their default value)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.merged = tf.summary.merge_all()

    def generator(self, z):
        with tf.variable_scope("generator"):

            h1 = tf.layers.dense(z, units=3*6*8*64, use_bias=False)
            h1 = tf.reshape(h1, shape=[-1, 3, 6, 64*8], name='h1_reshape')
            h2 = tf.layers.conv2d_transpose(inputs=h1, filters=64*8, kernel_size=(1, 5), strides=(1, 2), padding='same',
                                            use_bias=False, activation=tf.nn.relu)
            h3 = tf.layers.conv2d_transpose(inputs=h2, filters=32*4, kernel_size=(1, 5), strides=(1, 2), padding='same',
                                            use_bias=False, activation=tf.nn.relu)
            h4 = tf.layers.conv2d_transpose(inputs=h3, filters=16*2, kernel_size=(1, 5), strides=(1, 2), padding='same',
                                            use_bias=False, activation=tf.nn.relu)
            h5 = tf.layers.conv2d_transpose(inputs=h4, filters=1, kernel_size=(1, 5), strides=(1, 2), padding='same',
                                            use_bias=False, activation=tf.nn.relu)
            #h4 = tf.layers.max_pooling2d(h4, (10, 1), (1, 1), padding='valid')
            #tf.nn.relu(h4)

            return h5

    def discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):

            h1 = tf.layers.conv2d(x, 8, kernel_size=[1, 5], strides=[1, 2], padding='same', activation=tf.nn.relu)
            h2 = tf.layers.conv2d(h1, 16, kernel_size=[1, 5], strides=[1, 2], padding='same', activation=tf.nn.relu)
            h3 = tf.layers.conv2d(h2, 32, kernel_size=[1, 5], strides=[1, 2], padding='same', activation=tf.nn.relu)

            # h3 = tf.layers.average_pooling2d(h3, 1, 2)
            h3 = tf.contrib.layers.flatten(h3)
            h4 = tf.layers.dense(h3, 500)
            h4 = tf.nn.relu(h4)
            h5 = tf.layers.dense(h4, 100)
            h5 = tf.nn.relu(h5)

            out = tf.layers.dense(h5, 1)

            return out

    def encod(self, x, reuse=False):
        with tf.variable_scope('auto_encoder',reuse=reuse):
            h1 = tf.layers.conv2d(x, 8, kernel_size=[1, 4], strides=[1, 2], padding='same', activation=tf.nn.relu)
            h1 = tf.layers.max_pooling2d(h1, (1, 2), (1, 2))

            h2 = tf.layers.conv2d(h1, 16, kernel_size=[1, 4], strides=[1, 2], padding='same', activation=tf.nn.relu)
            h2 = tf.layers.max_pooling2d(h2, (1, 2), (1, 2))


            h3 = tf.layers.conv2d(h2, 32, kernel_size=[1, 4], strides=[1, 2], padding='same', activation=tf.nn.relu)
            h3 = tf.layers.max_pooling2d(h3, (1, 3), (1, 2))

            return h3

    def decod(self, x):
        with tf.variable_scope('auto_encoder'):
            h1 = tf.layers.conv2d_transpose(inputs=x, filters=16, kernel_size=(1, 4), strides=(1, 3),
                                            padding='same', activation=tf.nn.relu)
            h2 = tf.layers.conv2d_transpose(inputs=h1, filters=8, kernel_size=(1, 4), strides=(1, 4),
                                            padding='same', activation=tf.nn.relu)
            h3 = tf.layers.conv2d_transpose(inputs=h2, filters=4, kernel_size=(1, 4), strides=(1, 4),
                                            padding='same', activation=tf.nn.relu)
            h4 = tf.layers.conv2d_transpose(inputs=h3, filters=1, kernel_size=(1, 4), strides=(1, 2),
                                            padding='same', activation=tf.nn.relu)
            return h4

    def train(self, data):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(self.init)
            train_writer = tf.summary.FileWriter('logs', sess.graph)

            step = 0
            for i in range(self.epoch):
                start = time.time()
                data.randomize()
                for batch_x, batch_y, batch_x_auto in data.next_batch(self.batch_size):

                    # Generate noise to feed to the generator
                    z_sample = np.random.uniform(0., 1., size=[batch_x.shape[0], self.num_nodes]).astype('float32')
                    z_label = np.ones_like(batch_y, dtype='float32')
                    # Train

                    if self.use_autoencoder:
                        # make input of the autoencoder noisy
                        batch_x_noisy = batch_x_auto + np.random.normal(loc=0.0, scale=1.0, size=batch_x.shape)

                        feed_dict = {self.disc_input: batch_x, self.gen_input: z_sample,
                                     self.disc_target: batch_y, self.gen_target: z_label,
                                     self.auto_input: batch_x_noisy}

                        _, _, _, g_loss, d_loss, a_loss, acc = sess.run([self.train_gen,
                                                                         self.train_disc,
                                                                         self.train_auto,
                                                                         self.gen_loss,
                                                                         self.disc_loss,
                                                                         self.auto_loss,
                                                                         self.accuracy,],
                                                                         feed_dict=feed_dict)
                    else:
                        feed_dict = {self.disc_input: batch_x, self.gen_input: z_sample,
                                     self.disc_target: batch_y, self.gen_target: z_label}

                        _, _, g_loss, d_loss, acc = sess.run([self.train_gen,
                                                                 self.train_disc,
                                                                 self.gen_loss,
                                                                 self.disc_loss,
                                                                 self.accuracy],
                                                                 feed_dict=feed_dict)

                    if step % 10 == 0:
                        if self.use_autoencoder:
                            gen_img, auto_img, summary_tr = sess.run([self.gen_out, self.new_represent, self.merged],
                                                                 feed_dict=feed_dict)
                        else:
                            gen_img, summary_tr = sess.run([self.gen_out, self.merged], feed_dict=feed_dict)

                        train_writer.add_summary(summary_tr, step)

                        print('Step {}: Generator Loss: {:.2f}, Discriminator Loss: {:.2f}, accuracy: {:.2f}'.format(step, g_loss, d_loss, acc*100))
                        if self.use_autoencoder:
                            print('Step {}: Auto-encoder Loss: {:.2f}'.format(step, a_loss))
                        print('-----------------------------------------------------------------------------')

                        fig = plt.figure()
                        for idx in range(self.num_nodes):
                            if self.use_autoencoder:
                                ax1 = fig.add_subplot(self.num_nodes, 4, 4*idx + 1)
                                ax2 = fig.add_subplot(self.num_nodes, 4, 4*idx + 2)
                                ax3 = fig.add_subplot(self.num_nodes, 4, 4*idx + 3)
                                ax4 = fig.add_subplot(self.num_nodes, 4, 4*idx + 4)

                                ax1.plot(np.arange(1, 97), np.squeeze(batch_x[0, idx, :, 0]))
                                ax2.plot(np.arange(1, 97), np.squeeze(gen_img[0, idx, :, 0]))
                                ax3.plot(np.arange(1, 97), np.squeeze(auto_img[0, idx, :, 0]))
                                ax4.plot(np.arange(1, 97), np.squeeze(batch_x_auto[0, idx, :, 0]))

                                if idx == 0:
                                    ax1.set_title('Real data')
                                    ax2.set_title('Generator out')
                                    ax3.set_title('Autoencoder out')
                                    ax4.set_title('Desired data')

                            else:
                                ax1 = fig.add_subplot(self.num_nodes, 2, 2 * idx + 1)
                                ax2 = fig.add_subplot(self.num_nodes, 2, 2 * idx + 2)

                                ax1.plot(np.arange(1, 97), np.squeeze(batch_x[0, idx, :, 0]))
                                ax2.plot(np.arange(1, 97), np.squeeze(gen_img[0, idx, :, 0]))

                                if idx == 0:
                                    ax1.set_title('real data')
                                    ax2.set_title('generator output')
                        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                            wspace=.7, hspace=.7)
                        fig.savefig(os.path.join('figs', '{}_{}.png'.format(i, step)), bbox_inches='tight')
                        plt.close()

                        # Dimension Reduction:
                        # TODO: create function
                        gen_in = np.random.uniform(0., 1., size=[1000, self.num_nodes]).astype('float32')
                        generator_out = sess.run(self.gen_out, feed_dict={self.gen_input: gen_in})
                        generator_out = np.reshape(generator_out, (generator_out.shape[0], -1))
                        # PCA
                        pca_out = decomposition.PCA(n_components=3).fit_transform(generator_out)

                        fig = plt.figure()
                        ax1 = fig.add_subplot(121, projection='3d')
                        ax1.scatter(pca_out[:, 0], pca_out[:, 1], pca_out[:, 2],
                                     c='r', marker='^')
                        ax1.set_title('PCA')

                        fig.savefig(os.path.join('figs_dim_red', '{}_{}.png'.format(i, step)), bbox_inches='tight')
                        plt.close()

                    step += 1
                end = time.time()

                # Final dimension reduction check
                # TODO: create function
                print('epoch {} took {:.1f} secs'.format(i, end - start))
                if not os.direxists(os.path.join(os.getcwd(), self.checkpoint_dir, self.model_name)):
                    os.makedirs(os.path.join(os.getcwd(), self.checkpoint_dir, self.model_name))
                saved_path = self.saver.save(sess, os.path.join(self.checkpoint_dir, self.model_name, 'model.ckpt'), global_step=step)
                print("model saved in path: {}".format(saved_path))

            # Dimension Reduction:
            gen_in = np.random.uniform(0., 1., size=[1000, self.num_nodes]).astype('float32')
            generator_out = sess.run(self.gen_out, feed_dict={self.gen_input: gen_in})
            generator_out = np.reshape(generator_out, (generator_out.shape[0], -1))
            # PCA
            pca_out = decomposition.PCA(n_components=3).fit_transform(generator_out)
            # TSNE
            tsne_out = TSNE(n_components=3).fit_transform(generator_out)

            fig = plt.figure()
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(pca_out[:, 0], pca_out[:, 1], pca_out[:, 2],
                        c='r', marker='^')
            ax1.set_title('PCA')

            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(tsne_out[:, 0], tsne_out[:, 1], tsne_out[:, 2],
                         c='b', marker='o')
            ax2.set_title('t-SNE')

            fig.savefig(os.path.join('figs_dim_red', '{}_{}.png'.format(i, step)), bbox_inches='tight')
            plt.close()

    def test(self, data):

        with tf.Session() as sess:
            test_writer = tf.summary.FileWriter('logs_test', sess.graph)
            self.saver.restore(sess,
                               tf.train.latest_checkpoint(os.path.join(self.checkpoint_dir,
                                                                       self.model_name)))

            x_test = np.concatenate((data.X, data.X_Auto), axis=0)
            y_test = np.zeros((x_test.shape[0], 1))

            z_sample = np.random.uniform(0., 1., size=[x_test.shape[0], self.num_nodes]).astype('float32')
            z_label = np.ones_like(y_test, dtype='float32')

            if self.use_autoencoder:
                feed_dict = {self.disc_input: x_test, self.gen_input: z_sample,
                             self.disc_target: y_test, self.gen_target: z_label,
                             self.auto_input: x_test}

                g_loss, d_loss, a_loss, acc = sess.run([self.gen_loss,
                                                        self.disc_loss,
                                                        self.auto_loss,
                                                        self.accuracy],
                                                        feed_dict=feed_dict)
            else:
                feed_dict = {self.disc_input: x_test, self.gen_input: z_sample,
                             self.disc_target: y_test, self.gen_target: z_label}

                g_loss, d_loss, acc = sess.run([self.gen_loss,
                                                self.disc_loss,
                                                self.accuracy],
                                                feed_dict=feed_dict)

            # for i in range(epoch_test):
                # x_test_batch=x_test[i:i+batch_test,self.num_nodes,self.num_features,0]
                # y_test_batch=y_test[i:i+batch_test,:]
                # z_sample = np.random.uniform(0., 1., size=[x_test[i:i+batch_test,3,:,0].shape[0], self.num_nodes]).astype('float32')
                # feed_dict = {self.disc_input: np.reshape(x_test_batch,[-1,3,self.num_features,1]), self.gen_input: z_sample,
                #              self.disc_target: np.reshape(np.ones((batch_test,1),'float32'),[-1,1])}
                # gen_img_test, d_loss_real_test, d_loss_fake_test, acc_test,summ_test = sess.run(
                #     [self.gen_out,self.disc_real_loss,self.dist_fake_loss,self.accuracy,self.merged], feed_dict=feed_dict)
                #
                # test_writer.add_summary(summ_test, i)
                # fig = plt.figure()
                # ax = fig.add_subplot(111)
                # ax.plot(np.arange(1, 257), np.squeeze(gen_img_test[0, :, :, 0]))
                # fig.savefig(os.path.join('figs_test', '{}.png'.format(i)), bbox_inches='tight')
                # print('Step {}: Disc Real Loss: {}, Discriminator Fake Loss: {}, accuracy: {}'.format(i, d_loss_real_test,d_loss_fake_test,acc_test))







