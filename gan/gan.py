"""" Generative adversarial network (GAN) training example
Implementation based on the paper https://arxiv.org/pdf/1406.2661v1.pdf
Code is loosely inspired by the code examples from : 
    https://github.com/AYLIEN/gan-intro
    https://github.com/ericjang/genadv_tutorial

Author: Moustafa Alzantot (m.alzantot@gmail.com)

"""

import numpy as np
from scipy.stats import norm
import tensorflow as tf

import matplotlib.pyplot as plt

class NoiseModel(object):
    def __init__(self, range):
        self.range = range

    def sample(self, n):
        return np.random.uniform(-self.range, self.range, size=(n, 1))


class DataModel(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self, n):
        samples = np.random.normal(loc=self.mu, scale=self.sigma, size=(n, 1))
        samples.sort()
        return samples

def linear(input_node, output_dim, scope=None):
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(name = 'w', shape = [input_node.get_shape()[1], output_dim], 
                initializer = tf.truncated_normal_initializer())
        b = tf.get_variable('b', [output_dim], initializer = tf.constant_initializer())
    return tf.matmul(input_node, w) + b

class GenModel(object):
    def __init__(self, z_holder):
        self.z_holder = z_holder
        self.g1 = tf.nn.tanh(linear(self.z_holder, 128, 'g1'))
        self.g2 = tf.nn.tanh(linear(self.g1, 128, 'g2'))
        self.g_out = tf.nn.softplus(linear(self.g2, 1, 'g_out'))

class DiscrimModel(object):
    def __init__(self, x_holder):
        self.x_holder = x_holder
        self.d1 = tf.nn.tanh(linear(self.x_holder, 128, 'd1'))
        self.d2 = tf.nn.tanh(linear(self.d1, 128, 'd2'))
        self.d_out = tf.nn.sigmoid(linear(self.d2, 1, 'd_out'))


class GAN(object):
    def __init__(self):
        self.noise_model = NoiseModel(4)
        self.data_model = DataModel(2.5, 0.5)
        self.batch_size = 1000
        self.anim_every = 100
        self.z_holder = tf.placeholder(tf.float32, [None, 1], "z")
        self.x_holder = tf.placeholder(tf.float32, [None, 1], "x")

        self.anim = PltAnimation(6)

        with tf.variable_scope("DPre"):
            self.d_pre_y_holder = tf.placeholder(tf.float32, [None, 1], "pre_y")
            self.d_pre_model = DiscrimModel(self.x_holder)
            self.d_pre = self.d_pre_model.d_out
           
            

        with tf.variable_scope('G'):
            self.gen_model = GenModel(self.z_holder)
            self.g_out = self.gen_model.g_out

        with tf.variable_scope('D') as scope :
            self.d_model = DiscrimModel(self.x_holder)
            self.d_x = self.d_model.d_out
            scope.reuse_variables()
            self.d_model2 = DiscrimModel(self.g_out)
            self.d_z = self.d_model2.d_out
        
        self.pre_loss = -tf.reduce_mean(self.d_pre_y_holder * tf.log(self.d_pre) + (1-self.d_pre_y_holder)*(tf.log(1-self.d_pre)))
        self.g_loss = tf.reduce_mean(1 - tf.log(self.d_z))
        self.d_loss = 1 - (tf.reduce_mean(tf.log(self.d_x)) + tf.reduce_mean(tf.log(1-self.d_z)))
        self.train_vars = tf.trainable_variables()
        self.pre_vars = [x for x in self.train_vars if x.name.startswith("DPre/")]
        self.d_vars = [x for x in self.train_vars if x.name.startswith("D/")]
        self.g_vars = [x for x in self.train_vars if x.name.startswith("G/")]
        self.pre_opt = tf.train.RMSPropOptimizer(0.0001).minimize(self.pre_loss, var_list=self.pre_vars)
        self.d_opt = tf.train.RMSPropOptimizer(0.00001).minimize(self.d_loss, var_list=self.d_vars)
        self.g_opt = tf.train.GradientDescentOptimizer(0.00001).minimize(self.g_loss, var_list=self.g_vars)
        self.init_op = tf.initialize_all_variables()
        self.session = tf.Session()
        self.session.run(self.init_op)


    def train(self, n_steps, pre_steps=500):
        ## Pre-train the discriminator
        for step_idx in range(pre_steps):
            xx = np.random.uniform(-6, 6, 1000).reshape((-1, 1))
            yy = norm.pdf(xx, loc=2.5, scale=0.5)
            pre_loss, _ = self.session.run([self.pre_loss, self.pre_opt], feed_dict={
                self.x_holder: xx,
                self.d_pre_y_holder: yy 
                })
            if step_idx % self.anim_every == 0:
                batch_true = self.data_model.sample(1000)
                batch_z = self.noise_model.sample(1000)
                batch_gen = self.session.run(self.g_out, feed_dict={self.z_holder: batch_z})
                db = self.session.run(self.d_pre, feed_dict={self.x_holder: np.linspace(-6,6, 1000).reshape((-1,1))})
                self.anim.step(batch_true, batch_gen, np.linspace(-6,6,1000).reshape((-1,1)), db)


        d_weights = self.session.run(self.pre_vars)
        for i, w in enumerate(d_weights):
            self.session.run(tf.assign(self.d_vars[i], w))
        for step_idx in range(n_steps):
            ## Train the discriminator
            batch_true = self.data_model.sample(self.batch_size)
            batch_z = self.noise_model.sample(self.batch_size)
            _, d_loss = self.session.run([self.d_opt, self.d_loss], feed_dict={self.x_holder: batch_true,
                self.z_holder: batch_z})
            print(d_loss)
            ## Update the generator
            batch_z = self.noise_model.sample(self.batch_size)
            self.session.run([self.g_opt], feed_dict={self.z_holder: batch_z})

            if step_idx % self.anim_every == 0:
                batch_true = self.data_model.sample(5000)
                batch_z = self.noise_model.sample(5000)
                batch_gen = self.session.run(self.g_out, feed_dict={self.z_holder: batch_z})
                db = self.session.run(self.d_model.d_out, feed_dict={self.x_holder: np.linspace(-6,6, 1000).reshape((-1,1))})
                self.anim.step(batch_true, batch_gen, np.linspace(-6,6,1000).reshape((-1,1)), db)


    def close(self):
        self.session.close()

class PltAnimation(object):
    def __init__(self, range):
        self.range = range
        self.num_bins = 200
        self.figure = plt.figure()
        self.ax = plt.subplot(111)
        self.ax.set_ylim(-0.1, 1.1)

        self.x = np.linspace(-self.range, self.range, self.num_bins)
        self.l, self.l2, self.l3 = None, None, None
        plt.show(block=False)

    def step(self, x_true, x_gen, db_x=None, db_y=None):
        p_true, p_edges = np.histogram(x_true, bins=self.x, density=True)
        p_gen, g_edges = np.histogram(x_gen, bins=self.x, density=True)
        if (self.l == None):
            self.l, = self.ax.plot(self.x[:-1], p_true)
            self.l2, = self.ax.plot(self.x[:-1], p_gen)
        else:
            self.l.set_ydata(p_true)
            self.l2.set_ydata(p_gen)

        if not (db_x == None):
            if self.l3 == None:
                self.l3, = self.ax.plot(db_x, db_y)
            else:
                self.l3.set_data((db_x, db_y))
        plt.pause(0.0001)



if __name__ == '__main__':
    seed = 1234
    np.random.seed(seed)
    tf.set_random_seed(seed)


    model = GAN()
    model.train(10000, 250)
    model.close()
    input("Press any key to exit..")
