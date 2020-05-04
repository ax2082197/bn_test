import tensorflow as tf
import numpy as np 

class classification_model_32:
    def __init__(self, name):
        self.input = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.output = tf.placeholder(tf.float32, [None, 2])
        self.keep_rate = tf.placeholder(tf.float32, [])
        self.is_training = tf.placeholder(tf.bool, [])
        self.c = tf.placeholder(tf.float32, [])
        self.name = name
        self.build_model()

    def conv_block(self, x, name, kernel_width, kernel_height, inp_channel, out_channel, strides = [1, 1, 1, 1], padding='SAME'):

        conv_layer = tf.layers.conv2d( x, out_channel, 3, (1,1), padding='same', use_bias=False, activation=None)
        conv_layer = tf.layers.batch_normalization(conv_layer, training=self.is_training)
        return tf.nn.leaky_relu(conv_layer)

    def max_pool_2x2(self, x):
        return tf.nn.max_pool( x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    def dense_with_dropout(self, x, size, activation = tf.math.sigmoid):
        hidden = tf.layers.dense( x, size, activation = activation)
        with tf.variable_scope("dropout"):
            hidden = tf.layers.dropout(hidden, 1 - self.keep_rate)
        return hidden

    def build_model(self):
        with tf.variable_scope(self.name):
            conv1_1 = self.conv_block( self.input, 'conv1_1', 3, 3,   3,  32)
            conv1_2 = self.conv_block(    conv1_1, 'conv1_2', 3, 3,  32,  32)
            pool1 = self.max_pool_2x2( conv1_2)

            conv2_1 = self.conv_block(      pool1, 'conv2_1', 3, 3,  32,  64)
            conv2_2 = self.conv_block(    conv2_1, 'conv2_2', 3, 3,  64,  64)
            pool2 = self.max_pool_2x2( conv2_2)
            
            conv3_1 = self.conv_block(      pool2, 'conv3_1', 3, 3,  64, 128)
            conv3_2 = self.conv_block(    conv3_1, 'conv3_2', 3, 3, 128, 128)
            conv3_3 = self.conv_block(    conv3_2, 'conv3_3', 3, 3, 128, 128)
            conv3_4 = self.conv_block(    conv3_3, 'conv3_4', 3, 3, 128, 128)
            pool3 = self.max_pool_2x2( conv3_4)

            flatten = tf.reshape(pool3, [ -1, 2048])
            hidden1 = self.dense_with_dropout( flatten, 1024)
            hidden3 = self.dense_with_dropout( hidden1,   64)
            self.predict = self.dense_with_dropout( hidden3, 2, activation = tf.nn.softmax)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2( self.output, self.predict))
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_step = tf.train.AdamOptimizer(0.0001).minimize( self.loss, name='adam_train')

