import tensorflow as tf
import numpy as np
import math
import random


neural_net_params = {
    'lr': 1e-3,
    'dropout': 0.5,
    'batch_size': 32,
    'num_channels': 256,
    'epochs': 20
}

class ConvNet:
    def __init__(self, game):
        # Parameters needed for the network architecture
        self.game = game
        self.xdim, self.ydim = game.getBoardSize()

        # General Network Parameters.
        self.args = neural_net_params

        # Make naming more simple
        Relu = tf.nn.relu
        Tanh = tf.nn.tanh
        Softmax = tf.nn.softmax
        BatchNorm = tf.layers.batch_normalization
        Dropout = tf.layers.dropout
        Dense = tf.layers.dense
        Conv2d = tf.layers.conv2d

        # Initialize the graph
        self.graph = tf.Graph()
        self.saver = None
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, shape = [None, self.xdim, self.ydim])
            self.dropout = tf.placeholder(tf.float32, name = "dropout")
            self.is_training = tf.placeholder(tf.bool, name = "training_status")

            # Internal layers
            start = tf.reshape(self.x, [-1, self.xdim, self.ydim, 1])
            conv1 = Relu(BatchNorm(Conv2d(start, self.args['num_channels'], kernel_size = [3,3], padding = 'same', use_bias = False), axis = 3, training = self.is_training))
            conv2 = Relu(BatchNorm(Conv2d(conv1, self.args['num_channels'], kernel_size = [3,3], padding = 'same', use_bias = False), axis = 3, training = self.is_training))
            conv3 = Relu(BatchNorm(Conv2d(conv1, self.args['num_channels'], kernel_size = [3,3], padding = 'same', use_bias = False), axis = 3, training = self.is_training))
            conv4 = Relu(BatchNorm(Conv2d(conv1, self.args['num_channels'], kernel_size = [3,3], padding = 'same', use_bias = False), axis = 3, training = self.is_training))
            conv5 = Relu(BatchNorm(Conv2d(conv1, self.args['num_channels'], kernel_size = [3,3], padding = 'same', use_bias = False), axis = 3, training = self.is_training))
            conv6 = Relu(BatchNorm(Conv2d(conv1, self.args['num_channels'], kernel_size = [3,3], padding = 'same', use_bias = False), axis = 3, training = self.is_training))
            conv7 = Relu(BatchNorm(Conv2d(conv2, self.args['num_channels'], kernel_size = [3,3], padding = 'valid', use_bias = False), axis = 3, training = self.is_training))
            conv8 = Relu(BatchNorm(Conv2d(conv3, self.args['num_channels'], kernel_size = [3,3], padding = 'valid', use_bias = False), axis = 3, training = self.is_training))
            flattened = tf.reshape(conv4, [-1, self.args['num_channels']*(self.xdim-4)*(self.ydim-4)])

            # Now we split into the value and policy parts of the neural network
            fc1 = Dropout(Relu(BatchNorm(Dense(flattened, 1024), axis=1, training=self.is_training)), rate=self.dropout)
            fc2 = Dropout(Relu(BatchNorm(Dense(fc1, 512), axis = 1, training = self.is_training)), rate = self.dropout)
            policy = Dropout(Relu(BatchNorm(Dense(fc2, self.game.getActionSize()), axis = 1, training = self.is_training)), rate = self.dropout)
            self.pi = Softmax(policy)
            self.v = Tanh(Dense(fc2, 1))

            # Compute the loss
            self.target_pis = tf.placeholder(tf.float32, shape=[None, game.getActionSize()])
            self.target_vs = tf.placeholder(tf.float32, shape=[None])
            loss_pi =  tf.losses.softmax_cross_entropy(self.target_pis, self.pi)
            loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1,]))
            self.total_loss = loss_pi + loss_v


            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            # make training step
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.args['lr']).minimize(self.total_loss)

            # get session
            self.session = tf.Session(graph = self.graph)

            # initialize
            with tf.Session() as temp_sess:
                temp_sess.run(tf.global_variables_initializer())
            self.session.run(tf.variables_initializer(self.graph.get_collection('variables')))

    def predict(self, board):
        board = board[np.newaxis, :, :]
        prob, v =  self.session.run([self.pi, self.v], feed_dict = {self.x: board, self.dropout: 0, self.is_training: False})
        return prob[0], v[0]

    def train(self, data):
        # Train for requested number of epochs
        for epoch in range(self.args['epochs']):
            print("Epoch # " + str(epoch))
            for i in range(len(data)/self.args['batch_size']):
                indexes = np.random.randint(int(len(examples)/self.args['batch_size']))
                boards, pis, vs = list(zip(*data) for i in indexes)
                feed_dict = {self.x: boards, self.target_pis: pis, self.target_vs: vs, self.dropout: self.args['dropout'], self.isTraining: True}
                self.sess.run(self.train_step, feed_dict = feed_dict)

    def save(self, filename):
        if self.saver == None:
            self.saver = tf.train.Saver(self.graph.get_collection('variables'))
        self.saver.save(self.session, "/saves/" + filename)

    def load(self, filename):
        self.saver = tf.train.Saver()
        self.saver.restore(self.session, "/saves/" + filename)
