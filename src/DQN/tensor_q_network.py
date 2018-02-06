import numpy as np
import tensorflow as tf
import math

class DeepQLearner:
    """
    Deep Q-learning network using Lasagne.
    """

    def __init__(self, input_width, input_height, net_width, net_height, num_actions,
                 num_frames, discount, learning_rate, rho,
                 rms_epsilon, momentum, nesterov_momentum, clip_delta, freeze_interval,
                 batch_size, network_type, update_rule,
                 batch_accumulator, rng, input_scale=255.0):
        assert not ( momentum and nesterov_momentum ), "Choose only 1 momentum method!"

        self.input_width = input_width
        self.input_height = input_height
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.discount = discount
        self.rho = rho
        self.lr = learning_rate
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        self.nesterov_momentum = nesterov_momentum
        self.clip_delta = clip_delta
        self.freeze_interval = freeze_interval
        self.rng = rng
        self.network_width = net_width
        self.network_height = net_height
        self.sess = tf.Session()
        self.initializer = tf.truncated_normal_initializer(0,0.02)
        self.activation = tf.nn.relu
        self.batch_accumulator = batch_accumulator
        self.update_rule = update_rule
        self.S = tf.placeholder(tf.float32,[None, self.input_width], 's')
        self.S_ = tf.placeholder(tf.float32, [None, self.input_width], 's_')
        self.R = tf.placeholder(tf.float32,[None, 1],'r')
        self.A = tf.placeholder(tf.int32, [None, 1], 'a')
        self.T = tf.placeholder(tf.float32, [None, 1], 't')
        self.distributional = False

        if self.distributional:
            self.Vmin = -5
            self.Vmax = 5
            self.atoms = 51
            self.delta_z = float(self.Vmax-self.Vmin)/(self.atoms-1)
            self.Prob_i = tf.placeholder(tf.float32, [None, self.atoms], name='probability_function')
            self.q_target = tf.placeholder(tf.float32, [None, self.num_actions, self.atoms], name='Q_target')
        else:
            self.q_target =  tf.placeholder(tf.float32, [None, self.num_actions], name='Q_target')
        
        self.update_counter = 0
        with tf.variable_scope('eval'):
            state = self.S 
            if self.distributional:
                self.q_vals = self.build_rl_network_dnn(input_width, input_height,
                                        num_actions*self.atoms, num_frames, batch_size, state/input_scale,trainable = True)
            else:
                self.q_vals = self.build_rl_network_dnn(input_width, input_height,
                                        num_actions, num_frames, batch_size, state/input_scale,trainable = True)
        
        
        with tf.variable_scope('target'):
            state = self.S_
            if self.distributional:
                self.next_q_vals = self.build_rl_network_dnn(input_width,
                                                 input_height, num_actions*self.atoms,
                                                 num_frames, batch_size, state/input_scale,trainable = False)
            else:
                self.next_q_vals = self.build_rl_network_dnn(input_width,
                                                 input_height, num_actions,
                                                 num_frames, batch_size, state/input_scale,trainable = False)
       
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')
        
        if self.distributional:
            diff = -tf.reduce_sum(tf.multiply(self.q_target, self.q_vals), axis=2)
        else:
            diff = self.q_target - self.q_vals

        if self.clip_delta > 0:
            # If we simply take the squared clipped diff as our loss,
            # then the gradient will be zero whenever the diff exceeds
            # the clip bounds. To avoid this, we extend the loss
            # linearly past the clip point to keep the gradient constant
            # in that regime.
            #
            # This is equivalent to declaring d loss/d q_vals to be
            # equal to the clipped diff, then backpropagating from
            # there, which is what the DeepMind implementation does.
            quadratic_part = tf.minimum(abs(diff), self.clip_delta)
            linear_part = abs(diff) - quadratic_part
            loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        else:
            loss = 0.5 * diff ** 2

        if self.batch_accumulator == 'sum':
            loss = tf.reduce_sum(loss)
        elif self.batch_accumulator == 'mean':
            loss = tf.reduce_mean(loss)
        else:
            raise ValueError("Bad accumulator: {}".format(batch_accumulator))
        self.loss = loss


        if update_rule == 'deepmind_rmsprop':
            updates = deepmind_rmsprop(loss, params, self.lr, self.rho,
                                       self.rms_epsilon)
        elif update_rule == 'rmsprop':
            updates = lasagne.updates.rmsprop(loss, params, self.lr, self.rho,
                                              self.rms_epsilon)
        elif update_rule == 'adagrad':
		updates = lasagne.updates.adagrad(loss, params, self.lr,
							self.rms_epsilon)
        elif update_rule == 'adadelta':
		updates = lasagne.updates.adadelta(loss, params, self.lr, self.rho,
						self.rms_epsilon)
        elif update_rule == 'sgd':
            updates = lasagne.updates.sgd(loss, params, self.lr)
        
        elif update_rule == 'adam':
            self._train = tf.train.AdamOptimizer(self.lr).minimize(loss,var_list=self.e_params)
        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))
        
        self.sess.run(tf.global_variables_initializer())
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        if self.freeze_interval > 0:
            self.reset_q_hat()
        
    def build_network(self, network_type, input_width, input_height,
                      output_dim, num_frames, batch_size,trainable=True):
        if network_type == "nature_cuda":
            return self.build_nature_network(input_width, input_height,
                                             output_dim, num_frames, batch_size)
        if network_type == "nature_dnn":
            return self.build_nature_network_dnn(input_width, input_height,
                                                 output_dim, num_frames,
                                                 batch_size)
        elif network_type == "nips_cuda":
            return self.build_nips_network(input_width, input_height,
                                           output_dim, num_frames, batch_size)
        elif network_type == "nips_dnn":
            return self.build_nips_network_dnn(input_width, input_height,
                                               output_dim, num_frames,
                                               batch_size)
        elif network_type == "rl_dnn":
            return self.build_rl_network_dnn(input_width, input_height,
                                                output_dim, num_frames,
                                                batch_size,trainable)
        elif network_type == "linear":
            return self.build_linear_network(input_width, input_height,
                                             output_dim, num_frames, batch_size)
        else:
            raise ValueError("Unrecognized network: {}".format(network_type))

    def CA_Algorithm(self, states, actions, rewards, next_states, terminals):
        """
        Distributional Algorithm Implementation
        Categorical Algorithm
        
        Train one batch.
        Arguments:

        states - b x f x h x w numpy array, where b is batch size,
                 f is num frames, h is height and w is width.
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        next_states - b x f x h x w numpy array
        terminals - b x 1 numpy boolean array

        Returns: average loss
        """
        def BondReward(reward):
            if reward > self.Vmax:
                return self.Vmax
            elif reward < self.Vmin:
                return self.Vmin
            else:
                return reward

        def softmax(logits):
            e_x  = np.exp(logits)
            return e_x/np.sum(e_x)

        states = states.reshape(-1, self.input_width)
        next_states = next_states.reshape(-1, self.input_width)
        update_rule = self.update_rule
        terminals = terminals.astype(np.float32) 
        if (self.freeze_interval > 0 and
            self.update_counter % self.freeze_interval == 0):
            self.reset_q_hat()
        q_vals, next_q_vals = self.sess.run([self.q_vals, self.next_q_vals],{self.S: states, self.A: actions, self.R: rewards, self.S_: next_states, self.T: terminals})

        q_target = q_vals.copy()
        for batchIndex, r in enumerate(rewards):
            BestActIndex = 0
            BestQSum = -99999
            for actIndex in range(self.num_actions):
                # if dot cannot work, try multiply -> sum
                Q_sum = np.dot(next_q_vals[batchIndex][actIndex], softmax(q_vals[batchIndex][actIndex]))
                if Q_sum > BestQSum:
                    BestQSum = Q_sum
                    BestActIndex = actIndex
            
            p_next_astar = softmax(next_q_vals[batchIndex][actIndex])

            m = np.zeros(self.atoms)
            for j in range(self.atoms):
                if terminals[batchIndex][0]:
                    Tau_z_j = BondReward(r)
                else:
                    Tau_z_j = BondReward(r + self.discount * next_q_vals[batchIndex][actIndex][j])
                b_j = float(Tau_z_j-self.Vmin)/self.delta_z
                l = math.floor(b_j)
                u = math.ceil(b_j)
                l_int = int(l)
                u_int = int(u)
                m[l_int] = m[l_int] + p_next_astar[j]*(u - b_j)
                m[u_int] = m[u_int] + p_next_astar[j]*(b_j - l)
                print("Tau_z_j", Tau_z_j)
            q_target[batchIndex,BestActIndex,:] = m

        _,loss = self.sess.run([self._train,self.loss], feed_dict={self.S: states, self.q_target: q_target})
        self.update_counter += 1
        return loss


    def train(self, states, actions, rewards, next_states, terminals):
        """
        Train one batch.

        Arguments:

        states - b x f x h x w numpy array, where b is batch size,
                 f is num frames, h is height and w is width.
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        next_states - b x f x h x w numpy array
        terminals - b x 1 numpy boolean array

        Returns: average loss
        """
        states = states.reshape(-1, self.input_width)
        next_states = next_states.reshape(-1, self.input_width)
        update_rule = self.update_rule
        terminals = terminals.astype(np.float32) 
        if (self.freeze_interval > 0 and
            self.update_counter % self.freeze_interval == 0):
            self.reset_q_hat()
        q_vals, next_q_vals = self.sess.run([self.q_vals, self.next_q_vals],{self.S: states, self.A: actions, self.R: rewards, self.S_: next_states, self.T: terminals})
        q_target = q_vals.copy()

        term = (np.ones_like(terminals) - terminals)
        next_wq = np.multiply(term, np.max(next_q_vals, axis=1))
        for i,r in enumerate(rewards):
            q_target[i, actions[i][0]] = r + self.discount *next_wq[i][0]

        print(q_target)
        _,loss = self.sess.run([self._train,self.loss], feed_dict={self.S: states, self.q_target: q_target})
        self.update_counter += 1
        return np.sqrt(loss)

    def get_q_vals(self, state):
        states = state.reshape((1,self.input_width))
        q_vals = self.sess.run([self.q_vals],{self.S: states})
        return q_vals

    def get_q_vals_distributional(self, state):
        states = state.reshape((1,self.input_width))
        q_vals = self.sess.run([self.q_vals],{self.S: states})        
        return q_vals

    def choose_action_distributional(self, state, epsilon):
        def softmax(logits):
            e_x  = np.exp(logits)
            return e_x/np.sum(e_x)

        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.num_actions)
        q_vals = self.get_q_vals_distributional(state)

        BestActIndex = 0
        BestQSum = -99999
        for actIndex in range(self.num_actions):
            Q_sum = np.dot(q_vals[0][0][actIndex], softmax(q_vals[0][0][actIndex]))
            if Q_sum > BestQSum:
                BestQSum = Q_sum
                BestActIndex = actIndex
        return BestActIndex

    def choose_action(self, state, epsilon):
        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.num_actions)
        q_vals = self.get_q_vals(state)
        return np.argmax(q_vals)

    def reset_q_hat(self):
        self.sess.run(self.replace_target_op)

    def build_nature_network(self, input_width, input_height, output_dim,
                             num_frames, batch_size):
        """
        Build a large network consistent with the DeepMind Nature paper.
        """
        from lasagne.layers import cuda_convnet

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, num_frames, input_width, input_height)
        )

        l_conv1 = cuda_convnet.Conv2DCCLayer(
            l_in,
            num_filters=32,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(), # Defaults to Glorot
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_conv2 = cuda_convnet.Conv2DCCLayer(
            l_conv1,
            num_filters=64,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_conv3 = cuda_convnet.Conv2DCCLayer(
            l_conv2,
            num_filters=64,
            filter_size=(3, 3),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv3,
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        return l_out

    def build_nature_network_dnn(self, input_width, input_height, output_dim,
                                 num_frames, batch_size):
        """
        Build a large network consistent with the DeepMind Nature paper.
        """
        from lasagne.layers import dnn

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, num_frames, input_width, input_height)
        )

        l_conv1 = dnn.Conv2DDNNLayer(
            l_in,
            num_filters=32,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_conv2 = dnn.Conv2DDNNLayer(
            l_conv1,
            num_filters=64,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_conv3 = dnn.Conv2DDNNLayer(
            l_conv2,
            num_filters=64,
            filter_size=(3, 3),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv3,
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        return l_out

    def build_rl_network_dnn(self, input_width, input_height, output_dim,
                            num_frames, batch_size, layer, trainable):

        var = 0.01
        bias = 0.1
 
        for _ in xrange(self.network_height):
            layer = tf.layers.dense(
            layer,
            units=self.network_width,
            activation = self.activation,
            kernel_initializer=tf.truncated_normal_initializer(stddev=var),
            bias_initializer=tf.constant_initializer(bias),
            trainable = trainable
        )

        l_out = tf.layers.dense(
            layer,
            units=output_dim,
            #activation = self.activation,
            kernel_initializer=tf.truncated_normal_initializer(stddev=var),
            bias_initializer=tf.constant_initializer(bias),
            trainable = trainable
        )
        if self.distributional:
            l_out = tf.reshape(l_out, shape=[-1, self.num_actions, self.atoms])

        return l_out

    def build_linear_network(self, input_width, input_height, output_dim,
                             num_frames, batch_size):
        """
        Build a simple linear learner.  Useful for creating
        tests that sanity-check the weight update code.
        """

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, num_frames, input_width, input_height)
        )

        l_out = lasagne.layers.DenseLayer(
            l_in,
            num_units=output_dim,
            nonlinearity=None,
#            W=lasagne.init.Constant(0.0),
            W = lasagne.init.Normal(0.01,0),
            b=None
        )

        return l_out

def main():
    input_width, input_height = [100,100]
    num_actions = 10
    phi_length = 4 # phi length?  input 4 frames at once
    discount = 0.95
    learning_rate = 0.00025
    rms_decay = 0.99 # rms decay
    rms_epsilon = 0.1
    momentum = 0
    clip_delta = 1.0
    freeze_interval = 10000 #???  no freeze?
    batch_size = 32
    network_type = 'rl_dnn'
    update_rule = 'deepmind_rmsprop' # need update
    batch_accumulator = 'sum'
    rng = np.random.RandomState()
    ###### AGENT ######
    epsilon_start = 1.0
    epsilon_min = 0.1
    epsilon_decay = 1000000
    replay_memory_size = 1000000
    experiment_prefix = 'result/LF2'
    replay_start_size = 50000
    update_frequency = 4  #??
    ######
    num_epoch = 200
    step_per_epoch = 250000
    step_per_test = 125000

    network = DeepQLearner(input_width, input_height, num_actions,
                                           phi_length,
                                           discount,
                                           learning_rate,
                                           rms_decay,
                                           rms_epsilon,
                                           momentum,
                                           clip_delta,
                                           freeze_interval,
                                           batch_size,
                                           network_type,
                                           update_rule,
                                           batch_accumulator,
                                           rng)


if __name__ == '__main__':
    main()
