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

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)

        self.initializer = tf.truncated_normal_initializer(0,0.02)
        self.activation = tf.nn.relu
        self.batch_accumulator = batch_accumulator
        self.update_rule = update_rule

        self.S = tf.placeholder(tf.float32,[None, self.input_width], name='s')
        self.S_ = tf.placeholder(tf.float32, [None, self.input_width], name='s_')
        self.R = tf.placeholder(tf.float32,[None, 1], name='r')
        self.A = tf.placeholder(tf.int32, [None, 1], name='a')
        self.T = tf.placeholder(tf.float32, [None, 1], name='t')
        self.distributional = True
        self.dueling = False

        if self.distributional:
            self.Vmin = -100.0
            self.Vmax = 100.0
            self.atoms = 11
            self.delta_z = float(self.Vmax-self.Vmin)/(self.atoms-1)
            self.Z = np.linspace(self.Vmin, self.Vmax, num=self.atoms, endpoint=True)
            self.p_target = tf.placeholder(tf.float32, [None, self.num_actions, self.atoms], name='P_target')
        else:
            self.q_target =  tf.placeholder(tf.float32, [None, self.num_actions], name='Q_target')

        self.update_counter = 0

        if self.distributional:
            with tf.variable_scope('eval'):
                state = self.S
                self.p_vals = self.build_rl_network_dnn(input_width, input_height,
                                        num_actions*self.atoms, num_frames, batch_size, state/input_scale,trainable = True)
            with tf.variable_scope('target'):
                state = self.S_
                self.next_p_vals = self.build_rl_network_dnn(input_width, input_height,
                                        num_actions*self.atoms, num_frames, batch_size, state/input_scale,trainable = False)

            loss = -tf.reduce_sum(tf.multiply(self.p_target, tf.log(tf.clip_by_value(self.p_vals, 1e-10, 1.0))),axis=2)

        else:
            with tf.variable_scope('eval'):
                state = self.S
                self.q_vals = self.build_rl_network_dnn(input_width, input_height,
                                        num_actions, num_frames, batch_size, state/input_scale,trainable = True)
            with tf.variable_scope('target'):
                state = self.S_
                self.next_q_vals = self.build_rl_network_dnn(input_width, input_height,
                                        num_actions, num_frames, batch_size, state/input_scale,trainable = False)

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
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')

        if update_rule == 'adam':
            self._train = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=0.01/self.batch_size).minimize(self.loss, var_list=self.e_params)
        elif update_rule == 'RMS_prop':
            self._train = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss, var_list=self.e_params)
        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))

        self.sess.run(tf.global_variables_initializer())
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        if self.freeze_interval > 0:
            self.reset_q_hat()

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
        states = states.reshape(-1, self.input_width)
        next_states = next_states.reshape(-1, self.input_width)
        update_rule = self.update_rule
        terminals = terminals.astype(np.float32)
        if (self.freeze_interval > 0 and
            self.update_counter % self.freeze_interval == 0):
            self.reset_q_hat()
        p_vals, next_p_vals = self.sess.run([self.p_vals, self.next_p_vals],{self.S: states, self.A: actions, self.R: rewards, self.S_: next_states, self.T: terminals})

        p_target = p_vals.copy()
        BestAct = np.argmax(np.sum(np.multiply(self.Z, next_p_vals),axis =2), axis = 1)
        terminals_false = np.ones_like(terminals)-terminals

        for batchIndex, r in enumerate(rewards):
            m = np.zeros(self.atoms)
            for j in range(self.atoms):
                Tau_z_j = np.clip(r + terminals_false[batchIndex][0]*self.discount * next_q_vals[batchIndex][BestAct[batchIndex]][j], self.Vmin, self.Vmax)
                b_j = float(Tau_z_j-self.Vmin)/self.delta_z
                l = math.floor(b_j)
                u = math.ceil(b_j)
                l_int = int(l)
                u_int = int(u)
                m[l_int] = m[l_int] + next_p_vals[batchIndex][BestAct[batchIndex]][j]*(u - b_j)
                m[u_int] = m[u_int] + next_p_vals[batchIndex][BestAct[batchIndex]][j]*(b_j - l)
            p_target[batchIndex,BestAct[batchIndex],:] = m

        _,loss = self.sess.run([self._train,self.loss], feed_dict={self.S: states, self.p_target: p_target})
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

        _,loss = self.sess.run([self._train,self.loss], feed_dict={self.S: states, self.q_target: q_target})
        self.update_counter += 1
        return np.sqrt(loss)

    def get_q_vals(self, state):
        states = state.reshape((1,self.input_width))
        q_vals = self.sess.run([self.q_vals],{self.S: states})
        return q_vals

    def get_q_vals_distributional(self, state):
        states = state.reshape((1,self.input_width))
        p_vals = self.sess.run([self.p_vals],{self.S: states})
        return p_vals

    def choose_action_distributional(self, state, epsilon):
        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.num_actions)
        p_vals = self.get_q_vals_distributional(state)
        
        return np.argmax(np.sum(np.multiply(self.Z, p_vals[0][0]),axis = 1))

    def choose_action(self, state, epsilon):
        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.num_actions)
        q_vals = self.get_q_vals(state)
        return np.argmax(q_vals)

    def reset_q_hat(self):
        self.sess.run(self.replace_target_op)

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
        if self.dueling:
            V_out = tf.layers.dense(
                layer,
                units=output_dim,
                kernel_initializer=tf.truncated_normal_initializer(stddev=var),
                bias_initializer=tf.constant_initializer(bias),
                trainable = trainable
            )
            A_out = tf.layers.dense(
                layer,
                units=output_dim,
                kernel_initializer=tf.truncated_normal_initializer(stddev=var),
                bias_initializer=tf.constant_initializer(bias),
                trainable = trainable
            )
            # Q = V(s) + A(s,a)
            out = V_out + (A_out- tf.reduce_mean(A_out, axis=1, keep_dims=True))
        else:
            out = tf.layers.dense(
                layer,
                units=output_dim,
                kernel_initializer=tf.truncated_normal_initializer(stddev=var),
                bias_initializer=tf.constant_initializer(bias),
                trainable = trainable
            )

        if self.distributional:
            out = tf.reshape(out, shape=[-1, self.num_actions, self.atoms])
            out = tf.nn.softmax(out)

        return out

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
