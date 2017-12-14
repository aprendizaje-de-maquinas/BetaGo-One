import gym
import numpy as np
import tensorflow as tf
import os
import time
from xmcts import MCTS
from collections import defaultdict
import pachi_py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # prevent the logging of bs

from network import network

BOARD_SIZE           = 9
epochs               = 1 #10000
num_self_play_games  = -1 #10
num_test_games       = 100000 #10
MCTS_DEPTH           = 7
MCTS_NUM_SIMULATIONS = 160
RESIGN_THRESH        = 10
T_THRESH             = 30
TEST_INTERVAL        = 0 #10
RESET_INTERVAL       = 2

MODEL_NUMBER  =  130


class DoubleDeepQNN:
    def __init__(  self, build_layers , n_actions, input_shape , learning_rate=0.01, gamma=1.0,
                   replace_target_iter=60, batch_size=32, sess=None ):

        self.sess = tf.Session() if sess is None else sess # make sure we have a session
        self.sess.run(tf.global_variables_initializer()) # init

        self.prev_diff = 2400

        self.input_shape = input_shape 

        self.n_actions = n_actions # the size of the action space

        self.build_layers = build_layers

        self.lr = learning_rate
        self.gamma = gamma

        self.memory_counter = 0 # counts the size of memory

        self.memory = defaultdict( float )
        self.batch_size = batch_size # batch size for each training iteration (ie the minimize)
        
        self._build_net() # builds the NN
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)] # to swap the target and evaluation nets.
        self.replace_target_iter = replace_target_iter # the number of iterations between swapping the nets
        self.learn_step_counter = 0 # counter for determing when to swap


    def _build_net(self):
        # for the evaluation network
        self.s = tf.placeholder(tf.float32, self.input_shape , name='s')  # the current state
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
        

        with tf.variable_scope('eval_net'):
            c_names, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], tf.random_normal_initializer(0., .003),\
                tf.random_normal_initializer(0 , .003)  # config of layers

            self.q_eval = self.build_layers(self.s, c_names, w_initializer, b_initializer , num_output=BOARD_SIZE**2+1)

            
        # since we swap which set of weights is being trained every so often, we only need one loss
        # and train operation

        # for the target network
        self.s_ = tf.placeholder(tf.float32, self.input_shape , name='s_')    # the next state
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = self.build_layers(self.s_, c_names, w_initializer, b_initializer , num_output=BOARD_SIZE**2+1)


        reg = []
        for i in tf.get_collection( 'eval_net_params' ):
            reg.append( tf.nn.l2_loss(i) )
        with tf.variable_scope('loss'):
            self.loss = tf.scalar_mul( 64 , tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval)) ) + \
                        0.0001 * tf.reduce_sum( reg )
            tf.summary.scalar( 'loss' , self.loss )
        # see implied loss function in the notes plus cross entropy plus regularization
        with tf.variable_scope('train'):
            self._train_op = tf.train.MomentumOptimizer(self.lr , 0.9).minimize(self.loss) # use a built in minimizer.


        self.elo = tf.placeholder( tf.int32 , name = 'ELO-difference')
        tf.summary.scalar( 'elo-difference' , self.elo )
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter( 'training' , self.sess.graph )
        self.saver = tf.train.Saver()


    # BEGIN MEMORY MANAGEMENT
    #***********************

    
    # save the s , a , r , s' into memory
    def store_transition(self, s, a, r, s_):

        self.memory[ self.memory_counter] = { 's': s , 'a': a , 'r': r , 's_': s_ }
        self.memory_counter += 1
    # clear the memory so that we can start from scratch
    def clear_transitions( self ):

        self.memory = defaultdict( float )
        self.memory_counter = 0


    # ************************
    # END MEMORY MANAGEMENT

    def learn(self , epoch , batches):

        # *********************************************************
        # can i put these into a queue runner for prefetching??????
        # *********************************************************

        for b in range( batches ): 
            if self.learn_step_counter % self.replace_target_iter == 0:
                self.sess.run(self.replace_target_op)

            mx = self.memory_counter # make sure rand knows what it can go up to
            sample_index = np.random.choice( mx , size=self.batch_size ) # random batch

            s, a, r, s_ = [] , [] , [] , []
            for x in sample_index:
                s.append( self.memory[ x ]['s'] ) 
                a.append( self.memory[ x ]['a'] ) 
                r.append( self.memory[ x ]['r'] ) 
                s_.append( self.memory[ x ]['s_'] ) 

            assert len( s ) == self.batch_size , 'fail'

            # compute Q(S_{t+1} , a , theta_t' ), Q(S_{t+1} , a , theta_t )  at same op for all a 
            q_next, q_eval_next = self.sess.run( [self.q_next, self.q_eval],
                feed_dict={self.s_: s_,    # s' from the mem used for target
                           self.s: s_})    # s' from the mem used for eval

            # calculate Q(S_t , a ; theta_t )
            q_eval = self.sess.run(self.q_eval, {self.s: s}) # evaluate this state

            q_target = q_eval.copy()

            # perform the argmax in the double q formula
            max_action = np.argmax(q_eval_next, axis=1)

            # get the right q_next for each member of batch from the precomputed Q(S_{t+1} , a , theta_t' ) for all a
            # ie we can just use max_act_next to index
            # note that we have to use batch_index as index as well because of making the shapes line up
            indexes = np.arange(self.batch_size, dtype=np.int32)
            selected_q_next = q_next[indexes, max_action]
            
            # calc the reward+ gamma ( target )  term
            q_target[indexes, a] = r + self.gamma * selected_q_next

            #perform the gradient ascent update ( or whatever optimizer being used ) over the loss

            _, summ = self.sess.run( [self._train_op , self.merged] , feed_dict={self.s: s , self.q_target: q_target ,self.elo:self.prev_diff} )

            self.writer.add_summary( summ , epoch*batches + b )
            
            #inc the swapping counter
            self.learn_step_counter += 1



if  BOARD_SIZE == 19:
    env = gym.make('Go19x19-v0')
elif BOARD_SIZE==9:
    env = gym.make('Go9x9-v0')
else:
    raise 'Board size must be 19 or 9'

env = env.unwrapped
env.seed(1)


ACTION_SPACE = BOARD_SIZE**2+1
sess = tf.Session()

with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDeepQNN( network ,  n_actions=ACTION_SPACE, input_shape=[None , BOARD_SIZE , BOARD_SIZE , 17] , sess=sess)
    double_DQN.saver.restore( sess , 'training saves/model'+str(MODEL_NUMBER)+'.ckpt' )
    

mcts = MCTS( double_DQN , MCTS_NUM_SIMULATIONS , board_size=BOARD_SIZE )

def val( state ,action , all_actions , t ):
    s = max ( sum( [ mcts.tree[ ( state , a ) ]['VisitCount']**(1/t) for a in all_actions ] ) , 1 )
    return mcts.tree[ ( state , action ) ]['VisitCount']**(1/t) / s


def up( rolling_state ,  ob , test=False):
    rolling_state = np.delete( rolling_state , -3 , 2 )
    rolling_state = np.delete( rolling_state , -2 , 2 )
    rolling_state = np.concatenate( [ np.expand_dims( ob[1][:][:] , -1 ) , rolling_state ] , axis=2 )
    rolling_state = np.concatenate( [ np.expand_dims( ob[0][:][:] , -1 ) , rolling_state ] , axis=2 )
    
    if test:
        return
    
    if rolling_state[:,:,-1][0][0] == 0:
        rolling_state = np.delete( rolling_state , -1 , 2 )
        rolling_state = np.concatenate( [ rolling_state , np.expand_dims( np.ones( ( BOARD_SIZE,BOARD_SIZE) ) , -1 ) ] , axis=2 )

    else:
        rolling_state = np.delete( rolling_state , -1 , 2 )
        rolling_state = np.concatenate( [ rolling_state , np.expand_dims( np.zeros( ( BOARD_SIZE,BOARD_SIZE) ) , -1 ) ] , axis=2 )\

def pretty_print( env , back , string):
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'

    for j in range( 0 , back):
        print(CURSOR_UP_ONE + ERASE_LINE+CURSOR_UP_ONE)
    print (string )
    env.render()
       

def train(RL):

    action = None

    score = 0
    for g in range( num_test_games ):
        print ( 'Test Game     ' + str( g ) )

        env.reset()
        if g == 0:
            print ( '' )
            print ( '' )
            env.render()
        done = False
        cntr = 0

        pretty_print(env,BOARD_SIZE+8,'Test play number\t'+str(g+1)+ '\t Score:\t'+str(score )+'/' +str(g) )

        rolling_st = np.zeros( ( BOARD_SIZE , BOARD_SIZE ,17 ) )

        while done == False:
            all_actions = mcts.getActions( env.state.board , env.state.color )

            env.test = False
            mcts.runSimulation( action , env , MCTS_DEPTH )
            env.test = True

            if len( all_actions ) == 0:
                done = True
                reward = env.state.board.official_score
            else:
                t = 0.01
                action = max([( val(env.state.board , a , all_actions , t) , a )  for a in all_actions ] , key=lambda t: t[0] )[1]
                observation_, reward, done, info = env.step( action )

                up( rolling_st , observation_ , test=True )
                cntr += 1


            pretty_print(env,BOARD_SIZE+6,'Test play number\t'+str(g+1)+ '\t Score:\t'+str(score )+'/' +str(g) )


        if reward < 0:
            print ( 'We Lost' )
        elif reward > 0:
            score += 1
            print ( 'WE WON' )
        else:
            score += 0.5
            print ( 'WE TIED' )

        mcts.tree = defaultdict(float)
        env.test = False
    return
                
            
train(double_DQN)
