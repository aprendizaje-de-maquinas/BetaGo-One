#***************************************************
#******************COPYRIGHT 2017.******************
#**Jonathan Booher, Enrique De Alba, Nithin Kannan**
#****************ALL RIGHTS RESERVED****************
#***************************************************



import pachi_py # the backend for the open-ai go env
import gym
import collections
import numpy as np
import math
import copy

import sys
import time
import tensorflow as tf

# so wed ont run into any issues at all
sys.setrecursionlimit(1500)

class MCTS():

    def __init__( self, double_q , numGames , board_size=19 , epsilon=0.25 , dirchletParam=0.03 , c=0.4):

        self.numGames = numGames
        self.board_size = board_size
        self.epsilon = epsilon
        self.dirchletParam = dirchletParam
        self.C = c
        self.tree = collections.defaultdict( float )
        self.todel = None
        self.network = double_q

    # theses are copied from the openai's go.py
    #    for interfacing with the openai backend for go
    def _pass_action( self ):
        return self.board_size**2
    def _resign_action( board_size=9 ):
        return self.board_size**2 + 1
    def _coord_to_action( self , board , c ):
        if c == pachi_py.PASS_COORD: return self._pass_action()
        if c == pachi_py.RESIGN_COORD: return self._resign_action()
        i, j = board.coord_to_ij( c )
        return i* board.size + j

    def _action_to_coord( self , board , a ):
        if a == self._pass_action(board.size): return pachi_py.PASS_COORD
        if a == self._resign_action(board.size): return pachi_py.RESIGN_COORD
        return board.ij_to_coord(a // board.size, a % board.size)
    def getActions( self , board , color):
        pachi_legal_moves = board.get_legal_coords ( color )
        legal_actions = []
        for i in pachi_legal_moves:
            # disable resign as action for the agent.
            if i == pachi_py.RESIGN_COORD : continue #or i== pachi_py.PASS_COORD : continue
            legal_actions.append( self._coord_to_action( board , i ) )
        return legal_actions
    # end pach utils

    

    # this updates the state takes in the current observation and current rolling state
    #    removes the last three planes and adds the new observations at the top
    #    then changes the last plane to represent the next player to move
    def up( self , rolling_state ,  ob ):
        
        rolling_state = np.delete( rolling_state , -3 , 2 )
        rolling_state = np.delete( rolling_state , -2 , 2 )
        rolling_state = np.concatenate( [ np.expand_dims( np.array(ob[1][:][:]) , -1 ) , rolling_state ] , axis=2 )
        rolling_state = np.concatenate( [ np.expand_dims( np.array(ob[0][:][:]) , -1 ) , rolling_state ] , axis=2 )
        
        # handle the player plane
        if rolling_state[:,:,-1][0][0] == 0:
            rolling_state = np.delete( rolling_state , -1 , 2 )
            rolling_state = np.concatenate( [ rolling_state , np.expand_dims( np.ones( ( self.board_size,self.board_size) ) , -1 ) ] , axis=2 )
        else:
            rolling_state = np.delete( rolling_state , -1 , 2 )
            rolling_state = np.concatenate( [ rolling_state , np.expand_dims( np.zeros( ( self.board_size,self.board_size) ) , -1 ) ] , axis=2 )



    # modified version of PUTC algorithm in the Go paper
    def putc( self , state , action , all_actions ):
        s = math.sqrt( np.sum( [  self.tree[ (state,a) ]['VisitCount'] for a in all_actions ] ) )
        return self.tree[(state,action)]['MeanValue']+ self.C*self.tree[(state,action)]['Prior']*s/ ( 1+ self.tree[(state,action)]['VisitCount']  )


    # runs self.numGames siulations of MCTS
    def runSimulation( self , action_taken , state , L  , disp=False):

        # normalize
        def norm_probs( ps ):
            ps /= np.sum( ps )
            return ps
        
        # updates the tree with the action and the increased visit count
        def update( value , st , action ):
            self.tree[ ( st.state.board , action)]['VisitCount'] +=1
            self.tree[ ( st.state.board , action ) ]['TotalValue'] += value
            self.tree[ ( st.state.board , action ) ]['MeanValue'] = self.tree[ ( st.state.board , action ) ]['TotalValue'] / self.tree[ ( st.state.board , action ) ]['VisitCount']


        # recursive func for MCTS to allow for backing up the result of the game
        def rec( rolling_state , st , depth):

            '''
            # attempt at vectorization of a loop. had no positive efffect on runtime
            @vectorize( 'int32( float32 , int32 )' )
            def vectorizedLoopOverActions( p , action ):
                self.tree[ ( st.state.board , action ) ] = {'MeanValue':0.0,'VisitCount':0.0,'TotalValue':0.0,'Prior':p }
                return 0 
            '''
            # make a copy of the current state
            roll = rolling_state[:]

            # the expand portion of MCTS. As per the paper, we do not  have to go all the way down to the termination of the game
            if depth == 0:
                ps = self.network.sess.run( self.network.q_eval , {self.network.s : [roll] } )[0]

                ps = norm_probs(ps)
                
                for action in self.getActions ( st.state.board , st.state.color ):
                    p = ps[action]
                    self.tree [(st.state.board , action)] =  {'MeanValue':0.0,'VisitCount':0.0,'TotalValue':0.0,'Prior':p }

                val = np.max( ps )  # this would be from the network
                return val
        
            all_actions = self.getActions( st.state.board , st.state.color )
            if len( all_actions ) == 0:
                return 0.001

            # if all the actions from this state are already explored, then we can use PUTC
            if all ( self.tree[ ( st.state.board , action ) ] for action in all_actions ):

                action = max(  [ (  self.putc( st.state.board , a , all_actions ) , a )  for a in all_actions ] , key=lambda t: t[0] )[1]
                ps = self.network.sess.run( self.network.q_eval , {self.network.s: [roll]} )[0]

                p = ps[action]

                st_cp = copy.copy( st )
                ob, reward, done, info = st_cp.step( action )

                self.up( roll , ob )

                # update with the result of the recursive search
                value = rec ( roll , st_cp , depth -1 )
                update( value , st , action )

                return self.tree[ ( st.state.board , action ) ]['MeanValue']

            # we have not explored enough so take a random move as per the network
            ps = self.network.sess.run( self.network.q_eval , {self.network.s : [roll] } )[0]

            ps_cp = ps[all_actions]

            ps_cp = norm_probs( ps_cp ) 
            #ps_cp = self.network.sess.run( tf.nn.softmax( ps_cp ) ) 

            action = np.random.choice( self.getActions ( st.state.board , st.state.color ) , p=ps_cp )
            #action = np.random.choice( self.getActions ( st.state.board , st.state.color ) ) # we would sample this from the network dist
            p = ps[action]
            #p = np.random.rand()

            # advance the state
            st_cp = copy.copy( st )
            ob, reward, done, info = st_cp.step( action )
            self.up( roll , ob )


            # set the intital edge in the graph
            self.tree[ ( st_cp.state.board , action )] = {'MeanValue':0.0,'VisitCount':0.0,'TotalValue':0.0,'Prior':p }
            
            value = rec ( roll , st_cp , depth -1 )
            self.tree[ ( st.state.board , action )] = {'MeanValue':0.0,'VisitCount':0.0,'TotalValue':0.0,'Prior':p }
            
            update( value , st , action )
            
            return self.tree[ ( st.state.board , action ) ]['MeanValue']


        for _ in range( self.numGames ):

            rolling_state = np.zeros( ( self.board_size , self.board_size ,17 ) )

            
            # additional exploration at the root node
            dirchlet_noise =  np.random.dirichlet( tuple( [ self.dirchletParam ]*(self.board_size**2+1 )  ) , 1)
            all_actions = self.getActions( state.state.board , state.state.color )


            if len(all_actions ) == 0:
                return
            
            # special probabilities for the root node
            ps = self.network.sess.run( self.network.q_eval , {self.network.s : [rolling_state] } )[0]
            
            mask = np.ones( ps.shape , dtype=bool )
            mask[all_actions] = False
            ps_cp = ps[:]
            ps_cp[mask] = 0
            ps_cp = norm_probs( ps_cp )

            
            for action in all_actions:
                net_prob = ps_cp[action]
                p = ( 1- self.epsilon )* net_prob + self.epsilon*dirchlet_noise[0][ action ]
                self.tree[ ( state.state.board , action ) ] = {'MeanValue':0.0,'VisitCount':0.0,'TotalValue':0.0,'Prior':p }


            # chose the best and then go to the next state using that actions
            action = max(  [ (  self.tree[( state.state.board , a )]['Prior'] , a )  for a in all_actions ] , key=lambda t: t[0] )[1]

            #print ( action)
            st_cp = copy.copy( state ) 
            ob, reward, done, info = st_cp.step( action )

            self.up( rolling_state , ob )
            
            # init the entry of the tree
            self.tree[ ( state.state.board , action )] = {'MeanValue':0.0,'VisitCount':0.0,'TotalValue':0.0,'Prior':ps_cp[action] }
            
            value = rec ( rolling_state , st_cp , L - 1 )
            update ( value , state , action )

        return


# for testing purposes
if __name__ == '__main__':
    env = gym.make ( 'Go9x9-v0' )
    env.reset()
    env.illegal_move_mode='raise'
    mcts = MCTS( 160 )
    import time
    start = time.time()
    action = None
    def val( state ,action , all_actions):
        s = max ( sum( [ mcts.tree[ ( state , a ) ]['VisitCount']**(1/1) for a in all_actions ] ) , 1 )
        return mcts.tree[ ( state , action ) ]['VisitCount']**(1/1)*1.0 / s 
    while not env.state.board.is_terminal:
        all_actions = mcts.getActions( env.state.board , env.state.color )
        mcts.runSimulation( action , env , 3 )
        action = max(  [ ( val(env.state.board , a , all_actions) , a )  for a in all_actions ] , key=lambda t: t[0] )[1]
        env.step( action )
        env.render()
    print ( time.time() - start )
    print ( len( mcts.tree ) )

    


