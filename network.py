#***************************************************
#******************COPYRIGHT 2017.******************
#**Jonathan Booher, Enrique De Alba, Nithin Kannan**
#****************ALL RIGHTS RESERVED****************
#***************************************************

import tensorflow as tf
import numpy as np


def network(net_input , collection , w_initializer , b_initializer , tower_height=20 , num_conv_filters1=256 , num_conv_filters2=2 , num_output=362):
    TOWER_HEIGHT = tower_height
    NUM_FILTERS1 = num_conv_filters1
    NUM_FILTERS2 = num_conv_filters2
    NUM_OUTPUT = num_output

    # conv is an abstraction for a convolutional layer.
    #  we have the strides set to 1 as that is what is called for in the paper
    def conv( x ,w , name):
        return tf.nn.conv2d( x , w , strides=[1,1,1,1] , padding='SAME' , name=name )
    # abstraction to init weight variables with random samples from a normal dist
    def weight_var( shape , name ):
        init = tf.random_normal_initializer( 0.0 , 0.3 )
        return tf.get_variable( name , shape , initializer=w_initializer , collections=collection)
    # abstraction to init bias variables with a constant value of 0.1
    def bias_var( shape , name ):
        init = tf.constant_initializer( 0.1 )
        return tf.get_variable( name , shape , initializer=b_initializer , collections=collection)


    # the tf.variable_scope is so that the graph shows having distinct parts with these names (see the graph for details) 
    with tf.variable_scope( 'top-layer'):
        # moved net_input to param
        #net_input = tf.placeholder( tf.float32 , [ None , 19 , 19 , 17 ] , name='input')

        # in this block of code we apply as per the paper
        #   (1) A convolution of 256 filters of kernel size 3 ×​ 3 with stride 1
        #   (2) Batch normalization
        #   (3) A rectifier nonlinearity
        # see below for further explanation

        

        # from the paper we have that the top layer was 256 3x3 kernels with stride 1. The weights variable captures the dimensions
        #   of the kernel as the third dimmension is assumed to go all the way depthwise and we have NUM_FILTERS1 conv filters.
        #   we add in bias and perform the batch norm as described in the paper.
        w_conv1 = weight_var( [ 3 , 3 , 17 , NUM_FILTERS1 ] , 'weights' )
        b_conv1 = bias_var( [ NUM_FILTERS1 ] , 'bias')
        batch_norm1 = tf.layers.batch_normalization( conv( net_input , w_conv1 , 'conv' ) + b_conv1 , name='batch-norm' )

        # apply the rectifier to create the output of the top layer
        h_conv1 = tf.nn.relu( batch_norm1 , name='top-ouput' )


    # the residual tower naming
    with tf.variable_scope( 'residual-tower' ):

        # references stored to the previous layer and the input to the residual block
        # note that layer == None is simply an easy way to identify the begining of the tower
        # also note that layer will be used outside the loop so it is necessary that it is decalred outside
        layer = None
        res_block_input = h_conv1

        # make TOWER_HEIGHT residual blocks
        for x in range( TOWER_HEIGHT ):
            # make sure they are names properly
            name_scope = 'res-block-'+str( x )
            with tf.variable_scope( name_scope ):

                # each residual block (iteration through the loop) creates the following as per the paper:
                #   (1) A convolution of 256 filters of kernel size 3 ×​ 3 with stride 1
                #   (2) Batch normalization
                #   (3) A rectifier nonlinearity
                #   (4) A convolution of 256 filters of kernel size 3 ×​ 3 with stride 1
                #   (5) Batch normalization
                #   (6) A skip connection that adds the input to the block
                #   (7) A rectifier nonlinearity



                w=  weight_var( [ 3 , 3 , NUM_FILTERS1 , NUM_FILTERS1] , 'weight1')
                b = bias_var( [ NUM_FILTERS1 ] , 'bias1')

                # note that this can prob be simplified if layer= h_conv1 at the begining
                if layer == None:
                    layer = conv( h_conv1 , w , 'conv1') + b
                else:
                    layer = conv( layer , w , 'conv1') + b

                batch_norm =  tf.layers.batch_normalization( layer  , name='batch-norm1' )
                layer =  tf.nn.relu( batch_norm  , name='itermediate-output' )

                w_2 =  weight_var( [ 3 , 3 , NUM_FILTERS1 , NUM_FILTERS1] , 'weight2' )
                b_2 = bias_var( [ NUM_FILTERS1 ] , 'bias2' )

                layer = conv( layer , w_2 , 'conv2' ) + b_2

                batch_norm =  tf.layers.batch_normalization( layer  , name='batch-norm2' )


                # name the output correctly (note tower-output is still under the guise of the last 'res-block'
                # maybe i can fix that
                # this is where the shortcut is added in
                if x == TOWER_HEIGHT -1:
                    output = tf.nn.relu( batch_norm + res_block_input , name='tower-output' ) 
                else:
                    layer =  tf.nn.relu( batch_norm + res_block_input  , name='block-output')
                    
            # make the residual input to the next block the output of this block
            res_block_input = layer


    # this is the policy head as described in the paper:
    #   (1) A convolution of 2 filters of kernel size 1 ×​ 1 with stride 1
    #   (2) Batch normalization
    #   (3) A rectifier nonlinearity
    #   (4) A fully connected linear layer that outputs a vector of size 19**2  +​  1  =​  362,
    #            corresponding to logit probabilities for all intersections and the pass move
    # note that 362 is defined as NUM_OUTPUT and that NUM_FILTERS2 is set to 2 by default.
    
    with tf.variable_scope( 'head' ):
        w = weight_var( [ 1 , 1 , NUM_FILTERS1 , NUM_FILTERS2 ] , 'weight1' )
        b = bias_var( [NUM_FILTERS2] , 'bias1' )

        layer = conv( output , w , 'conv' ) + b
        batch_norm = tf.layers.batch_normalization( layer , name='batch-norm' )
        layer = tf.nn.relu( batch_norm , name='rectifier' )

        # needed to ensure that the shapes match. Note that if I new the numbers I could hard code it here
        #    but that would take too much work to figure out what it is RIP.
        d = int( layer.get_shape()[1] )* int( layer.get_shape()[2] )
        w = weight_var( [ d* NUM_FILTERS2 , NUM_OUTPUT ] , 'weight2')
        b = bias_var( [ NUM_OUTPUT ] , 'bias2')

        # note the -1 that represents the number of batches and the output is a matrix of [batch_size , NUM_OUTPUT ] as required.
        b2 = tf.constant( 0.01 , shape=[NUM_OUTPUT] , dtype=tf.float32 , name='sftmaxbias' )
        output = tf.nn.softmax( tf.add ( tf.add(  tf.matmul( tf.reshape( layer , [ -1 , d*NUM_FILTERS2 ] ) , w ) ,  b ) , b2 ) , name='network-output')

    return output



#for testing purposes
if __name__ == '__main__':
    import time


    BATCH_SIZE = 3

    i = np.random.rand(  1600 , 9 , 9 , 17 )

    # note that the input as defined in the paper is [ 19 , 19 , 17 ]. prepend another dimmension for batches
    #   the 17 comes from the fact that board states from previous time steps for both players are stored.
    #   ie 8 of them correspond to black and 8 for white and the last element (C) is a 19x19 matrix of constants ( 1 for black and 0 for white )
    #   representing whose turn it is to move.
    #   organization along that dimension should be [ X_t , Y_t , X_{t-1} , Y_{t-1] , ... , X_{t-7} , Y_{t-7} , C ]
    #   where every X_i and Y_i are 19x19 and contain a 1 if the respective player has a stone there at time t and 0 otherwise
    net_input = tf.placeholder( tf.float32 , [ None , 9 , 9 , 17 ] , name='input')
    
    output = network( net_input , ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES] , tf.random_normal_initializer( 0.0 , 0.3 ) , tf.constant_initializer( 0.1 ) )
    
    with tf.Session() as sess:
        sess.run( tf.global_variables_initializer() )
        
        #output the graph
        writer = tf.summary.FileWriter("tmp", sess.graph)
        
        # start the network
        start = time.time()
        print ( sess.run( output , feed_dict={ net_input: i } ).shape , time.time() - start  ) # this does through all init and takes time
        
        
        # now that everything is inited, we can see how long it will actually take per batch
        start = time.time()
        print ( sess.run( output , feed_dict={ net_input: i } ).shape , time.time() - start  ) # this will be the actual speed
        
        # to write out all the names of all the vars in the graph.
        #   use this or do 'tensorboard --logdir=tmp' and then go to 'localhost:6006' in a browser
        #for op in tf.get_default_graph().get_operations():
        #    print ( str(op.name) ) 
        
