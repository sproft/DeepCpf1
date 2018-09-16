import sys
import tensorflow as tf
from os import system
from numpy import *

def PREPROCESS(lines, data_len, data_offset):
    data_n = len(lines)-1
    DATA_X = zeros((data_n,1,data_len,4), dtype=int)
    DATA_Y = zeros((data_n,1), dtype=float)
    DATA_CA = ones((data_n,1), dtype=float)

    for l in range(data_n):
        data = lines[l+1].split()
        seq = data[1]
        for i in range(data_len):
            if seq[i+data_offset]in "Aa":
                DATA_X[l, 0, i, 0] = 1
            elif seq[i+data_offset] in "Cc":
                DATA_X[l, 0, i, 1] = 1
            elif seq[i+data_offset] in "Gg":
                DATA_X[l, 0, i, 2] = 1
            elif seq[i+data_offset] in "Tt":
                DATA_X[l, 0, i, 3] = 1
            else:
                print "Non-ATGC character " + seq[i+data_offset]
                sys.exit()
                
        if len(data) > 2:
            DATA_Y[l,0] = float(data[2])
        
        if len(data) > 3:
            DATA_CA[l,0] = float(data[3])

    return DATA_X, DATA_Y, DATA_CA

class Seq_deepCpf1(object):
    def __init__(self):   
        self.inputs = tf.placeholder(tf.float32, [None, 1, 34, 4])
        self.targets = tf.placeholder(tf.float32, [None, 1])
        self.is_training = tf.placeholder(tf.bool)

        with tf.variable_scope('Convolution_Layer'):
            W_conv = tf.get_variable("W_conv", shape=[1, 5, 4, 80])
            B_conv = tf.get_variable("B_conv", shape=[80])
            L_conv_pre = tf.nn.bias_add(tf.nn.conv2d(self.inputs, W_conv, strides=[1,1,1,1], padding='VALID'), B_conv)
            L_conv = tf.nn.relu(L_conv_pre)
            L_conv_drop = tf.layers.dropout(L_conv, 0.3, self.is_training)
            
        with tf.variable_scope('AveragePooling_Layer'):
            L_pool = tf.nn.avg_pool(L_conv_drop, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')

        with tf.variable_scope('Fully_Connected_Layer1'):
            L_flatten = tf.reshape(L_pool, [-1, 15*80])
            W_fcl1 = tf.get_variable("W_fcl1", shape=[15*80, 80])
            B_fcl1 = tf.get_variable("B_fcl1", shape=[80])
            L_fcl1_pre = tf.nn.bias_add(tf.matmul(L_flatten, W_fcl1), B_fcl1)
            L_fcl1 = tf.nn.relu(L_fcl1_pre)
            L_fcl1_drop = tf.layers.dropout(L_fcl1, 0.3, self.is_training)

        with tf.variable_scope('Fully_Connected_Layer2'):
            W_fcl2 = tf.get_variable("W_fcl2", shape=[80, 40])
            B_fcl2 = tf.get_variable("B_fcl2", shape=[40])
            L_fcl2_pre = tf.nn.bias_add(tf.matmul(L_fcl1_drop, W_fcl2), B_fcl2)
            L_fcl2 = tf.nn.relu(L_fcl2_pre)
            L_fcl2_drop = tf.layers.dropout(L_fcl2, 0.3, self.is_training)
            
        with tf.variable_scope('Fully_Connected_Layer3'):
            W_fcl3 = tf.get_variable("W_fcl3", shape=[40, 40])
            B_fcl3 = tf.get_variable("B_fcl3", shape=[40])
            L_fcl3_pre = tf.nn.bias_add(tf.matmul(L_fcl2_drop, W_fcl3), B_fcl3)
            L_fcl3 = tf.nn.relu(L_fcl3_pre)
            L_fcl3_drop = tf.layers.dropout(L_fcl3, 0.3, self.is_training)
            
        with tf.variable_scope('Output_Layer'):
            W_out = tf.get_variable("W_out", shape=[40, 1])
            B_out = tf.get_variable("B_out", shape=[1])
            self.outputs = tf.nn.bias_add(tf.matmul(L_fcl3_drop, W_out), B_out)

        # Define loss function and optimizer
        self.obj_loss =  tf.reduce_mean(tf.square(self.targets - self.outputs))
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.obj_loss)  
        
class DeepCpf1(object):
    def __init__(self):   
        self.inputs = tf.placeholder(tf.float32, [None, 1, 34, 4])
        self.ca = tf.placeholder(tf.float32, [None, 1])
        self.targets = tf.placeholder(tf.float32, [None, 1])
        self.is_training = tf.placeholder(tf.bool)

        with tf.variable_scope('Convolution_Layer'):
            W_conv = tf.get_variable("W_conv", shape=[1, 5, 4, 80])
            B_conv = tf.get_variable("B_conv", shape=[80])
            L_conv_pre = tf.nn.bias_add(tf.nn.conv2d(self.inputs, W_conv, strides=[1,1,1,1], padding='VALID'), B_conv)
            L_conv = tf.nn.relu(L_conv_pre)
            L_conv_drop = tf.layers.dropout(L_conv, 0.3, self.is_training)
            
        with tf.variable_scope('AveragePooling_Layer'):
            L_pool = tf.nn.avg_pool(L_conv_drop, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')

        with tf.variable_scope('Fully_Connected_Layer1'):
            L_flatten = tf.reshape(L_pool, [-1, 15*80])
            W_fcl1 = tf.get_variable("W_fcl1", shape=[15*80, 80])
            B_fcl1 = tf.get_variable("B_fcl1", shape=[80])
            L_fcl1_pre = tf.nn.bias_add(tf.matmul(L_flatten, W_fcl1), B_fcl1)
            L_fcl1 = tf.nn.relu(L_fcl1_pre)
            L_fcl1_drop = tf.layers.dropout(L_fcl1, 0.3, self.is_training)

        with tf.variable_scope('Fully_Connected_Layer2'):
            W_fcl2 = tf.get_variable("W_fcl2", shape=[80, 40])
            B_fcl2 = tf.get_variable("B_fcl2", shape=[40])
            L_fcl2_pre = tf.nn.bias_add(tf.matmul(L_fcl1_drop, W_fcl2), B_fcl2)
            L_fcl2 = tf.nn.relu(L_fcl2_pre)
            L_fcl2_drop = tf.layers.dropout(L_fcl2, 0.3, self.is_training)
            
        with tf.variable_scope('Fully_Connected_Layer3'):
            W_fcl3 = tf.get_variable("W_fcl3", shape=[40, 40])
            B_fcl3 = tf.get_variable("B_fcl3", shape=[40])
            L_fcl3_pre = tf.nn.bias_add(tf.matmul(L_fcl2_drop, W_fcl3), B_fcl3)
            L_fcl3 = tf.nn.relu(L_fcl3_pre)
            L_fcl3_drop = tf.layers.dropout(L_fcl3, 0.3, self.is_training)
            
        with tf.variable_scope('CA_Integration_Layer'):
            W_ca = tf.get_variable("W_ca", shape=[1, 40])
            B_ca = tf.get_variable("B_ca", shape=[40])
            L_ca_pre = tf.nn.bias_add(tf.matmul(self.ca, W_ca), B_ca)
            L_ca = tf.nn.relu(L_ca_pre)
            L_int = tf.multiply(L_fcl3, L_ca)            
            L_int_drop = tf.layers.dropout(L_int, 0.3, self.is_training)
            
        with tf.variable_scope('Output_Layer'):
            W_out = tf.get_variable("W_out", shape=[40, 1])
            B_out = tf.get_variable("B_out", shape=[1])
            self.outputs = tf.nn.bias_add(tf.matmul(L_int_drop, W_out), B_out)

        # Define loss function and optimizer
        self.obj_loss =  tf.reduce_mean(tf.square(self.targets - self.outputs))
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.obj_loss, 
                                                                var_list= (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CA_Integration_Layer')
                                                                          +tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Output_Layer')))
             
def Model_PreTrain(sess, model, TRAIN_DATA_X, TRAIN_DATA_Y, VAL_DATA_X, VAL_DATA_Y):
    saver = tf.train.Saver(max_to_keep=None)
    Val_loss = zeros((100))

    Patience = 5
    train_batch = TRAIN_DATA_X.shape[0]/10
    test_batch = 1000
    
    patience = Patience
    optimizer = model.optimizer
        
    ep = -1
    while 1:
        ep += 1
        if ep == Val_loss.shape[0]:
            Val_loss = concatenate((Val_loss, zeros((100))))
         
        if patience == 0:
            saver.restore(sess, "model_checkpoints/PreTrain-ep-%d" % (nanargmin(Val_loss[:ep])))
            system("rm model_checkpoints/PreTrain-ep-*")
            saver.save(sess, "model_checkpoints/PreTrain-Final")
            return

        if ep >= 1:
            for i in range(int(ceil(float(TRAIN_DATA_X.shape[0])/float(train_batch)))):
                _, obj_loss = sess.run([optimizer, model.obj_loss], 
                                       feed_dict={model.inputs: TRAIN_DATA_X[i*train_batch:(i+1)*train_batch],
                                                  model.targets: TRAIN_DATA_Y[i*train_batch:(i+1)*train_batch],
                                                  model.is_training: True})
                
        for i in range(int(ceil(float(VAL_DATA_X.shape[0])/float(test_batch)))):
            val_loss = sess.run([model.obj_loss], 
                                feed_dict={model.inputs: VAL_DATA_X[i*test_batch:(i+1)*test_batch],
                                           model.targets: VAL_DATA_Y[i*test_batch:(i+1)*test_batch]})
            Val_loss[ep] += val_loss
            
        if argmin(Val_loss[:ep+1]) != ep:
            patience -= 1
        else:
            patience = Patience
            
        saver.save(sess, "model_checkpoints/PreTrain-ep", global_step=ep)
        
    return

def Model_FineTune(sess, model, TRAIN_DATA_X, TRAIN_DATA_Y, TRAIN_DATA_CA):
    saver = tf.train.Saver(max_to_keep=None)
    train_batch = TRAIN_DATA_X.shape[0]/10
    test_batch = 100

    optimizer = model.optimizer
    for ep in range(10):
        for i in range(int(ceil(float(TRAIN_DATA_X.shape[0])/float(train_batch)))):
            _, obj_loss = sess.run([optimizer, model.obj_loss], 
                                   feed_dict={model.inputs: TRAIN_DATA_X[i*train_batch:(i+1)*train_batch],
                                              model.ca: TRAIN_DATA_CA[i*train_batch:(i+1)*train_batch],
                                              model.targets: TRAIN_DATA_Y[i*train_batch:(i+1)*train_batch],
                                              model.is_training: True})

    saver.save(sess, "model_checkpoints/FineTune-Final")
    return 

def Model_Load(sess, mode):
    if mode == "DeepCpf1":
        model = DeepCpf1()
        saver = tf.train.Saver()
        saver.restore(sess, "model_checkpoints/FineTune-Final")
    elif mode == "Seq-deepCpf1":
        model = Seq_deepCpf1()
        saver = tf.train.Saver()
        saver.restore(sess, "model_checkpoints/PreTrain-Final")
    else:
        print "mode must be either \"DeepCpf1\" or \"Seq-deepCpf1\""
    return model
            
def Model_Prediction(sess, model, mode, TEST_DATA_X, TEST_DATA_CA=None):
    TEST_DATA_Z = zeros((TEST_DATA_X.shape[0], 1), dtype=float)
    test_batch = 500
    
    for i in range(int(ceil(float(TEST_DATA_X.shape[0])/float(test_batch)))):
        if mode == "DeepCpf1":
            Dict = {model.inputs: TEST_DATA_X[i*test_batch:(i+1)*test_batch],
                    model.ca: TEST_DATA_CA[i*test_batch:(i+1)*test_batch],
                    model.is_training: False}
        elif mode == "Seq-deepCpf1":
            Dict = {model.inputs: TEST_DATA_X[i*test_batch:(i+1)*test_batch], model.is_training: False}
        else:
            print "mode must be either \"DeepCpf1\" or \"Seq-deepCpf1\""
            
        TEST_DATA_Z[i*test_batch:(i+1)*test_batch] = sess.run([model.outputs], feed_dict=Dict)[0]
    
    return TEST_DATA_Z