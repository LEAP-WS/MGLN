# -*- coding: utf-8 -*-
import tensorflow as tf
from CNNLayer import *
import numpy as np
from funcCNN import *
from CNNLayer import uniform
from CNNLayer import glorot



def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= tf.transpose(mask)
    return tf.reduce_mean(tf.transpose(loss))


    


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= tf.transpose(mask)
    return tf.reduce_mean(tf.transpose(accuracy_all))

class GGeModel(object):
    def __init__(self, features, labels, learning_rate, num_classes, mask,
                 idea_A_tr, mat01_tr, pos_A_equals_1, nei01):

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.classlayers = []
        self.mlplayers = []
        self.globallayers = []
        self.labels = labels
        [self.sp_num, self.fea_num] = np.shape(features)
        self.inputs = features
        self.posx = []
        self.loss = 0
        self.idea_A_tr = idea_A_tr 
        self.mat01_tr = mat01_tr 
        
        self.emd_out = []
        self.adj_pred = []
        self.inputs_W1 = []
        self.C = []
        self.W1 = []
        self.tmp = []

        self.concat_vec = []
        self.outputs = None
        self.num_classes = num_classes
        self.mask = mask
        self.Wd =[] 
        self.nei01 = nei01
        self.pos_A_equals_1 = pos_A_equals_1
        self.CalAttenCoef()
        self.para_loss =  [tf.abs(tf.Variable(0.0)) + 0.01]
        self.para_scale = [tf.abs(tf.Variable(0.5)), tf.abs(tf.Variable(0.1)), tf.abs(tf.Variable(0.1)), tf.abs(tf.Variable(0.1)), tf.abs(tf.Variable(0.1))]
        self.para_lo_gl = [tf.abs(tf.Variable(0.5))]
        self.build()
        
        
    def _build(self):
        global_activations = []
        activations = []
        activations.append(tf.matmul(self.inputs, self.W1))
        for scale_idx in range(2):
            self.C.append(self.C[0]*self.nei01[scale_idx])
            self.C[-1] = tf.transpose(tf.transpose(self.C[-1])/tf.reduce_sum(self.C[-1], axis = 1))
            self.C[-1] = tf.SparseTensor(self.pos_A_equals_1[scale_idx], tf.gather_nd(self.C[-1], self.pos_A_equals_1[scale_idx]), self.C[-1].get_shape())

            self.classlayers.append(GraphConvolution(act = tf.nn.relu,
                                      input_dim = 32,
                                      output_dim = 128,
                                      support = self.C[-1],
                                      bias = True,
                                      isSparse = True
                                      ))   
            layer = self.classlayers[-1]        
            hidden = layer(activations[0])
            activations.append(hidden) 
        activations.append(tf.nn.l2_normalize(activations[1] + self.para_scale[1] * activations[2], dim = 1))
        activations.append(tf.nn.l2_normalize(activations[2] + self.para_scale[2] * activations[1], dim = 1))                    
        
        for scale_idx in range(2):
            self.classlayers.append(GraphConvolution(act = lambda x:x,
                                        input_dim = 128,
                                       output_dim = 128,
                                       support = self.C[scale_idx + 1],
                                       bias = True,
                                       isSparse = True
                                       ))   
            layer = self.classlayers[-1]
            hidden = layer(activations[scale_idx + 3])
            activations.append(hidden)
        
        
        self.emd_out = self.para_scale[0]*activations[5] + (1-self.para_scale[0])*activations[6] + self.para_scale[3] * activations[1] + self.para_scale[4] * activations[2]
        self.emd_out = tf.nn.l2_normalize(self.emd_out, dim = 1)


        hidden_Wd = self.emd_out*self.emd_out
        E1 = tf.concat([hidden_Wd, tf.ones([self.sp_num, 128])], 1)
        E2 = tf.concat([tf.ones([self.sp_num, 128]), hidden_Wd], 1)
        E1_2 = tf.matmul(E1, tf.transpose(E2))-2*tf.matmul(self.emd_out, tf.transpose(self.emd_out))
        self.adj_pred = tf.exp(-E1_2)
        
        
        self.mlplayers.append(MLP(act = lambda x:x,
                                  input_dim = 128,
                                  output_dim = self.num_classes,
                                  bias = True
                                  ))   
        layer = self.mlplayers[-1]
        hidden = layer(self.emd_out)
        activations.append(hidden) 
        
        
        global_A = self.nnGraph(self.adj_pred, 0.75)
        self.globallayers.append(GraphConvolution(act = tf.nn.relu,
                                  input_dim = self.fea_num,
                                  output_dim = 128,
                                  support = global_A,
                                  bias = True,
                                  isSparse = True
                                  ))  
        layer = self.globallayers[-1]  
        hidden = layer(self.inputs)
        global_activations.append(tf.nn.l2_normalize(hidden, dim = 1))
        
        self.globallayers.append(GraphConvolution(act = lambda x:x,
                                  input_dim = 128,
                                  output_dim = self.num_classes,
                                  support = global_A,
                                  bias = True,
                                  isSparse = True
                                  ))
        layer = self.globallayers[-1]  
        hidden = layer(global_activations[-1])
        global_activations.append(tf.nn.l2_normalize(hidden, dim = 0))
        
        self.concat_vec = self.para_lo_gl[0] * activations[-1] + (1-self.para_lo_gl[0]) * global_activations[-1]



                

            
    def build(self):
        self._build()
        self.outputs = self.concat_vec
        self._loss()
        self._accuracy()
        self.opt_op = self.optimizer.minimize(self.loss)        
        
    def _loss(self):
        self.loss += self.para_loss[0] * masked_softmax_cross_entropy(self.outputs, self.labels, self.mask)
        self.loss +=  (1.0-self.para_loss[0]) * tf.reduce_mean(tf.square(self.mat01_tr*(self.idea_A_tr - self.adj_pred)))

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.labels, self.mask)     
        

    def CalAttenCoef(self):
        self.W1 = uniform([self.fea_num, 32], name='W1')
        a_var_part1 = uniform([32, 1], name='a1')
        a_var_part2 = uniform([32, 1], name='a2')
        self.inputs_W1 = tf.matmul(self.inputs, self.W1)
        inputs1 = tf.matmul(self.inputs_W1, a_var_part1)
        inputs2 = tf.matmul(self.inputs_W1, a_var_part2)
        
        inputs2 = tf.transpose(tf.concat([inputs2, tf.ones([self.sp_num, 1])], axis = 1))
        inputs1 = tf.concat([tf.ones([self.sp_num, 1]), inputs1], axis = 1)
        
        self.C.append(tf.exp(tf.nn.leaky_relu(tf.matmul(inputs1, inputs2))))
        
          

        
    def nnGraph(self, mat1, th): 
        arr_idx = tf.where(mat1>=th)
        arr_sparse = tf.SparseTensor(arr_idx, tf.gather_nd(mat1, arr_idx), mat1.get_shape())
        s1 = tf.sparse.reduce_sum(arr_sparse, 1)
        return tf.sparse.transpose(tf.sparse.transpose(arr_sparse).__div__(s1))
        

    def Dense2Sparse(self, arr_tensor):
        arr_idx = tf.where(tf.not_equal(arr_tensor, 0))
        arr_sparse = tf.SparseTensor(arr_idx, tf.gather_nd(arr_tensor, arr_idx), arr_tensor.get_shape())
        return arr_sparse        
        
    
