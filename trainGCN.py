# -*- coding: utf-8 -*-
import numpy as np
from funcCNN import *
from GGeModel import GGeModel
from BuildSPInst_A import *
import tensorflow as tf
import time
import tensorflow.contrib.eager as tfe
import sys
import os




time_start=time.time()



def GCNevaluate(mask1, labels1, model, sess):
    t_test = time.time()

    outs_val = sess.run([model.loss, model.accuracy], feed_dict={labels: labels1, mask: mask1})
    return outs_val[0], outs_val[1], (time.time() - t_test)
learning_rate1 = 0.0001
num_classes = 16
epochs1 = 2000 
data_name = 'IP' 
seed = 123 


img_gyh = data_name+'_gyh'

img_gt = data_name+'_gt'



Data = load_HSI_data(data_name)
model = GetInst_A(Data['useful_sp_lab'], Data[img_gyh], Data[img_gt], Data['trpos'], Data['tepos'])
sp_mean = np.array(model.sp_mean, dtype='float32')
sp_label = np.array(model.sp_label, dtype='float32')
trmask = np.matlib.reshape(np.array(model.trmask, dtype='bool'), [np.shape(model.trmask)[0], 1])
temask = np.matlib.reshape(np.array(model.temask, dtype='bool'), [np.shape(model.trmask)[0], 1])
trmask_sp = np.matlib.reshape(np.array(model.trmask_sp, dtype='bool'), [np.shape(model.trmask_sp)[0], 1])
temask_sp = np.matlib.reshape(np.array(model.temask_sp, dtype='bool'), [np.shape(model.trmask_sp)[0], 1])
sp_A_notSym =  np.array(model.sp_A_notSym[0], dtype='float32')
sp_label_sp = model.sp_label_sp
sp_label_vec = model.sp_label_vec
[idea_A_tr, mat01_tr] = Cal01DisTrainMat(trmask_sp, sp_label_vec-1, 'zero')
idea_A_tr =  np.array(idea_A_tr, dtype='float32')
mat01_tr =  np.array(mat01_tr, dtype='float32')




mask = tf.placeholder("int32", [None, 1])
labels = tf.placeholder("float", [None, num_classes])

excel_data=[]


acc1 = []

sp_mean = Normalization_2dMat(sp_mean)
np.random.seed(seed)
tf.set_random_seed(seed)

sp_num = np.shape(sp_A_notSym)[0]
pos_A_equals_1 = []
pos_A_equals_1.append(np.argwhere(sp_A_notSym>0))
nei01 = []
nei01.append(np.zeros([sp_num, sp_num]))
nei01[0][pos_A_equals_1[0][:,0], pos_A_equals_1[0][:,1]] = 1

nei01.append(model.AddConnectionFor01(nei01[-1]))
pos_A_equals_1.append(np.argwhere(nei01[-1]>0))
nei01.append(model.AddConnectionFor01(nei01[-1]))
pos_A_equals_1.append(np.argwhere(nei01[-1]>0))
nei01.append(model.AddConnectionFor01(nei01[-1]))
pos_A_equals_1.append(np.argwhere(nei01[-1]>0))
del nei01[1]
del nei01[1]
del pos_A_equals_1[1]
del pos_A_equals_1[1]




GGemodel = GGeModel( features = sp_mean, labels = labels, learning_rate = learning_rate1, 
                    num_classes = num_classes, mask = mask,
                    idea_A_tr = idea_A_tr, mat01_tr = mat01_tr,  
                    pos_A_equals_1 = pos_A_equals_1, nei01 = nei01)



sess=tf.Session()
sess.run(tf.global_variables_initializer())


for epoch in range(epochs1):
    t = time.time() 
    outs = sess.run([GGemodel.opt_op, GGemodel.loss], feed_dict={ labels:sp_label_sp, 
                    mask:trmask_sp })
    if epoch >= 500:
        train_cost, train_acc, train_duration = GCNevaluate(trmask_sp, sp_label_sp, GGemodel, sess)
        if train_acc >= 0.993:
            break

    if epoch % 50 == 0:
        train_cost, train_acc, train_duration = GCNevaluate(trmask_sp, sp_label_sp, GGemodel, sess)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]), " train_accuracy=", "{:.5f}".format(train_acc), "time=", "{:.5f}".format((time.time() - t)))


outputs = sess.run(GGemodel.outputs)
pixel_wise_pred = np.argmax(outputs, axis=1)
pred_mat = AssignLabels(Data['useful_sp_lab'], np.argmax(sp_label_sp, axis=1), pixel_wise_pred, trmask_sp, temask_sp)
OA = PixelWiseAccuracy(Data[img_gt].copy(), pred_mat.copy(), Data['trpos'])
excel_data.append(GetExcelData(Data[img_gt], pred_mat.copy(), Data['trpos']-1))
scio.savemat('excel_data.mat',{'excel_data':excel_data}) 



