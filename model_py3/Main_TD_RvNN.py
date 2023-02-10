# -*- coding: utf-8 -*-
"""
@object: Twitter
@task: Main function of recursive NN (4 classes)
@author: majing
@structure: Top-Down recursive Neural Networks
@variable: Nepoch, lr, obj, fold
@time: Jan 24, 2018
"""

## LabelMap = {
##    "nonR": 1, 
##    "f": 2,
##    "t": 3, 
##    "u": 4
## }

import os
import pdb
import sys
import math
import time
import random
import datetime
import argparse
import importlib
import numpy as np
from tqdm import tqdm
from numpy.testing import assert_array_almost_equal

importlib.reload(sys)
#sys.setdefaultencoding("utf-8")

import theano
from theano import tensor as T

import TD_RvNN
from evaluate import *
#from Util import *

def parse_args():
	parser = argparse.ArgumentParser(description="Rumor Detection")

	## Others
	parser.add_argument("--lr", type=float, default=0.005)
	parser.add_argument("--Nepoch", type=int, default=200)
	parser.add_argument("--Nclass", type=int, default=4)
	parser.add_argument("--hidden_dim", type=int, default=100)
	parser.add_argument("--vocabulary_size", type=int, default=5000)
	parser.add_argument("--tag", type=str, default="_u2b")
	parser.add_argument("--fold", type=str, default="2", help="fold index, choose from 0-4")
	parser.add_argument("--obj", type=str, default="Twitter15", choices=["Twitter15", "Twitter16"])

	args = parser.parse_args()

	return args

#floss = open(lossPath, 'a+')

################################### tools #####################################
def str2matrix(Str, MaxL): # str = index:wordfreq index:wordfreq
	wordFreq, wordIndex = [], []
	l = 0
	for pair in Str.split(' '):
		wordFreq.append(float(pair.split(':')[1]))
		wordIndex.append(int(pair.split(':')[0]))
		l += 1
	ladd = [ 0 for i in range( MaxL-l ) ]
	wordFreq += ladd 
	wordIndex += ladd 
	#print MaxL, l, len(Str.split(' ')), len(wordFreq)
	#print Str.split(' ')
	return wordFreq, wordIndex 

def loadLabel(label, l1, l2, l3, l4):
	labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
	if label in labelset_nonR:
	   y_train = [1,0,0,0]
	   l1 += 1
	if label in labelset_f:
	   y_train = [0,1,0,0] 
	   l2 += 1
	if label in labelset_t:
	   y_train = [0,0,1,0] 
	   l3 += 1 
	if label in labelset_u:
	   y_train = [0,0,0,1] 
	   l4 += 1
	return y_train, l1,l2,l3,l4

def constructTree(tree):
	## tree: {index1:{'parent':, 'maxL':, 'vec':}
	## 1. ini tree node
	index2node = {}
	for i in tree:
		node = TD_RvNN.Node_tweet(idx=i)
		index2node[i] = node
	## 2. construct tree
	for j in tree:
		indexC = j 
		indexP = tree[j]['parent']
		nodeC = index2node[indexC]
		wordFreq, wordIndex = str2matrix( tree[j]['vec'], tree[j]['maxL'] )
		#print tree[j]['maxL']
		nodeC.index = wordIndex
		nodeC.word = wordFreq
		#nodeC.time = tree[j]['post_t']
		## not root node ## 
		if not indexP == 'None':
		   nodeP = index2node[int(indexP)]
		   nodeC.parent = nodeP
		   nodeP.children.append(nodeC)
		## root node ##
		else:
		   root = nodeC
	## 3. convert tree to DNN input    
	parent_num = tree[j]['parent_num'] 
	ini_x, ini_index = str2matrix( "0:0", tree[j]['maxL'] )
	#x_word, x_index, tree = tree_gru_u2b.gen_nn_inputs(root, ini_x, ini_index) 
	x_word, x_index, tree = TD_RvNN.gen_nn_inputs(root, ini_x) 
	return x_word, x_index, tree, parent_num       
			   
################################# load data ###################################
def loadData():
	print("loading tree label")
	labelDic = {}
	for line in open(labelPath):
		line = line.rstrip()
		label, eid = line.split('\t')[0], line.split('\t')[2]
		labelDic[eid] = label.lower()
	print(len(labelDic))
	
	print("reading tree") ## X
	treeDic = {}
	for line in open(treePath):
		line = line.rstrip()
		eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
		parent_num, maxL = int(line.split('\t')[3]), int(line.split('\t')[4])
		Vec =  line.split('\t')[5] 
		#if not treeDic.has_key(eid):
		if eid not in treeDic:
		   treeDic[eid] = {}
		treeDic[eid][indexC] = {'parent':indexP, 'parent_num':parent_num, 'maxL':maxL, 'vec':Vec}   
	print("tree no:", len(treeDic))
	
	print("loading train set") 
	tree_train, word_train, index_train, y_train, parent_num_train, c = [], [], [], [], [], 0
	l1,l2,l3,l4 = 0,0,0,0
	for eid in open(trainPath):
		#if c > 8: break
		#eid = eid.rstrip()
		eid = eid.rstrip().split("\t")[0]
		#if not labelDic.has_key(eid): continue
		#if not treeDic.has_key(eid): continue 
		if eid not in labelDic: continue
		if eid not in treeDic: continue
		if len(treeDic[eid]) <= 0: 
		   #print labelDic[eid]
		   continue
		## 1. load label
		label = labelDic[eid]
		y, l1,l2,l3,l4 = loadLabel(label, l1, l2, l3, l4)
		y_train.append(y)
		## 2. construct tree
		#print eid
		x_word, x_index, tree, parent_num = constructTree(treeDic[eid])
		tree_train.append(tree)
		word_train.append(x_word)
		index_train.append(x_index)
		parent_num_train.append(parent_num)
		#print treeDic[eid]
		#print tree, child_num
		#exit(0)
		c += 1
	print(l1,l2,l3,l4)
	
	print("loading test set") 
	tree_test, word_test, index_test, parent_num_test, y_test, c = [], [], [], [], [], 0
	l1,l2,l3,l4 = 0,0,0,0
	for eid in open(testPath):
		#if c > 4: break
		#eid = eid.rstrip()
		eid = eid.rstrip().split("\t")[0]
		#if not labelDic.has_key(eid): continue
		#if not treeDic.has_key(eid): continue 
		if eid not in labelDic: continue
		if eid not in treeDic: continue 
		if len(treeDic[eid]) <= 0: 
		   #print labelDic[eid] 
		   continue        
		## 1. load label        
		label = labelDic[eid]
		y, l1,l2,l3,l4 = loadLabel(label, l1, l2, l3, l4)
		y_test.append(y)
		## 2. construct tree
		x_word, x_index, tree, parent_num = constructTree(treeDic[eid])
		tree_test.append(tree)
		word_test.append(x_word)  
		index_test.append(x_index) 
		parent_num_test.append(parent_num)
		c += 1
	print(l1,l2,l3,l4)
	print("train no:", len(tree_train), len(word_train), len(index_train),len(parent_num_train), len(y_train))
	print("test no:", len(tree_test), len(word_test), len(index_test), len(parent_num_test), len(y_test))
	#print("dim1 for 0:", len(tree_train[0]), len(word_train[0]), len(index_train[0]))
	#print("case 0:", tree_train[0][0], word_train[0][0], index_train[0][0], parent_num_train[0])
	#print index_train[0]
	#print word_train[0]
	#print tree_train[0]    
	#exit(0)
	return tree_train, word_train, index_train, parent_num_train, y_train, tree_test, word_test, index_test, parent_num_test, y_test

##################################### MAIN #####################################
if __name__ == "__main__":
	args = parse_args()
	
	unit = "TD_RvNN-" + args.obj + str(args.fold) + "-vol." + str(args.vocabulary_size) + args.tag
	#lossPath = "../loss/loss-" + unit + ".txt"
	#modelPath = "../param/param-" + unit + ".npz" 
	
	treePath   = "../resource/data.TD_RvNN.vol_" + str(args.vocabulary_size) + ".txt" 
	#trainPath  = "../nfold/RNNtrainSet_" + args.obj + str(args.fold) + "_tree.txt" 
	#testPath   = "../nfold/RNNtestSet_" + args.obj + str(args.fold) + "_tree.txt"
	trainPath  = "../resource/myData/{}/split_{}/train.txt".format(args.obj, args.fold)
	testPath   = "../resource/myData/{}/split_{}/test.txt".format(args.obj, args.fold)
	labelPath  = "../resource/" + args.obj + "_label_All.txt"
	resultPath = "../result/{}.txt".format(args.obj)

	if not os.path.isfile(resultPath):
		f_res = open(resultPath, "w")
		f_res.write("{:4s}\t{:6s}\t{:6s}\t{:6s}\t{:6s}\t{:6s}\t{:6s}\t{:6s}\t{:6s}\t{:6s}\t{:6s}\t{:6s}\t{:6s}\t{:6s}\t{:6s}\t{:6s}\t{:6s}\t{:6s}\t{:6s}\n".format(
			"Fold", "Acc." , "microF", 
			"Acc1", "Prec1", "Recll1", "F1",
			"Acc2", "Prec2", "Recll2", "F2",
			"Acc3", "Prec3", "Recll3", "F3",
			"Acc4", "Prec4", "Recll4", "F4"
		))
	else:
		f_res = open(resultPath, "a")

	print("\n*** {} Fold [{}] ***\n".format(args.obj, args.fold))

	## 1. load tree & word & index & label
	tree_train, word_train, index_train, parent_num_train, y_train, tree_test, word_test, index_test, parent_num_test, y_test = loadData()
	
	## 2. ini RNN model
	t0 = time.time()
	model = TD_RvNN.RvNN(args.vocabulary_size, args.hidden_dim, args.Nclass)
	t1 = time.time()
	print("Recursive model established,", (t1-t0)/60)
	
	#if os.path.isfile(modelPath):
	#   load_model_Recursive_gruEmb(modelPath, model)
	#   lr = 0.0001
	   
	######debug here######
	#print len(tree_test[121]), len(index_test[121]), len(word_test[121])
	#print tree_test[121]
	#exit(0)
	#loss, pred_y = model.train_step_up(word_test[121], index_test[121], tree_test[121], y_test[121], lr)
	#print loss, pred_y
	#exit(0)
	'''i=568
	loss, pred_y = model.train_step_up(word_train[i], index_train[i], parent_num_train[i], tree_train[i], y_train[i], lr)
	print loss, pred_y
	print len(tree_train[i]), len(word_train[i]), parent_num_train[i]
	print tree_train[i]
	print word_train[i]
	print 'final_state:',model._evaluate(word_train[i], index_train[i], parent_num_train[i], tree_train[i])
	tree_states=model._evaluate2(word_train[i], index_train[i], parent_num_train[i], tree_train[i])
	print 'tree_states:', tree_states
	print tree_states[-1:].mean(axis=0)
	tree_states_test=model._evaluate3(word_train[i], index_train[i], tree_train[i])
	print 'l:',len(tree_states_test)
	print 'lo:',tree_states_test[parent_num_train[i]:]'''
	######################
	
	## 3. looping SGD
	best_acc, best_res = 0, None
	losses_5, losses = [], []
	num_examples_seen = 0
	for epoch in range(args.Nepoch):
		## one SGD 
		indexs = [i for i in range(len(y_train))]
		#random.shuffle(indexs) 
		for i in tqdm(indexs, desc="train"):
			'''print i,":", len(tree_train[i])
			print tree_train[i]
			tree_state = model._state(word_train[i], index_train[i], child_num_train[i], tree_train[i])
			print len(tree_state)
			print tree_state
			evl = model._evaluate(word_train[i], index_train[i], child_num_train[i], tree_train[i])
			print len(evl) 
			print evl'''
			loss, pred_y = model.train_step_up(word_train[i], index_train[i], parent_num_train[i], tree_train[i], y_train[i], args.lr)
			#print loss, pred_y
			losses.append(np.round(loss,2))
			'''if math.isnan(loss):
			#   continue 
			   print loss, pred_y
			   print i
			   print len(tree_train[i]), len(word_train[i]), parent_num_train[i]
			   print tree_train[i]
			   print word_train[i]
			   print 'final_state:',model._evaluate(word_train[i], index_train[i], parent_num_train[i], tree_train[i])'''
			num_examples_seen += 1
		#print("epoch=%d: loss=%f" % (epoch, np.mean(losses)))
		print("Fold [{}], epoch={}, loss={}".format(args.fold, epoch, np.mean(losses)))
		#floss.write(str(time)+": epoch="+str(epoch)+" loss="+str(loss) +'\n')
		sys.stdout.flush()
		#print losses
		#exit(0)
		
		## cal loss & evaluate
		if epoch % 5 == 0:
			losses_5.append((num_examples_seen, np.mean(losses))) 
			time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
			print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, np.mean(losses)))
			#floss.write(str(time)+": epoch="+str(epoch)+" loss="+str(loss) +'\n') 
			#floss.flush()       
			sys.stdout.flush()
			prediction = []
			for j in tqdm(range(len(y_test)), desc="evaluate"):
				#print j
				prediction.append(model.predict_up(word_test[j], index_test[j], parent_num_test[j], tree_test[j]) )   
			res = evaluation_4class(prediction, y_test) 
			print("results:", res)
			#floss.write(str(res)+'\n')
			#floss.flush() 
			sys.stdout.flush()
			## Adjust the learning rate if loss increases
			if len(losses_5) > 1 and losses_5[-1][1] > losses_5[-2][1]:
			   args.lr = args.lr * 0.5   
			   print("Setting learning rate to %f" % args.lr)
			   #floss.write("Setting learning rate to:"+str(lr)+'\n')
			   #floss.flush() 
			   sys.stdout.flush()
			#save_model_Recursive_gruEmb(modelPath, model)

			## NEW: Record best metrics based on accuracy
			if res[1] > best_acc:
				best_acc = res[1]
				best_res = res

		#floss.flush()
		losses = []

	f_res.write("{:4s}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
		args.fold, res[ 1], res[ 3], 
		res[ 7], res[ 8], res[ 9], res[10], 
		res[12], res[13], res[14], res[15],
		res[17], res[18], res[19], res[20],
		res[22], res[23], res[24], res[25] 
	))
		
	#floss.close()    
