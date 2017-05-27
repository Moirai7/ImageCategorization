import torchfile
import cv2
import json
import torch
import numpy as np
from sklearn.externals import joblib

def classification(types,finetune):
	if finetune == 0 :
		#trainData = torch.load('feats/'+types+'/train_features_labels.t7')
		trainData = np.load('feats/'+types+'/train_features_labels.npy')
		trainLabels = np.load('feats/train_labels.npy')
	elif finetune == 1 :
		trainData = torchfile.load('feats/'+types+'/train_features_labels_'+types+'.t7')
		trainLabels = trainData.labels
		trainData = trainData.features
	elif finetune == 2:
		trainData = torchfile.load('feats/'+types+'/train_features_labels_'+types+'_finetune.t7')
		trainLabels = trainData.labels
		trainData = trainData.features
	
	'''
	#_images = json.load(open('ic-data/train_label.json'))	
	#trainLabels = [v for k,v in _images.items()]
	
	trainLabels = np.array(trainLabels)
	svm = cv2.SVM()
	svm_params = dict( kernel_type = cv2.SVM_RBF,
                    svm_type = cv2.SVM_C_SVC,
                    C=0.01, gamma=1.383 )
	print 'start training!'
	svm.train(trainData, trainLabels, params=svm_params)
	'''

	from sklearn import svm
	clf = svm.SVC()
	clf.fit(trainData, trainLabels)
	
	if finetune == 2:
		joblib.dump(clf, 'result/'+types+'/digits_svm_model_finetune.yml')
	else:
		joblib.dump(clf, 'result/'+types+'/digits_svm_model.yml')
	print types+' '+str(finetune)+' svm file saved!'
	

def test(types,finetune):
	if finetune == 0 :
                testData = np.load('feats/'+types+'/test_features_labels.npy')
		testLabels = np.load('feats/test_labels.npy')
        elif finetune == 1 :
		testData = torchfile.load('feats/'+types+'/test_features_labels_'+types+'.t7')
                testLabels = testData.labels
                testData = testData.features
        elif finetune == 2:
		testData = torchfile.load('feats/'+types+'/test_features_labels_'+types+'_finetune.t7')
		testLabels = testData.labels
                testData = testData.features
	
	if finetune == 2:
		clf = joblib.load('result/'+types+'/digits_svm_model_finetune.yml')
	else:
		clf = joblib.load('result/'+types+'/digits_svm_model.yml')
	results = clf.score(testData,testLabels)
	print results

classification('hog',0)
test('hog',0)
classification('sift',0)
test('sift',0)
classification('vgg',1)
test('vgg',1)
classification('vgg',2)
test('vgg',2)
classification('resnet',1)
test('resnet',1)
classification('resnet',2)
test('resnet',2)
