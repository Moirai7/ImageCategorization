import torchfile
#import cv2
import json
#import torch
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

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
	print 'load Data and Labels!'
	from sklearn import svm
	from sklearn import linear_model
	from sklearn.naive_bayes import GaussianNB
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.tree import DecisionTreeClassifier
	
	clfs = {'svm':svm.SVC(),'linear':linear_model.SGDClassifier(loss="hinge", penalty="l2"), 'guassian':GaussianNB(),'KNN':KNeighborsClassifier(),'Random':RandomForestClassifier(),'Decision':DecisionTreeClassifier()}

	for name,clf in clfs.items():
		clf.fit(trainData, trainLabels)
	
		if finetune == 2:
			joblib.dump(clf, 'result/'+types+'/digits_'+name+'_model_finetune.yml')
		else:
			joblib.dump(clf, 'result/'+types+'/digits_'+name+'_model.yml')
		print types+' '+str(finetune)+' '+name +' file saved!'
	

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
	
	names = ['svm','linear','guassian','KNN','Random','Decision']
	for name in names:
		if finetune == 2:
			clf = joblib.load('result/'+types+'/digits_'+name+'_model_finetune.yml')
		else:
			clf = joblib.load('result/'+types+'/digits_'+name+'_model.yml')
		results = clf.predict(testData)#testLabels)
		print name + ' result'
		print 'Mean_square: '+str(mean_squared_error(testLabels,results))
		try:
			print 'accuracy_score: '+str(accuracy_score(testLabels,results))
		except:
			pass

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
