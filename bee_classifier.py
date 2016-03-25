from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import time
import numpy as np
import sklearn
from sklearn.neural_network import BernoulliRBM
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn import svm
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet,ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal




N = 15;
ds = ClassificationDataSet(N*N*3, 1,nb_classes=2)
path = "/media/sai/New Volume1/Practice/beeImages/bee_images/train/"
labels = pd.read_csv("/media/sai/New Volume1/Practice/beeImages/train_label.csv")
channels = ["b","g","r"];
train_X = [];
train_Y = []; 
for index,row in tqdm(labels.iterrows()):
	fname =  path + str(int(row['id'])) + ".jpg";
	image = cv2.imread(fname,1);
	features = [];
	image = cv2.resize(image, (N, N))
	for i,color in enumerate(channels):	
		hist = image[:,:,i]
		features += [hist];		
	features = np.hstack(features).flatten();
	ds.addSample(features,row['genus']);
tstdata_temp, trndata_temp = ds.splitWithProportion( 0.25 );
tstdata = ClassificationDataSet(N*N*3, 1, nb_classes=2)
for n in xrange(0, tstdata_temp.getLength()):
    tstdata.addSample( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )
trndata = ClassificationDataSet(N*N*3, 1, nb_classes=2)
for n in xrange(0, trndata_temp.getLength()):
    trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )
print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]
fnn = buildNetwork( trndata.indim,100, trndata.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01);
trainer.trainUntilConvergence(dataset=trndata,maxEpochs=100,verbose=True,continueEpochs=10,validationProportion=0.1)
print "training complete\n"
trnresult = sklearn.metrics.roc_auc_score(trndata['class'], trainer.testOnClassData(dataset=trndata), average='macro', sample_weight=None);
tstresult = sklearn.metrics.roc_auc_score(tstdata['class'], trainer.testOnClassData(dataset=tstdata ), average='macro',sample_weight=None);

print "epoch: %4d" % trainer.totalepochs, \
  "  train error: %5.2f%%" % trnresult, \
  "  test error: %5.2f%%" % tstresult

'''
train_X = np.vstack(train_X);
train_Y = np.hstack(train_Y);
(trainX, testX, trainY, testY) = train_test_split(train_X, train_Y, test_size = 0.2, random_state = 32)
params_to_try = {
    'C': [10**i for i in range(0, 2)],
    'gamma': [10**i for i in range(-7, -5)],
}

gs = GridSearchCV(svm.SVC(class_weight='auto', kernel='linear', probability=True),
                  param_grid=params_to_try,
                  cv=3,
                  scoring='roc_auc',
                  n_jobs=-1)

gs.fit(trainX, trainY)

print "Best parameters:", gs.best_params_
print "Best score:", gs.best_score_ 
'''

