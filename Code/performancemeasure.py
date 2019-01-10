# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit,  StratifiedKFold
from sklearn.ensemble import RandomForestRegressor,ExtraTreesClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import math
from sklearn.linear_model import LogisticRegression
#from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras import metrics
#import tensorflow as tf
#from tensorflow.keras import backend as K
#from tensorflow.keras.layers.core import Activation




featuresFilePathCompare = os.path.abspath(os.path.join("Desktop", "Code", "FinalAudioAndTextFiles", "compareJoinedFeatures.csv"))
featuresFilePathIS09 = os.path.abspath(os.path.join("Desktop", "Code", "FinalAudioAndTextFiles", "IS09JoinedFeatures.csv"))
featuresFilePathEmoLarge = os.path.abspath(os.path.join("Desktop", "Code", "FinalAudioAndTextFiles", "emoLargeJoinedFeatures.csv"))
audioFeatutresPath = os.path.abspath(os.path.join("Desktop", "Code", "FinalAudioAndTextFiles","Audio","AudioFeaturesFile","CsvFiles", "compare2016_Feat.csv"))
textFeaturesPath = os.path.abspath(os.path.join("Desktop", "Code", "FinalAudioAndTextFiles","Text","TextFeatureFiles", "tfidfTextFeatures.csv"))
# C:\Users\ravi\Desktop\Code\FinalAudioAndTextFiles\Text\TextFeatureFiles
compareAudioFeatures = pd.read_csv(audioFeatutresPath)
compareTextFeatures = pd.read_csv(textFeaturesPath)

target  = compareAudioFeatures["sentiment"][:].values
inputAudioCompare = compareAudioFeatures.iloc[:,0:6373].values
inputTextCompare = compareTextFeatures.iloc[:,0:3940].values

#inputAudioCompare inputTextCompare

compareFeatures = pd.read_csv(featuresFilePathCompare)
IS09Features = pd.read_csv(featuresFilePathIS09)
emoLargeFeatures = pd.read_csv(featuresFilePathEmoLarge)

target  = compareFeatures["sentiment"][:].values
inputCompare = compareFeatures.iloc[:,0:10313].values
inputIS09 = IS09Features.iloc[:,0:4324].values
inputEmoLarge = emoLargeFeatures.iloc[:,0:10493].values
       
        
def splitDataAfterPCA(input, output):
    # test_size: what proportion of original data is used for test set
    train_data, test_data, train_lbl, test_lbl = train_test_split(input, output, test_size=0.20, random_state=0)
    
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(train_data)

    # Apply transform to both the training set and the test set.    
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    
    pca = PCA(.95) 
    # Fit on training set only.
    pca.fit(train_data)
    
    # Apply transform to both the training set and the test set.
    train_data = pca.transform(train_data)
    test_data = pca.transform(test_data)
    return train_data, test_data, train_lbl, test_lbl 
    
def PCAWithoutSplitting(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    pca = PCA(.95) 
    pca.fit(X)
    X = pca.transform(X)
    return X

def cross_score(X,y):
    train_data_comp_PCA = PCAWithoutSplitting(X)
    score = cross_val_score(SVC(), train_data_comp_PCA, y, cv=10)
    print(score)
    
def trainModelAndGetScore(model,train_data, test_data, train_lbl, test_lbl):
    model.fit(train_data,train_lbl)
    result = model.score(test_data,test_lbl)
    pred = model.predict(test_data) 
    acc = accuracy_score(test_lbl, pred)
    f1 = f1_score(test_lbl, pred, average = "micro")
    prec = precision_score(test_lbl, pred,average = "micro")
    rec = recall_score(test_lbl, pred,average = "micro")
    result = {'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
    return result

def findBestEstimator(model, tuned_parameters, X_train, X_test, y_train, y_test):
    scores = ['precision', 'recall']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(model, tuned_parameters, cv=10,
                        scoring='%s_micro' % score)
        clf.fit(X_train, y_train)
    
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
       # print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        
        print()
    
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
    

#Uses SVC model to give the result
def getSVCModelResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com):
    
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.01, 0.001, 0.0001],
                        'C': [0.001,0.01,0.1,1, 10, 100, 1000]},
                        {'kernel': ['linear'],'gamma': [0.01, 0.001, 0.0001], 'C': [0.001,0.01,0.1,1, 10, 100, 1000]}]
    #findBestEstimator(SVC(),tuned_parameters,train_data_Com, test_data_Com, train_lbl_Com, test_lbl_Com)
    
    tuned_parameters_lin = [{'dual': [False],'C': [1, 10, 100, 1000],'multi_class' :['ovr']},
                        {'dual': [False],'C': [1, 10, 100, 1000],'multi_class' :['crammer_singer']},
                        {'dual': [True],'C': [1, 10, 100, 1000],'multi_class' :['crammer_singer']}]
    
    tuned_parameters_nuSVC = [{'nu':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], 'gamma': [0.01, 0.001, 0.0001]}]
    
    findBestEstimator(SVC(),tuned_parameters,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)
    findBestEstimator(LinearSVC(),tuned_parameters_lin,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)
    findBestEstimator(NuSVC(),tuned_parameters_nuSVC,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)
 
def getRandomForestClassifierResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com):    
    tune_params_RF = [{'n_estimators': [10,20,30,40,50,60],'max_features': [None],'max_depth':[None],'min_samples_split':[2], 'random_state':[0],'n_jobs':[1]}]
    tune_param_DClf = [{'max_depth':[3],'min_samples_leaf':[1]}]
    
    findBestEstimator(RandomForestClassifier(),tune_params_RF,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)
    findBestEstimator(ExtraTreesClassifier(),tune_params_RF,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)
    findBestEstimator(DecisionTreeClassifier(),tune_param_DClf,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)


def getLogisticRegressionResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com):        
    tune_params_LR2 = [{'penalty':['l2'], 'solver':['newton-cg','lbfgs','sag'], 'max_iter':[2000],'multi_class':['multinomial','ovr']}]
    tune_params_LR1 = [{'penalty':['l1'], 'solver':['liblinear','saga'],'max_iter':[2000],'multi_class':['ovr']}]
    findBestEstimator(LogisticRegression(),tune_params_LR1,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com )
    findBestEstimator(LogisticRegression(),tune_params_LR2,train_X_Com, test_X_Com, train_Y_Com, test_Y_Com )

def getSupervisedModelResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com):
    getSVCModelResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)
    getRandomForestClassifierResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)
    getLogisticRegressionResults(train_X_Com, test_X_Com, train_Y_Com, test_Y_Com)

train_X_Com, test_X_Com, train_Y_Com, test_Y_Com = splitDataAfterPCA(inputCompare, target)

#inputAudioCompare inputTextCompare

train_X_AudioCom, test_X_AudioCom, train_Y_AudioCom, test_Y_AudioCom = splitDataAfterPCA(inputAudioCompare, target)
train_X_TextCom, test_X_TextCom, train_Y_TextCom, test_Y_TextCom = splitDataAfterPCA(inputTextCompare, target)

test_X = test_X_Com
test_Y = test_Y_Com
train_data_IS, test_data_IS, train_lbl_IS, test_lbl_IS = splitDataAfterPCA(inputIS09, target)
train_data_Emo, test_data_Emo, train_lbl_Emo, test_lbl_Emo = splitDataAfterPCA(inputEmoLarge, target)
input_feat = (train_X_Com.shape[0])
input_feat = input_feat + 1
print(test_X_Com.shape)

#getSVCModelResults(train_X_TextCom, test_X_TextCom, train_Y_TextCom, test_Y_TextCom)

#getRandomForestClassifierResults(train_X_AudioCom, test_X_AudioCom, train_Y_AudioCom, test_Y_AudioCom)
#getRandomForestClassifierResults(train_X_TextCom, test_X_TextCom, train_Y_TextCom, test_Y_TextCom)
#getLogisticRegressionResults(train_X_AudioCom, test_X_AudioCom, train_Y_AudioCom, test_Y_AudioCom)
#getLogisticRegressionResults(train_X_TextCom, test_X_TextCom, train_Y_TextCom, test_Y_TextCom)
#train_X_AudioCom, test_X_AudioCom, train_Y_AudioCom, test_Y_AudioCom
#train_X_TextCom, test_X_TextCom, train_Y_TextCom, test_Y_TextCom
#dividing further the training set to train and validation set
trainX, valX, trainY, valY = train_test_split(train_X_AudioCom, train_Y_AudioCom, test_size=0.20, random_state=0)
trainX = np.expand_dims(trainX, 2)
valX = np.expand_dims(valX, 2)

trainY = to_categorical(trainY)
valY = to_categorical(valY)




#testing set to evaluate the model
test_X_AudioCom = np.expand_dims(test_X_Com, 2)
test_Y_AudioCom = to_categorical(test_Y_Com)

def as_keras_metric(method):
    import functools
    from tensorflow.keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

#precision = as_keras_metric(tf.metrics.precision)
#recall = as_keras_metric(tf.metrics.recall)

def LSTModel(trainX, trainY, valX, valY, test_X_Com, test_Y_Com):
    model = Sequential()
    #model.add(Embedding(input_feat, 108))
    model.add(LSTM(187, dropout=0.2, return_sequences=True, recurrent_dropout=0.2, input_shape = (165,1)))
    model.add(LSTM(94, dropout=0.2, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(48, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='sigmoid'))    
    
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy' ])    
        
    model.fit(trainX, trainY,
            batch_size=80,
            epochs=3,
            verbose=2,
            validation_data=(valX, valY))
    
    score, acc = model.evaluate(test_X_Com, test_Y_Com,
                                batch_size=32,
                                verbose=2)
    
    print('Test score:', score)
    print('Test accuracy:', acc)


LSTModel(trainX, trainY, valX, valY, test_X_Com, test_Y_Com)

