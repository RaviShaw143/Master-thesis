import pandas as pd
import os
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


featuresFilePathCompare = os.path.abspath(os.path.join("Desktop", "Code", "FinalAudioAndTextFiles", "compareJoinedFeatures.csv"))
featuresFilePathIS09 = os.path.abspath(os.path.join("Desktop", "Code", "FinalAudioAndTextFiles", "IS09JoinedFeatures.csv"))
featuresFilePathEmoLarge = os.path.abspath(os.path.join("Desktop", "Code", "FinalAudioAndTextFiles", "emoLargeJoinedFeatures.csv"))

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
def getSVCModelResults():
    
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.01, 0.001, 0.0001],
                        'C': [0.001,0.01,0.1,1, 10, 100, 1000]},
                        {'kernel': ['linear'],'gamma': [0.01, 0.001, 0.0001], 'C': [0.001,0.01,0.1,1, 10, 100, 1000]}]
    #findBestEstimator(SVC(),tuned_parameters,train_data_Com, test_data_Com, train_lbl_Com, test_lbl_Com)
    
    tuned_parameters_lin = [{'dual': [False],'C': [1, 10, 100, 1000],'multi_class' :['ovr']},
                        {'dual': [False],'C': [1, 10, 100, 1000],'multi_class' :['crammer_singer']},
                        {'dual': [True],'C': [1, 10, 100, 1000],'multi_class' :['crammer_singer']}]
    
    tuned_parameters_nuSVC = [{'nu':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], 'gamma': [0.01, 0.001, 0.0001]}]
    findBestEstimator(SVC(),tuned_parameters,train_data_Com, test_data_Com, train_lbl_Com, test_lbl_Com)
    findBestEstimator(LinearSVC(),tuned_parameters_lin,train_data_Com, test_data_Com, train_lbl_Com, test_lbl_Com)
    findBestEstimator(NuSVC(),tuned_parameters_nuSVC,train_data_Com, test_data_Com, train_lbl_Com, test_lbl_Com)
 

train_data_Com, test_data_Com, train_lbl_Com, test_lbl_Com = splitDataAfterPCA(inputCompare, target)
train_data_IS, test_data_IS, train_lbl_IS, test_lbl_IS = splitDataAfterPCA(inputIS09, target)
train_data_Emo, test_data_Emo, train_lbl_Emo, test_lbl_Emo = splitDataAfterPCA(inputEmoLarge, target)
   
getSVCModelResults()


        
    
    
