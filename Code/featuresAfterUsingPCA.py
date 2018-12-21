import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

featuresFilePath = os.path.abspath(os.path.join("Desktop", "Code", "LabelFiles", "YouTubeDataFeaturesWithOutput.csv"))

features = pd.read_csv(featuresFilePath)
print (features.shape)
output  = features["output"][:].values
input = features.iloc[:,0:10493].values
print("Number of features before using PCA:" + str(len(input[0])))

#standardize the data
scaler = StandardScaler()
scaler.fit(input)
input = scaler.transform(input)

#choose the minimum number of principal components such that 95% of the variance is retained
pca = PCA(.95)

#fittng PCA on the total number of Features
pca.fit(input)

#applying the mapping to input set
input = pca.transform(input)
print("Number of features after using PCA:" + str(len(input[0])))












