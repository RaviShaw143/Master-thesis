import pandas as pd
import os


#path to the compare2016_Feat.csv file
audioFeaturePathCompare16 = os.path.abspath(os.path.join("Desktop","Code","FinalAudioAndTextFiles","Audio","AudioFeaturesFile","CsvFiles", "compare2016_Feat.csv"))

#path to the IS09_Feat.csv file
audioFeaturePathIS09 = os.path.abspath(os.path.join("Desktop","Code","FinalAudioAndTextFiles","Audio","AudioFeaturesFile","CsvFiles", "IS09_Feat.csv"))

#path to the emo_large_Feat.csv file
audioFeaturePathEmo_large = os.path.abspath(os.path.join("Desktop","Code","FinalAudioAndTextFiles","Audio","AudioFeaturesFile","CsvFiles", "emo_large_Feat.csv"))


#path to the tfidfTextFeatures.csv file
textFeaturePath = os.path.abspath(os.path.join("Desktop","Code","FinalAudioAndTextFiles","Text","TextFeatureFiles", "tfidfTextFeatures.csv"))

#reads the csv files of audio features and text features and create dataframe objects
compareDf = pd.read_csv(audioFeaturePathCompare16)
ISO9Df= pd.read_csv(audioFeaturePathIS09)
emoLargeDf= pd.read_csv(audioFeaturePathEmo_large)
tfidfTextDF= pd.read_csv(textFeaturePath)

#joins the textFeatures columns header with compare2016 features headers 
audioFeatureHeadersCompare = list(compareDf.iloc[:,0:6373])
textFeatureHeaders = list(tfidfTextDF.iloc[:,0:3941])
featuresHeadersCompare = audioFeatureHeadersCompare + textFeatureHeaders

#joins the textFeatures columns header with IS09 features headers 
audioFeatureHeadersIS09 = list(ISO9Df.iloc[:,0:384])
featuresHeadersIS09 = audioFeatureHeadersIS09 + textFeatureHeaders

#joins the textFeatures columns header with emoLarge features headers 
audioFeatureHeadersEmoLarge = list(emoLargeDf.iloc[:,0:6553])
featuresHeadersEmoLarge = audioFeatureHeadersEmoLarge + textFeatureHeaders

#extracts the text features values from the text feature file
textFeatures = tfidfTextDF.iloc[0:269,0:3941].values

#extracts the audio features values from each of the audio feature file
audioFeaturesCompare = compareDf.iloc[0:269,0:6373].values
audioFeaturesIS09 = ISO9Df.iloc[0:269,0:384].values
audioFeaturesEmoLarge = emoLargeDf.iloc[0:269,0:6553].values

completeFeaturesCompare = []
completeFeaturesIS09 = []
completeFeaturesEmoLarge = []


def joinAudioAndTextFeatures(audioFeatures, featureFrom):
    i=0
    while i < len(audioFeatures):
        feature = list(audioFeatures[i]) + list(textFeatures[i])
        i=i+1
        if featureFrom == "compare":
            completeFeaturesCompare.append(feature)
        if featureFrom == "IS09":
            completeFeaturesIS09.append(feature)
        if featureFrom == "emoLarge":    
            completeFeaturesEmoLarge.append(feature)
            
            
        

 
joinAudioAndTextFeatures(audioFeaturesCompare,"compare")       
joinAudioAndTextFeatures(audioFeaturesIS09,"IS09")       
joinAudioAndTextFeatures(audioFeaturesEmoLarge,"emoLarge")       

compareJoinedFeatruesDf = pd.DataFrame(data = completeFeaturesCompare, columns =featuresHeadersCompare)
IS09JoinedFeatruesDf = pd.DataFrame(data = completeFeaturesIS09, columns =featuresHeadersIS09)
emoLargeJoinedFeatruesDf = pd.DataFrame(data = completeFeaturesEmoLarge, columns =featuresHeadersEmoLarge)

compareJoinedFeatruesDf.to_csv(r'C:\Users\ravi\Desktop\Code\FinalAudioAndTextFiles\compareJoinedFeatures.csv', encoding='utf-8', index=False)
IS09JoinedFeatruesDf.to_csv(r'C:\Users\ravi\Desktop\Code\FinalAudioAndTextFiles\IS09JoinedFeatures.csv', encoding='utf-8', index=False)
emoLargeJoinedFeatruesDf.to_csv(r'C:\Users\ravi\Desktop\Code\FinalAudioAndTextFiles\emoLargeJoinedFeatures.csv', encoding='utf-8', index=False)

