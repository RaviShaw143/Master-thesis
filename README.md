Steps followed to get to the Reduced You Tube Features Set
* The given data set was organized in the following way:
    * There are in total  48 You Tube videos which were divided in to 280 video parts and each Video parts are annotated with either Positive, negative, or neutral sentiment.
* Used an audio cutter to obtain the 280 audio parts from 48 you tube videos
* Then used OPENSmile software to obtain the audio features (6553) of the audio parts and stored it in CSV format
* Then used the label column of SentimentAnnotations.csv file (provided in the You tube dataset) to assign the label for each audio parts, hence resulting in a file (AudioFeaturesWithLabel.csv)which has audio features of each segment and the sentiment associated with it.  
* Then Used Google Speech API to obtain the transcription of each Audio parts
* Then extracted the text feature using the obtained transcription of each audio parts from Google Speech API 
    * Used TFIDF with 2 grams for extracting the text features, resulted in 3940 features 
* Then created a file (textFeaturesWithLabel.csv) which has text features of each audio parts and the label associated with it.
* Then joined  AudioFeaturesWithLabel.csv  and  textFeaturesWithLabel.csv to create a file ( YouTubeDataFeaturesWithOutput.csv ), which has both the audio features and the text features and the label associated for YouTube audio parts. So in total after combining the features (audio and text), each audio data has 10493 features.
* To reduce the number of features used PCA (Principal Component Analysis) 
    * When used 95% variance the total no. of features were reduced to 210 features from 10493 features 