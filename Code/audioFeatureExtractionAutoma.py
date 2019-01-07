from natsort import natsorted
import os

path = os.path.abspath(os.path.join("FinalAudioAndTextFiles","TestOutputOrder"))
audioFiles = []
lists = []   
def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            #print(filename)
            lists.append(filename)
            filename = os.path.join(root, filename)
            audioFiles.append(filename)

readFiles(path)
#print(audioFiles)
print(lists)
print("After")

print(natsorted(lists, key=lambda y: y.lower()))
print(audioFiles)
print("After")
audioFiles = natsorted(audioFiles, key=lambda y: y.lower())
print(audioFiles)

emo_large_config = "SMILExtract_Release -C config/emo_large.conf -I "

IS09_config = "SMILExtract_Release -C config/IS09_emotion.conf"

Compare_2016_config = "SMILExtract_Release -C config/ComParE_2016.conf"

pathToOutputFile = " -O C:\\Users\\ravi\\Desktop\\Code\\FinalAudioAndTextFiles\\AudioLabelsFile\\emo_large_Feat.arff"
listofCommand = []

for filePath in audioFiles:
    command = emo_large_config + filePath + pathToOutputFile
    listofCommand.append(command)
        
for commands in listofCommand:
        os.system(listofCommand)