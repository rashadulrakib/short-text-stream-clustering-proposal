import json
from Document import Document
from textblob import TextBlob
#from txt_process_util import detectNPPhrase

class DocumentSet:

    def __init__(self, dataDir, wordToIdMap, wordList):
        self.D = 0  # The number of documents
        # self.clusterNoArray = []
        self.documents = []
        with open(dataDir) as input:
            line = input.readline()
            while line:
                self.D += 1
                obj = json.loads(line)
                text = obj['textCleaned']
                #process the text
                '''oldText= text 				
                text=detectNPPhrase(text)
                			  
                #end process the text'''
                clusterNo = obj['clusterNo']
                #print(oldText, ",", text, clusterNo) 				
                document = Document(text, clusterNo, -1, wordToIdMap, wordList, int(obj['Id'])) #rakib
                self.documents.append(document)
                line = input.readline()
        print("number of documents is ", self.D)