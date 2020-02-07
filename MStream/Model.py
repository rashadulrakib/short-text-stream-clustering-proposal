import random
from random import randint
from itertools import chain 
import os
import time
import json
import copy
import numpy
from collections import Counter
#from CacheEmbeddings import getWordEmbedding

from operator import add
from ClassifyUsingSimilarity import classifyBySimilaritySingle
#from outliers_detection_connectedComp import removeOutlierConnectedComponentLexical
#from outliers_detection_connectedComp import removeOutlierConnectedComponentLexicalIndex
from general_util import change_pred_label
from general_util import findIndexByItems
from general_util import extrcatLargeClusterItems
from general_util import findMinMaxLabel
from general_util import print_by_group
from general_util import extractBySingleIndex
from general_util import maxSim_Count_lex
from general_util import maxSim_Count_semantic
from general_util import productLexSemanticSims
from general_util import findMaxKeyAndValue
from general_util import calculateFarCloseDist

from numpy import array


from groupTxt_ByClass import groupItemsBySingleKeyIndex
from groupTxt_ByClass import groupTxtByClass

from txt_process_util import combineDocsToSingle
from txt_process_util import commonWordSims
from txt_process_util import commonWordSims_clusterGroup
from txt_process_util import semanticSims
from txt_process_util import processTxtRemoveStopWordTokenized_wordArr
from txt_process_util import RemoveHighClusterEntropyWordsIndex
from txt_process_util import stem_text
from txt_process_util import getScikitLearn_StopWords
from txt_process_util import processTextsRemoveStopWordTokenized
from txt_process_util import processTxtRemoveStopWordTokenized
from txt_process_util import getDocFreq

from compute_util import computeTextSimCommonWord
from compute_util import compute_sim_value
from compute_util import computeClose_Far_vec

from groupTxt_ByClass import groupTxtByClass_Txtindex
from cluster_file_connected_component import clusterByConnectedComponentIndex
from cluster_file_connected_component import clusterByConnectedComponentWordCooccurIndex
from cluster_file_word_embedding import clusterByWordEmbeddingIntelligent
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from sent_vecgenerator import generate_sent_vecs_toktextdata
from sent_vecgenerator import generate_weighted_sent_vecs_toktextdata
from sent_vecgenerator import generate_sent_vecs_toktextdata_DCT
#from sent_vecgenerator import generate_sent_vecs_toktextdata_autoencoder

import sys
from print_cluster_evaluation import appendResultFile
#from outlier_detection_sd import detect_outlier_sd_vec_auto
from outlier_detection_XStream import detect_outlier_XStream
from outliers_detection_others import detect_outlier_other

from datetime import datetime
from scipy.spatial.distance import cosine
import statistics
#from pyod.models.so_gaal import SO_GAAL
#from pyod.models.mo_gaal import MO_GAAL

#sys.path.append("clustering/")
#from clustering_embedding import clusteringDCT

from evaluation import Evaluate
from read_pred_true_text import ReadPredTrueText
from cluster_file_leadNonOverlapWords import clusterByLeadingOnOverlappingWords
from Chains import Chains
from sklearn.ensemble import IsolationForest
import sys
import itertools


threshold_fix = 1.1
threshold_init = 0

class Model:

    def __init__(self, K, Max_Batch, V, iterNum,alpha0, beta, dataset, ParametersStr, sampleNo, wordsInTopicNum, timefil):
        self.dataset = dataset
        self.ParametersStr = ParametersStr
        self.alpha0 = alpha0
        self.beta = beta
        self.K = K
        self.Kin = K
        self.V = V
        self.iterNum = iterNum
        self.beta0 = float(V) * float(beta)
        self.sampleNo = sampleNo
        self.wordsInTopicNum = copy.deepcopy(wordsInTopicNum)
        self.Max_Batch = Max_Batch  # Max number of batches we will consider
        self.phi_zv = []

        self.batchNum2tweetID = {} # batch num to tweet id
        self.batchNum = 1 # current batch number
        self.readTimeFil(timefil)
		
    def readTimeFil(self, timefil):
        try:
            with open(timefil) as timef:
                for line in timef:
                    buff = line.strip().split(' ')
                    if buff == ['']:
                        break
                    self.batchNum2tweetID[self.batchNum] = int(buff[1])
                    self.batchNum += 1
            self.batchNum = 1
        except:
            print("No timefil!")
        # print("There are", self.batchNum2tweetID.__len__(), "time points.\n\t", self.batchNum2tweetID)

    def getAveBatch(self, documentSet, AllBatchNum):
        self.batchNum2tweetID.clear()
        temp = self.D_All / AllBatchNum
        _ = 0
        count = 1
        for d in range(self.D_All):
            if _ < temp:
                _ += 1
                continue
            else:
                document = documentSet.documents[d]
                documentID = document.documentID
                self.batchNum2tweetID[count] = documentID
                count += 1
                _ = 0
        self.batchNum2tweetID[count] = -1

    '''
    improvements
    At initialization, V_current has another choices:
    set V_current is the number of words analyzed in the current batch and the number of words in previous batches,
        which means beta0s of each document are different at initialization.
    Before we process each batch, alpha will be recomputed by the function: alpha = alpha0 * docStored
    '''
    def outlierBySmallestGroupsIndex(self, newPred_OldPred_true_text_inds, batchDocs, maxPredLabel):
      outliersInCluster_Index=[]
      non_outliersInCluster_Index=[]
   
      np_arr=np.array(newPred_OldPred_true_text_inds)
      new_preds=np_arr[:,0].tolist()

      new_pred_dict=Counter(new_preds)
      #print("new_pred_dict="+str(new_pred_dict))	  
      maxKey = max(new_pred_dict, key=new_pred_dict.get)
      newPredMaxLabel= maxKey
  
      for newPred_OldPred_true_text_ind in newPred_OldPred_true_text_inds:
        newPredLabel=newPred_OldPred_true_text_ind[0]
        oldPredLabel=newPred_OldPred_true_text_ind[1]
        trueLabel=newPred_OldPred_true_text_ind[2]
        word_arr=newPred_OldPred_true_text_ind[3]
        ind=newPred_OldPred_true_text_ind[4]	
        
        if str(newPredLabel)==str(newPredMaxLabel):          
          non_outliersInCluster_Index.append([oldPredLabel, trueLabel, word_arr, ind, oldPredLabel])
        else:
          #print("newPredLabel="+str(newPredLabel)+","+str(new_pred_dict[str(newPredLabel)]))
          if new_pred_dict[str(newPredLabel)]>1 or new_pred_dict[newPredLabel]>1:
            #print("update model for="+text+", old="+str(oldPredLabel)+", true="+str(trueLabel))
            chngPredLabel=int(str(newPredLabel)) + int(str(maxPredLabel)) +1
            self.updateModelParametersForSingle(oldPredLabel, chngPredLabel, batchDocs[int(str(ind))])
            non_outliersInCluster_Index.append([str(chngPredLabel), trueLabel, word_arr, ind, oldPredLabel])			
          else:	  
            outliersInCluster_Index.append([oldPredLabel, trueLabel, word_arr, ind, oldPredLabel])
            #self.updateModelParametersForSingleDelete(oldPredLabel, batchDocs[int(str(ind))])			

      #print(outliersInCluster_Index)	  
      maxPredLabel=int(str(maxPredLabel))+int(str(newPredMaxLabel))+1
      return [outliersInCluster_Index, non_outliersInCluster_Index, maxPredLabel]

    def removeOutlierSimDistributionIndex(self,list_pred_true_text_ind,batchDocs,maxPredLabel, wordVectorsDic):
      sd_avgItemsInCluster=0
      sd_out_pred_true_text_ind_prevPreds=[]
      sd_non_out_pred_true_text_ind_prevPreds=[]
      	  
      dic_tupple_class= groupItemsBySingleKeyIndex(list_pred_true_text_ind, 0)
 
      for label, cluster_pred_true_txt_inds in dic_tupple_class.items():
        if len(cluster_pred_true_txt_inds)==1:
          sd_out_pred_true_text_ind_prevPreds.extend(cluster_pred_true_txt_inds)
          continue	  
        elif len(cluster_pred_true_txt_inds)<3:
          sd_non_out_pred_true_text_ind_prevPreds.extend(cluster_pred_true_txt_inds)
          continue 		  
		
        nparr=np.array(cluster_pred_true_txt_inds) 
        preds=list(nparr[:,0])
        trues=list(nparr[:,1])
        texts=list(nparr[:,2])
        inds=list(nparr[:,3])
        X=generate_sent_vecs_toktextdata(texts, wordVectorsDic, 300)
        outlierLabels=detect_outlier_sd_vec_auto(X)		
        '''X_train = X
        print("before MO_GAAL")		
        print(Counter(outlierLabels))     		

        contamination=0.1
        # train lscp
        #_neighbors=max(int(len(X_train)/2),2)		
        #clf_name = 'LSCP'
        #detector_list = [LOF(n_neighbors=_neighbors), LOF(n_neighbors=1)]
        #clf = LSCP(detector_list, random_state=42)
        #clf.fit(X_train)
        clf = SO_GAAL(contamination=contamination)
        clf.fit(X_train)		

        # get the prediction labels and outlier scores of the training data
        y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        y_train_scores = clf.decision_scores_  # raw outlier scores

        y_train_pred=list(y_train_pred)
        new_out_list=[] 
        ind=-1		
        for y_pred in y_train_pred:
          ind=ind+1		
          if y_pred==0:
            new_out_list.append(1)
          else:
            new_out_list.append(-1)
            outlierLabels[ind]=-1			

        #outlierLabels=new_out_list
        print("after MO_GAAL")		
        print(Counter(outlierLabels))  '''       		
        if len(outlierLabels)==len(cluster_pred_true_txt_inds):
          for i in range(len(outlierLabels)):
            flag=outlierLabels[i]
            if flag==-1:
              sd_out_pred_true_text_ind_prevPreds.append(cluster_pred_true_txt_inds[i])
              oldPredLabel=cluster_pred_true_txt_inds[i][0]
              ind =cluster_pred_true_txt_inds[i][3]			  
              #self.updateModelParametersForSingleDelete(oldPredLabel, batchDocs[int(str(ind))])			  
            else:
              sd_non_out_pred_true_text_ind_prevPreds.append(cluster_pred_true_txt_inds[i]) 			
        else:
          print("sd outlier problem-label="+label)  		
		
      sd_avgItemsInCluster=len(list_pred_true_text_ind)/len(dic_tupple_class)

      '''final_sd_out_pred_true_text_ind_prevPreds=sd_out_pred_true_text_ind_prevPreds
      if len(sd_out_pred_true_text_ind_prevPreds)>1:
        final_sd_out_pred_true_text_ind_prevPreds=[]	  
        dic_tupple_class_sd_out=groupTxtByClass(sd_out_pred_true_text_ind_prevPreds, False)
        for label, cluster_pred_true_txt_inds in dic_tupple_class_sd_out.items():
          if len(cluster_pred_true_txt_inds)==1:
            final_sd_out_pred_true_text_ind_prevPreds.extend(cluster_pred_true_txt_inds)
          else:
            sd_non_out_pred_true_text_ind_prevPreds.extend(cluster_pred_true_txt_inds)
            #self.updateModelParametersForMultiAdd(cluster_pred_true_txt_inds) #to do'''			
      	  
      sd_out_pred_true_text_ind_prevPreds = [x + [x[0]] for x in sd_out_pred_true_text_ind_prevPreds]
      sd_non_out_pred_true_text_ind_prevPreds = [x + [x[0]] for x in sd_non_out_pred_true_text_ind_prevPreds]	   
      
      return [sd_out_pred_true_text_ind_prevPreds, sd_non_out_pred_true_text_ind_prevPreds, sd_avgItemsInCluster, maxPredLabel, len(dic_tupple_class)]		  

    '''def removeOutlierConnectedComponentLexicalIndex(self, listtuple_pred_true_text_ind, batchDocs, maxPredLabel):
      avgItemsInCluster=0
      outlier_pred_true_text_ind_prevPreds=[]
      non_outlier_pred_true_text_ind_prevPreds=[]
  
      dic_tupple_class=groupTxtByClass(listtuple_pred_true_text_ind, False) 
       #index is the original index in listtuple_pred_true_text

      for label, cluster_pred_true_txt_inds in dic_tupple_class.items():

        randBinary=randint(0,1)
        if len(cluster_pred_true_txt_inds)==1: #and randBinary==0:
          #self.updateModelParametersForSingleDelete(cluster_pred_true_txt_inds[0][0], batchDocs[int(str(cluster_pred_true_txt_inds[0][3]))])		  
          outlier_pred_true_text_ind_prevPreds.append([cluster_pred_true_txt_inds[0][0], cluster_pred_true_txt_inds[0][1], cluster_pred_true_txt_inds[0][2],cluster_pred_true_txt_inds[0][3],cluster_pred_true_txt_inds[0][0]])
          continue	  
		
        newList=[]
        i=-1
        for pred_true_txt_ind in cluster_pred_true_txt_inds:
          i=i+1
          newList.append(pred_true_txt_ind+[pred_true_txt_ind[0]])
        cluster_pred_true_txt_inds=newList      
    
        dic_new_tupple_class=groupItemsBySingleKeyIndex(cluster_pred_true_txt_inds,0)
		
        maxKey, maxValue= findMaxKeyAndValue(dic_new_tupple_class)
        if maxValue<6:
          non_outlier_pred_true_text_ind_prevPreds.extend(cluster_pred_true_txt_inds)
          self.updateModelParametersList(cluster_pred_true_txt_inds, batchDocs)		  
          continue
		  
        for key, items in dic_new_tupple_class.items():
          if key==maxKey:
            continue
          non_outlier_pred_true_text_ind_prevPreds.extend(dic_new_tupple_class[key])
          self.updateModelParametersList(items, batchDocs)		  
            
	
        non_outliersInCluster_Index_maxGroup= dic_new_tupple_class[maxKey]
        nparr=np.array(non_outliersInCluster_Index_maxGroup) 
        preds=list(nparr[:,0])
        trues=list(nparr[:,1])	
        texts=list(nparr[:,2])
        inds=list(nparr[:,3])
        prevPreds=list(nparr[:,4])		

        #chngPredLabel=int(str(newPredLabel)) + int(str(maxPredLabel)) +1 
        clusterLabels, totalDocsClustered=clusterByLeadingOnOverlappingWords(texts)
        #change labels of all, if outlier , remove from model . if not update model	        
        maxChangeLabel=-1
        listbyLeadWord_pred_true_text_ind_prevPred=[]
        for i in range(len(texts)):
          chngPredLabel=int(str(clusterLabels[i]))+int(str(maxPredLabel))+1
          if maxChangeLabel<chngPredLabel:
            maxChangeLabel=chngPredLabel
          listbyLeadWord_pred_true_text_ind_prevPred.append([str(chngPredLabel), trues[i], texts[i], inds[i], prevPreds[i]])			
          				
        maxPredLabel=maxChangeLabel+1
        	
        dic_new_lead_tupple_class=groupItemsBySingleKeyIndex(listbyLeadWord_pred_true_text_ind_prevPred,0)
        for key, items in dic_new_lead_tupple_class.items():
          if len(items)==1:
            outlier_pred_true_text_ind_prevPreds.extend(items)
          else:
            non_outlier_pred_true_text_ind_prevPreds.extend(items)
            self.updateModelParametersList(items, batchDocs)		  
		
      avgItemsInCluster=len(listtuple_pred_true_text_ind)/len(dic_tupple_class) 
   
      return [outlier_pred_true_text_ind_prevPreds, non_outlier_pred_true_text_ind_prevPreds, avgItemsInCluster, maxPredLabel, len(dic_tupple_class)] ''' 
	  
    def removeOutlierOtherIndex(self, list_pred_true_text_ind, batchDocs, maxPredLabel, wordVectorsDic) :
      avgItemsInCluster=0
      outlier_pred_true_text_ind_prevPreds=[]
      non_outlier_pred_true_text_ind_prevPreds=[]	
	  
      dic_tupple_class= groupItemsBySingleKeyIndex(list_pred_true_text_ind, 0)	

      for label, cluster_pred_true_txt_inds in dic_tupple_class.items():
        if len(cluster_pred_true_txt_inds)==1:
          outlier_pred_true_text_ind_prevPreds.extend(cluster_pred_true_txt_inds)
          continue	  
        elif len(cluster_pred_true_txt_inds)<3:
          non_outlier_pred_true_text_ind_prevPreds.extend(cluster_pred_true_txt_inds)
          continue 		  
		
        nparr=np.array(cluster_pred_true_txt_inds) 
        preds=list(nparr[:,0])
        trues=list(nparr[:,1])
        texts=list(nparr[:,2])
        inds=list(nparr[:,3])
        X=generate_sent_vecs_toktextdata(texts, wordVectorsDic, 300)
        outlierLabels=detect_outlier_other(X)		
          		
        if len(outlierLabels)==len(cluster_pred_true_txt_inds):
          for i in range(len(outlierLabels)):
            flag=outlierLabels[i]
            if flag==-1:
              outlier_pred_true_text_ind_prevPreds.append(cluster_pred_true_txt_inds[i])
              #oldPredLabel=cluster_pred_true_txt_inds[i][0]
              #ind =cluster_pred_true_txt_inds[i][3]			  
              #self.updateModelParametersForSingleDelete(oldPredLabel, batchDocs[int(str(ind))])			  
            else:
              non_outlier_pred_true_text_ind_prevPreds.append(cluster_pred_true_txt_inds[i]) 			
        else:
          print("Xoutlier problem-label="+label)  		
		
      sd_avgItemsInCluster=len(list_pred_true_text_ind)/len(dic_tupple_class)
 
      outlier_pred_true_text_ind_prevPreds = [x + [x[0]] for x in outlier_pred_true_text_ind_prevPreds]
      non_outlier_pred_true_text_ind_prevPreds = [x + [x[0]] for x in non_outlier_pred_true_text_ind_prevPreds]	   
 
      return [outlier_pred_true_text_ind_prevPreds, non_outlier_pred_true_text_ind_prevPreds, avgItemsInCluster, maxPredLabel, len(dic_tupple_class)]	
  
    def removeOutlierXStreamIndex(self, list_pred_true_text_ind, batchDocs, maxPredLabel, wordVectorsDic):  
      avgItemsInCluster=0
      outlier_pred_true_text_ind_prevPreds=[]
      non_outlier_pred_true_text_ind_prevPreds=[]	
	  
      dic_tupple_class= groupItemsBySingleKeyIndex(list_pred_true_text_ind, 0)	

      for label, cluster_pred_true_txt_inds in dic_tupple_class.items():
        if len(cluster_pred_true_txt_inds)==1:
          outlier_pred_true_text_ind_prevPreds.extend(cluster_pred_true_txt_inds)
          continue	  
        elif len(cluster_pred_true_txt_inds)<3:
          non_outlier_pred_true_text_ind_prevPreds.extend(cluster_pred_true_txt_inds)
          continue 		  
		
        nparr=np.array(cluster_pred_true_txt_inds) 
        preds=list(nparr[:,0])
        trues=list(nparr[:,1])
        texts=list(nparr[:,2])
        inds=list(nparr[:,3])
        X=generate_sent_vecs_toktextdata(texts, wordVectorsDic, 300)
        outlierLabels=detect_outlier_XStream(X)		
          		
        if len(outlierLabels)==len(cluster_pred_true_txt_inds):
          for i in range(len(outlierLabels)):
            flag=outlierLabels[i]
            if flag==-1:
              outlier_pred_true_text_ind_prevPreds.append(cluster_pred_true_txt_inds[i])
              #oldPredLabel=cluster_pred_true_txt_inds[i][0]
              #ind =cluster_pred_true_txt_inds[i][3]			  
              #self.updateModelParametersForSingleDelete(oldPredLabel, batchDocs[int(str(ind))])			  
            else:
              non_outlier_pred_true_text_ind_prevPreds.append(cluster_pred_true_txt_inds[i]) 			
        else:
          print("Xoutlier problem-label="+label)  		
		
      sd_avgItemsInCluster=len(list_pred_true_text_ind)/len(dic_tupple_class)
 
      outlier_pred_true_text_ind_prevPreds = [x + [x[0]] for x in outlier_pred_true_text_ind_prevPreds]
      non_outlier_pred_true_text_ind_prevPreds = [x + [x[0]] for x in non_outlier_pred_true_text_ind_prevPreds]	   
 
      return [outlier_pred_true_text_ind_prevPreds, non_outlier_pred_true_text_ind_prevPreds, avgItemsInCluster, maxPredLabel, len(dic_tupple_class)]
  
    def removeOutlierConnectedComponentLexicalIndex(self, listtuple_pred_true_text_ind, batchDocs, maxPredLabel):
      avgItemsInCluster=0
      outlier_pred_true_text_ind_prevPreds=[]
      non_outlier_pred_true_text_ind_prevPreds=[]
       	  
      dic_tupple_class=groupItemsBySingleKeyIndex(listtuple_pred_true_text_ind, 0) 
       #index is the original index in listtuple_pred_true_text

      for label, cluster_pred_true_txt_inds in dic_tupple_class.items():

        #randBinary=randint(0,1)
        if len(cluster_pred_true_txt_inds)==1: #and randBinary==0:
          #self.updateModelParametersForSingleDelete(cluster_pred_true_txt_inds[0][0], batchDocs[int(str(cluster_pred_true_txt_inds[0][3]))])		  
          outlier_pred_true_text_ind_prevPreds.append([cluster_pred_true_txt_inds[0][0], cluster_pred_true_txt_inds[0][1], cluster_pred_true_txt_inds[0][2],cluster_pred_true_txt_inds[0][3],cluster_pred_true_txt_inds[0][0]])
          continue	  
		
        #_components,newPred_OldPred_true_text_inds=clusterByConnectedComponentIndex(cluster_pred_true_txt_inds)
        _components,newPred_OldPred_true_text_inds=clusterByConnectedComponentWordCooccurIndex(cluster_pred_true_txt_inds)		
        #find the smaller components and use it as outliers	
        outliersInCluster_Index,non_outliersInCluster_Index, maxPredLabel=self.outlierBySmallestGroupsIndex(newPred_OldPred_true_text_inds, batchDocs, maxPredLabel)
	
        outlier_pred_true_text_ind_prevPreds.extend(outliersInCluster_Index)
        non_outlier_pred_true_text_ind_prevPreds.extend(non_outliersInCluster_Index) #un comment if the following do not work well
        #apply clusterByLeadingOnOverlappingWords to find more outliers
        '''dic_new_tupple_class=groupItemsBySingleKeyIndex(non_outliersInCluster_Index,0)
        
        maxKey, maxValue= findMaxKeyAndValue(dic_new_tupple_class)
        if maxValue<6:
          non_outlier_pred_true_text_ind_prevPreds.extend(non_outliersInCluster_Index)		
          continue
 
        #Assign non max groups into non outliers
        for key, items in dic_new_tupple_class.items():
          if key==maxKey:
            continue
          non_outlier_pred_true_text_ind_prevPreds.extend(dic_new_tupple_class[key])			
            
	
        non_outliersInCluster_Index_maxGroup= dic_new_tupple_class[maxKey]
        nparr=np.array(non_outliersInCluster_Index_maxGroup) 
        preds=list(nparr[:,0])
        trues=list(nparr[:,1])	
        texts=list(nparr[:,2])
        inds=list(nparr[:,3])
        prevPreds=list(nparr[:,4])		

        #chngPredLabel=int(str(newPredLabel)) + int(str(maxPredLabel)) +1 
        clusterLabels, totalDocsClustered=clusterByLeadingOnOverlappingWords(texts)
        #change labels of all, if outlier , remove from model . if not update model	        
        maxChangeLabel=-1
        listbyLeadWord_pred_true_text_ind_prevPred=[]
        for i in range(len(texts)):
          chngPredLabel=int(str(clusterLabels[i]))+int(str(maxPredLabel))+1
          if maxChangeLabel<chngPredLabel:
            maxChangeLabel=chngPredLabel
          listbyLeadWord_pred_true_text_ind_prevPred.append([str(chngPredLabel), trues[i], texts[i], inds[i], prevPreds[i]])			
          				
        maxPredLabel=maxChangeLabel+1
        	
        dic_new_lead_tupple_class=groupItemsBySingleKeyIndex(listbyLeadWord_pred_true_text_ind_prevPred,0)
        for key, items in dic_new_lead_tupple_class.items():
          if len(items)==1:
            outlier_pred_true_text_ind_prevPreds.extend(items)
          else:
            non_outlier_pred_true_text_ind_prevPreds.extend(items)
            #self.updateModelParametersList(items, batchDocs)'''			
          			

      avgItemsInCluster=len(listtuple_pred_true_text_ind)/len(dic_tupple_class) 
   
      return [outlier_pred_true_text_ind_prevPreds, non_outlier_pred_true_text_ind_prevPreds, avgItemsInCluster, maxPredLabel, len(dic_tupple_class)]
    
    def removeOutlierConnectedComponentLexical(self, listtuple_pred_true_text, batchDocs, maxPredLabel):
      outlier_pred_true_text_ind_prevPreds=[]
      non_outlier_pred_true_text_ind_prevPreds=[]
      avgItemsInCluster=0

      dic_tupple_class=groupTxtByClass(listtuple_pred_true_text, False)

      for label, pred_true_txts_inds in dic_tupple_class.items():
        _components,newPred_OldPred_true_text_inds=clusterByConnectedComponentIndex(pred_true_txts_inds)
        #find the smallest components and use it as outliers	
        outliersInCluster_Index,non_outliersInCluster_Index, maxPredLabel=self.outlierBySmallestGroupsIndex(newPred_OldPred_true_text_inds, batchDocs, maxPredLabel)
	
        outlier_pred_true_text_ind_prevPreds.extend(outliersInCluster_Index)
        non_outlier_pred_true_text_ind_prevPreds.extend(non_outliersInCluster_Index) 	
	
      avgItemsInCluster=len(listtuple_pred_true_text)/len(dic_tupple_class)
    
      return [outlier_pred_true_text_ind_prevPreds, non_outlier_pred_true_text_ind_prevPreds, avgItemsInCluster, maxPredLabel, len(dic_tupple_class)]	
    
    def populateNonOutliers(self,non_outlier_pred_true_text_ind_prevPreds):      
      self.flat_non_outs.extend(non_outlier_pred_true_text_ind_prevPreds)      

    def populateNonOutliersAfterAssign(self, pred_true_text_ind_prevPreds):
      for pred_true_text_ind_prevPred in pred_true_text_ind_prevPreds:
        pred=pred_true_text_ind_prevPred[0]
        self.docsPerCluster_nonoutlier_afterAssign.setdefault(pred, []).append(pred_true_text_ind_prevPred)		
     
    def populateOutliers(self,outlier_pred_true_text_ind_prevPreds):
      self.flat_outs.extend(outlier_pred_true_text_ind_prevPreds)


    def populateOutliersAfterAssign(self, pred_true_text_ind_prevPreds):
      for pred_true_text_ind_prevPred in pred_true_text_ind_prevPreds:
        pred=pred_true_text_ind_prevPred[0]
        self.docsPerCluster_outlier_afterAssign.setdefault(pred, []).append(pred_true_text_ind_prevPred)		

    def populateClusterCenters_AutoEncoder(self, pred_true_text_ind_prevPreds, documentSet, wordVectorsDic):
	  #get number of clusters
      #get the embedding of all sentences
      #dic_cluster_sentVectrors={'c1': [v1, v2]}
	
      dic_itemGroups=groupItemsBySingleKeyIndex(pred_true_text_ind_prevPreds,0)
      dic_ClusterGroups={}
      dic_cluster_noTexts={}
	  
      n_clusters=len(dic_itemGroups)
      g_nparr=np.array(pred_true_text_ind_prevPreds) 
      #g_preds=list(g_nparr[:,0])
      #g_trues=list(g_nparr[:,1])
      g_word_arr=list(g_nparr[:,2])
      #g_inds=list(g_nparr[:,3])	  
             	  

      for label, cluster_pred_true_txt_inds in dic_itemGroups.items():
        	  
        nparr=np.array(cluster_pred_true_txt_inds) 
        preds=list(nparr[:,0])
        trues=list(nparr[:,1])
        word_arr=list(nparr[:,2])
        inds=list(nparr[:,3])
        X=generate_sent_vecs_toktextdata_autoencoder(word_arr, wordVectorsDic, 300)

        merged = list(itertools.chain.from_iterable(word_arr))
        dic_ClusterGroups[label]=[Counter(merged), len(merged)]
        totalTexts=len(cluster_pred_true_txt_inds)		
        dic_cluster_noTexts[label]=totalTexts        
		
        centVec=np.sum(np.array(X), axis=0)
        intlabel= int(str(label))
        if intlabel in self.centerVecs:
          centVec=centVec+np.array(self.centerVecs[intlabel])
        centVec=list(centVec) 		  
        self.centerVecs[intlabel]=centVec
	
        fardist, closdist=calculateFarCloseDist(np.true_divide(centVec, totalTexts+1), X)
		
        if intlabel in self.centerFarthestDist.keys():
          centfarD=self.centerFarthestDist[intlabel]
          fardist=max(fardist, centfarD)		  
        self.centerFarthestDist[intlabel]=fardist
		
        if intlabel in self.centerclosestDist.keys():
          centcloseD=self.centerclosestDist[intlabel]
          closdist=min(closdist, centcloseD)
		  
        self.centerclosestDist[intlabel]=closdist		
		
      #clusterSizes=list(map(len, dic_itemGroups.values()))		
      #self.avgClusterSize=statistics.mean(clusterSizes) 
      #self.stdClusterSize=statistics.stdev(clusterSizes)
      #print(clusterSizes)
      return [dic_ClusterGroups, dic_cluster_noTexts]  	
		
    def populateClusterCenters(self, pred_true_text_ind_prevPreds, documentSet, wordVectorsDic):
      dic_itemGroups=groupItemsBySingleKeyIndex(pred_true_text_ind_prevPreds,0)
      dic_ClusterGroups={}
      dic_cluster_noTexts={}	  

      for label, cluster_pred_true_txt_inds in dic_itemGroups.items():
        	  
        nparr=np.array(cluster_pred_true_txt_inds) 
        preds=list(nparr[:,0])
        trues=list(nparr[:,1])
        word_arr=list(nparr[:,2])
        inds=list(nparr[:,3])
        X=generate_sent_vecs_toktextdata(word_arr, wordVectorsDic, 300)

        merged = list(itertools.chain.from_iterable(word_arr))
        dic_ClusterGroups[label]=[Counter(merged), len(merged)]
        totalTexts=len(cluster_pred_true_txt_inds)		
        dic_cluster_noTexts[label]=totalTexts        
		
        centVec=np.sum(np.array(X), axis=0)
        intlabel= int(str(label))
        if intlabel in self.centerVecs:
          centVec=centVec+np.array(self.centerVecs[intlabel])
        centVec=list(centVec) 		  
        self.centerVecs[intlabel]=centVec
	
        fardist, closdist=calculateFarCloseDist(np.true_divide(centVec, totalTexts+1), X)
		
        if intlabel in self.centerFarthestDist.keys():
          centfarD=self.centerFarthestDist[intlabel]
          fardist=max(fardist, centfarD)		  
        self.centerFarthestDist[intlabel]=fardist
		
        if intlabel in self.centerclosestDist.keys():
          centcloseD=self.centerclosestDist[intlabel]
          closdist=min(closdist, centcloseD)
		  
        self.centerclosestDist[intlabel]=closdist		
		
      #clusterSizes=list(map(len, dic_itemGroups.values()))		
      #self.avgClusterSize=statistics.mean(clusterSizes) 
      #self.stdClusterSize=statistics.stdev(clusterSizes)
      #print(clusterSizes)
      return [dic_ClusterGroups, dic_cluster_noTexts]	  
     
    def populateBatchDocs_wordArr(self, skStopWords, documentSet):	  
      batchDocs=[]##contains only the docs in a batch			
      maxPredLabel=-1000#we use maxPredLabel to increment new labels (newPredLabel=maxPredLabel+1)
      pred_true_text_inds=[]
      #processTxtRemoveStopWordTokenized(batchDoc.text,skStopWords)  	
      i=-1
      for d in range(self.startDoc, self.currentDoc):
        i=i+1
        batchDoc=documentSet.documents[d]
        documentID = batchDoc.documentID
        self.docsPerBatch.setdefault(self.batchNum,[]).append(batchDoc)		
        cluster = self.z[documentID]
        self.docsPerCluster.setdefault(cluster,[]).append(documentSet.documents[d])		
        if maxPredLabel <cluster:
          maxPredLabel= cluster
        #print("batchDoc.text=",batchDoc.text)		  
        procText_wordArr=processTxtRemoveStopWordTokenized_wordArr(batchDoc.text,skStopWords)		  
        pred_true_text_inds.append([str(cluster), str(batchDoc.clusterNo), procText_wordArr , i])		  
        batchDocs.append(batchDoc)
  		
      print("maxPredLabel="+str(maxPredLabel))     
      return [batchDocs, pred_true_text_inds, maxPredLabel]
	 
      	
    def populateBatchDocs(self, documentSet):	  
      batchDocs=[]##contains only the docs in a batch			
      maxPredLabel=-1000#we use maxPredLabel to increment new labels (newPredLabel=maxPredLabel+1)
      pred_true_text_inds=[]
      #processTxtRemoveStopWordTokenized(batchDoc.text,skStopWords)  	
      i=-1
      skStopWords=getScikitLearn_StopWords()	  
      for d in range(self.startDoc, self.currentDoc):
        i=i+1
        batchDoc=documentSet.documents[d]
        documentID = batchDoc.documentID
        self.docsPerBatch.setdefault(self.batchNum,[]).append(batchDoc)		
        cluster = self.z[documentID]
        self.docsPerCluster.setdefault(cluster,[]).append(documentSet.documents[d])		
        if maxPredLabel <cluster:
          maxPredLabel= cluster
        procText=processTxtRemoveStopWordTokenized(batchDoc.text,skStopWords)		  
        pred_true_text_inds.append([str(cluster), str(batchDoc.clusterNo), procText , i])		  
        batchDocs.append(batchDoc)
  		
      print("maxPredLabel="+str(maxPredLabel))     
      return [batchDocs, pred_true_text_inds, maxPredLabel]   

    def assignOutlierToClusterSimilarity(self, outlier_pred_true_text_ind_prevPreds, cleaned_non_outlier_pred_true_txt_inds, batchDocs,maxPredLabel):
      
      dic_itemGroups=groupItemsBySingleKeyIndex(cleaned_non_outlier_pred_true_txt_inds, 0)

      i=-1	  
      for pred_true_text_ind_prevPred in outlier_pred_true_text_ind_prevPreds:
        pred=pred_true_text_ind_prevPred[0]
        true=pred_true_text_ind_prevPred[1]
        text=pred_true_text_ind_prevPred[2]
        ind=pred_true_text_ind_prevPred[3]
        prevPred=pred_true_text_ind_prevPred[4]
        cluster=self.getSuitableClusterIndexSimilarity(dic_itemGroups, text, pred, maxPredLabel)
		
        self.updateModelParametersForSingle(pred, cluster, batchDocs[int(str(ind))])		
        i=i+1
        outlier_pred_true_text_ind_prevPreds[i][0]=str(cluster)	
        if int(str(maxPredLabel)) < int(str(cluster)):
          maxPredLabel=cluster 	 		
	  
      return [outlier_pred_true_text_ind_prevPreds, cleaned_non_outlier_pred_true_txt_inds, maxPredLabel]	  
 
    def assignOutlierToClusterSimilarityEmbedding(self, outlier_pred_true_text_ind_prevPreds, dic_itemGroups, batchDocs,maxPredLabel, wordVectorsDic):
      
      i=-1	  
      for pred_true_text_ind_prevPred in outlier_pred_true_text_ind_prevPreds:
        i=i+1  	  
        pred=pred_true_text_ind_prevPred[0]
        true=pred_true_text_ind_prevPred[1]
        text=pred_true_text_ind_prevPred[2]
        ind=pred_true_text_ind_prevPred[3]
        prevPred=pred_true_text_ind_prevPred[4]
        
        cluster=self.getSuitableClusterIndexSimilarityEmbedding(dic_itemGroups, text, pred, maxPredLabel, wordVectorsDic)
        self.updateModelParametersForSingle(pred, cluster, batchDocs[int(str(ind))])
       		
        outlier_pred_true_text_ind_prevPreds[i][0]=str(cluster)	
        if int(str(maxPredLabel)) < int(str(cluster)):
          maxPredLabel=cluster
      return [outlier_pred_true_text_ind_prevPreds,maxPredLabel]
   
    def assignOutlierToClusterEmbeddingCenterCalculated(self, flat_outs, documentSet, wordVectorsDic, dic_ClusterGroups, maxPredLabel, batchDocs, dic_cluster_noTexts):
      
      new_outs=[]	  
      #dic_itemGroups=groupItemsBySingleKeyIndex(flat_non_outs, 0)
	 
      '''lkeys=list(dic_itemGroups.keys())
      skeys=list(self.centerVecs.keys())	  
      
      print(len(lkeys)==len(skeys))
      #print(lkeys, skeys)'''	 
	  
      	  
      i=-1	  
      for pred_true_text_ind_prevPred in flat_outs:
        i=i+1  	  
        pred=pred_true_text_ind_prevPred[0]
        true=pred_true_text_ind_prevPred[1]
        word_arr=pred_true_text_ind_prevPred[2]
        ind=pred_true_text_ind_prevPred[3]
        prevPred=pred_true_text_ind_prevPred[4]
        X=generate_sent_vecs_toktextdata([word_arr], wordVectorsDic, 300)
        text_Vec=X[0]

        t11=datetime.now()
        dic_lex_Sim_CommonWords, maxPredLabel_lex, maxSim_lex, maxCommon_lex, minSim_lex=commonWordSims_clusterGroup(word_arr, dic_ClusterGroups)
        t12=datetime.now()	  
        t_diff = t12-t11
        print("batch", self.batchNum,"time diff millisecs-out-dic_lex_Sim_CommonWords=",t_diff.microseconds/1000 )

        t11=datetime.now()
        dic_semanticSims, maxPredLabel_Semantic, maxSim_Semantic, minSim_semantic=semanticSims(text_Vec, self.centerVecs, dic_cluster_noTexts)
        t12=datetime.now()	  
        t_diff = t12-t11
        print("batch", self.batchNum,"time diff millisecs-out-dic_semanticSims=",t_diff.microseconds/1000)		

        '''t11=datetime.now()
        maxPredLabel_lex, maxSim_lex, maxCommon_lex ,maxPredLabel_Semantic, maxSim_Semantic, maxSim_product, maxPredLabel_product, minSim_semantic=productLexSemanticSims(dic_lex_Sim_CommonWords, dic_semanticSims)
        t12=datetime.now()	  
        t_diff = t12-t11
        print("batch", self.batchNum,"time diff millisecs-out-productLexSemanticSims=",t_diff.microseconds/1000)'''	

        '''maxPredLabel_lex, maxSim_lex, maxCommon_lex=maxSim_Count_lex(dic_lex_Sim_CommonWords)
        maxPredLabel_Semantic, maxSim_Semantic=maxSim_Count_semantic(dic_semanticSims)'''
        
        '''if maxSim_Semantic>0.5 and maxCommon_lex>1:
          pred_true_text_ind_prevPred[0]=maxPredLabel_Semantic	  
          new_outs.append(pred_true_text_ind_prevPred)
          self.updateModelParametersList([pred_true_text_ind_prevPred], batchDocs)		  
        #elif maxSim_lex>0 and maxCommon_lex>1:
        #  pred_true_text_ind_prevPred[0]=maxPredLabel_lex	  
        #  new_outs.append(pred_true_text_ind_prevPred)		
        #  self.updateModelParametersList([pred_true_text_ind_prevPred], batchDocs)		  
        else:
          maxPredLabel=int(str(maxPredLabel))+1	
          pred_true_text_ind_prevPred[0]=str(maxPredLabel)		  
          new_outs.append(pred_true_text_ind_prevPred)'''

        #also better		  
        '''intCenterLabel=int(str(maxPredLabel_Semantic))
        if intCenterLabel in self.centerclosestDist.keys():          
          close_dist=self.centerclosestDist[intCenterLabel]
          minSim_semantic_dist=1-maxSim_Semantic
          if minSim_semantic_dist<=close_dist:
            pred_true_text_ind_prevPred[0]=str(maxPredLabel_Semantic)	  
            new_outs.append(pred_true_text_ind_prevPred)
            self.updateModelParametersList([pred_true_text_ind_prevPred], batchDocs)
          else:
            maxPredLabel=int(str(maxPredLabel))+1	
            pred_true_text_ind_prevPred[0]=str(maxPredLabel)		  
            new_outs.append(pred_true_text_ind_prevPred) 
        else:		
            maxPredLabel=int(str(maxPredLabel))+1	
            pred_true_text_ind_prevPred[0]=str(maxPredLabel)		  
            new_outs.append(pred_true_text_ind_prevPred)'''        

        #better (main)
        '''intCenterLabel=int(str(maxPredLabel_Semantic))
        if intCenterLabel in self.centerFarthestDist.keys():          
          fardist=self.centerFarthestDist[intCenterLabel]
          maxSim_semantic_dist=1-minSim_semantic
          if maxSim_semantic_dist<=fardist:
            pred_true_text_ind_prevPred[0]=str(maxPredLabel_Semantic)	  
            new_outs.append(pred_true_text_ind_prevPred)
            self.updateModelParametersList([pred_true_text_ind_prevPred], batchDocs)
          else:
            maxPredLabel=int(str(maxPredLabel))+1	
            pred_true_text_ind_prevPred[0]=str(maxPredLabel)		  
            new_outs.append(pred_true_text_ind_prevPred) 
        else:		
            maxPredLabel=int(str(maxPredLabel))+1	
            pred_true_text_ind_prevPred[0]=str(maxPredLabel)		  
            new_outs.append(pred_true_text_ind_prevPred)'''			
            		  
        '''else:
          print("FALSE: if intCenterLabel in self.centerFarthestDist.keys():")		
          #working stable
          #if str(maxPredLabel_lex)==str(maxPredLabel_Semantic):		
          if maxCommon_lex>1 and str(maxPredLabel_lex)==str(maxPredLabel_Semantic):
            pred_true_text_ind_prevPred[0]=str(maxPredLabel_Semantic)
            #pred_true_text_ind_prevPred[4]=pred		  
            new_outs.append(pred_true_text_ind_prevPred)
            self.updateModelParametersList([pred_true_text_ind_prevPred], batchDocs)
          else:
            maxPredLabel=int(str(maxPredLabel))+1	
            pred_true_text_ind_prevPred[0]=str(maxPredLabel)		  
            new_outs.append(pred_true_text_ind_prevPred)
          #working stable end	'''	
        
		
           		  
           		  
        		
        #working stable
        #if str(maxPredLabel_lex)==str(maxPredLabel_Semantic):		
        if maxCommon_lex>1 and str(maxPredLabel_lex)==str(maxPredLabel_Semantic):
          pred_true_text_ind_prevPred[0]=str(maxPredLabel_Semantic)
          #pred_true_text_ind_prevPred[4]=pred		  
          new_outs.append(pred_true_text_ind_prevPred)
          self.updateModelParametersList([pred_true_text_ind_prevPred], batchDocs)
        else:
          maxPredLabel=int(str(maxPredLabel))+1	
          pred_true_text_ind_prevPred[0]=str(maxPredLabel)		  
          new_outs.append(pred_true_text_ind_prevPred)
        #working stable end 		  
		

		  
        #print("maxPredLabel_lex, maxSim_lex, maxCommon_lex")		
        #print(maxPredLabel_lex, maxSim_lex, maxCommon_lex)
        #print(maxPredLabel_Semantic, maxSim_Semantic)
        #print(maxPredLabel_product, maxSim_product)		

      return [new_outs, maxPredLabel]
 
    def detectOutlierAndEnhance(self, documentSet):
      batchDocs, pred_true_texts, maxPredLabel=self.populateBatchDocs(documentSet)      

      #this method also adds some small connected components to new clusters and update model parameters 	  
      outlier_pred_true_text_ind_prevPreds, non_outlier_pred_true_text_ind_prevPreds, avgItemsInCluster, maxPredLabel, clusters=self.removeOutlierConnectedComponentLexicalIndex( pred_true_texts, batchDocs, maxPredLabel)	 
      
      #print("outlier_pred_true_text_inds")
      #print_by_group(outlier_pred_true_text_ind_prevPreds)

      #print("non_outlier_pred_true_text_inds")
      #print_by_group(non_outlier_pred_true_text_ind_prevPreds)

      #get cleaned text representations from non_outlier_pred_true_text_inds
      cleaned_non_outlier_pred_true_txt_ind_prevPreds=RemoveHighClusterEntropyWordsIndex(non_outlier_pred_true_text_ind_prevPreds)	  
      #assign each text in outlier_pred_true_text_inds to one of the clusters in cleaned_non_outlier_pred_true_txt_inds or	create a new cluster
      #update the Model based on assign to an existing cluster or new cluster	  
      	  
      '''#old code working 
      if len(outlier_pred_true_text_inds)<=0:
        return  	  
      np_arr=array(outlier_pred_true_text_inds)
      #print(np_arr)	  
      outlier_indecies=np_arr[:,3].tolist()
      outlier_pred_true_texts=np_arr[:,0:3].tolist()	  
	  #outlier_indecies=findIndexByItems(outlier_pred_true_texts, pred_true_texts) #do not use
      print("len(outlier_indecies), len(set(outlier_indecies))")
      print(len(outlier_indecies), len(set(outlier_indecies)))
      
      #assign new labels to the outliers    		
      outlier_newpred_true_txts=change_pred_label(outlier_pred_true_texts, maxPredLabel)

      #change the model.py variables
      self.updateModelParameters(outlier_pred_true_texts, outlier_newpred_true_txts, outlier_indecies, batchDocs)'''    
      #assign outlier to proper cluster by similarity
      #update the model for each text and remove empty clusters
      #outlier_pred_true_text_ind_prevPreds, cleaned_non_outlier_pred_true_txt_ind_prevPreds, maxPredLabel=self.assignOutlierToClusterSimilarity(outlier_pred_true_text_ind_prevPreds, cleaned_non_outlier_pred_true_txt_ind_prevPreds,batchDocs,maxPredLabel)

      #print("outlier_pred_true_text_ind_prevPreds")
      #print_by_group(outlier_pred_true_text_ind_prevPreds+)

      #print("all by group")
      #print_by_group(outlier_pred_true_text_ind_prevPreds+cleaned_non_outlier_pred_true_txt_ind_prevPreds)	  
      #self.updateModelParametersBySimilarity(outlier_pred_true_text_ind_prevPreds, cleaned_pred_true_txt_ind_prevPreds, batchDocs, maxPredLabel)	  
     
    def detectOutlierAndEnhanceByEmbedding(self, documentSet, wordVectorsDic):
      batchDocs, pred_true_text_inds, maxPredLabel=self.populateBatchDocs(documentSet)
	  
      outlier_pred_true_text_ind_prevPreds, non_outlier_pred_true_text_ind_prevPreds, avgItemsInCluster, maxPredLabel, all_pred_clusters=self.removeOutlierConnectedComponentLexicalIndex(pred_true_text_inds, batchDocs, maxPredLabel)
      print("outlier="+str(len(outlier_pred_true_text_ind_prevPreds))+", non-outlier="+str(len(non_outlier_pred_true_text_ind_prevPreds))+",=maxPredLabel="+str(maxPredLabel))
      #print("outlier_pred_true_text_ind_prevPreds")       
      #print_by_group(outlier_pred_true_text_ind_prevPreds)
      #print("non_outlier_pred_true_text_ind_prevPreds")       
      #print_by_group(non_outlier_pred_true_text_ind_prevPreds)	  
	  
      #dic_itemGroups=groupItemsBySingleKeyIndex(non_outlier_pred_true_text_ind_prevPreds,0)
	  
      #modified_all_outliers, maxPredLabel=self.assignOutlierToClusterSimilarityEmbedding(outlier_pred_true_text_ind_prevPreds, dic_itemGroups, batchDocs,maxPredLabel, wordVectorsDic)
	  
      '''cleaned_non_outlier_pred_true_txt_ind_prevPreds=RemoveHighClusterEntropyWordsIndex(non_outlier_pred_true_text_ind_prevPreds)
	  
      sd_out_pred_true_text_ind_prevPreds, sd_non_out_pred_true_text_ind_prevPreds, sd_avgItemsInCluster,maxPredLabel,sd_pred_clusters=self.removeOutlierSimDistributionIndex(cleaned_non_outlier_pred_true_txt_ind_prevPreds,batchDocs,maxPredLabel, wordVectorsDic)
      print("sd outlier="+str(len(sd_out_pred_true_text_ind_prevPreds))+", sd-non-outlier="+str(len(sd_non_out_pred_true_text_ind_prevPreds))+",=maxPredLabel="+str(maxPredLabel))
      print("sd_out_pred_true_text_ind_prevPreds")       
      print_by_group(sd_out_pred_true_text_ind_prevPreds)
      print("sd_non_out_pred_true_text_ind_prevPreds")
      print_by_group(sd_non_out_pred_true_text_ind_prevPreds)
        
      #print("before remove outlier")	
      #Evaluate(pred_true_texts)
      #print("after remove outlier")	
      #Evaluate(non_outlier_pred_true_text_ind_prevPreds)'''
      appendResultFile(non_outlier_pred_true_text_ind_prevPreds, 'result/mstr-enh')
      '''appendResultFile(sd_non_out_pred_true_text_ind_prevPreds, 'result/mstr-enh-sd')	  
      
      	  
      #print_by_group(pred_true_texts)	  
      #print("after remove outlier-connected component")
      #Evaluate(non_outlier_pred_true_text_ind_prevPreds)
      #print("#outlier_pred_true_text_ind_prevPreds")	  
      #print_by_group(outlier_pred_true_text_ind_prevPreds)
      #print("#non_outlier_pred_true_text_ind_prevPreds")	  
      #print_by_group(non_outlier_pred_true_text_ind_prevPreds)	  
	  
      #cleaned_non_outlier_pred_true_txt_ind_prevPreds=RemoveHighClusterEntropyWordsIndex(non_outlier_pred_true_text_ind_prevPreds)
	  
      #self.clusteringByEmbedding(cleaned_non_outlier_pred_true_txt_ind_prevPreds, wordVectorsDic, batchDocs, maxPredLabel, outlier_pred_true_text_ind_prevPreds)''' 

    #does not work well
    def detectOutlierForgetEnhanceByEmbeddingSimProduct_DUAL(self, documentSet, skStopWords, wordVectorsDic):
      batchDocs, pred_true_text_inds, maxPredLabel=self.populateBatchDocs_wordArr(skStopWords, documentSet)
      	
      '''if self.batchNum>2:	  
        targetbatchNo=self.batchNum-1
        print("to delete batch", self.batchNum, targetbatchNo)
        print(self.docsPerBatch.keys())	
        
        t11=datetime.now()		
        self.updateModelParametersList_DeleteDocsUpdateCenter(targetbatchNo, self.docsPerBatch[targetbatchNo], wordVectorsDic)  	  
        t12=datetime.now()	  
        t_diff = t12-t11
        print("batch", self.batchNum,"time diff secs-out-DeleteDocsUpdateCenter=",t_diff.seconds)'''	  
	  
      t11=datetime.now()	  
      outlier_pred_true_text_ind_prevPreds, non_outlier_pred_true_text_ind_prevPreds, avgItemsInCluster, maxPredLabel, all_pred_clusters=self.removeOutlierConnectedComponentLexicalIndex(pred_true_text_inds, batchDocs, maxPredLabel)
      print("outlier="+str(len(outlier_pred_true_text_ind_prevPreds))+", non-outlier="+str(len(non_outlier_pred_true_text_ind_prevPreds))+",=maxPredLabel="+str(maxPredLabel))
      t12=datetime.now()	  
      t_diff = t12-t11
      print("batch", self.batchNum,"time diff secs-out-Connected=",t_diff.seconds)	  
	  
      self.outsNumberPerBatch[self.batchNum]=len(non_outlier_pred_true_text_ind_prevPreds)	  

      t11=datetime.now()	      
      non_outlier_pred_true_text_ind_prevPreds=RemoveHighClusterEntropyWordsIndex(non_outlier_pred_true_text_ind_prevPreds)
      t12=datetime.now()	  
      t_diff = t12-t11
      print("batch", self.batchNum,"time diff secs-out-removeEntropyWords=",t_diff.seconds)		  

      self.populateOutliers(outlier_pred_true_text_ind_prevPreds)
      self.populateNonOutliers(non_outlier_pred_true_text_ind_prevPreds)

      flat_non_outs= self.flat_non_outs
      flat_outs= self.flat_outs

      print("outsPerCluster.values(), nonOuts", len(flat_outs), len(flat_non_outs))
	  
      #DUAL test purpuse	  
      clusterByWordEmbeddingIntelligent(flat_non_outs, wordVectorsDic)
	  
      dic_ClusterGroups, dic_cluster_noTexts=self.populateClusterCenters(flat_non_outs, documentSet, wordVectorsDic)	#need to use  
 
      if self.batchNum>0: #or self.batchNum==24:
        t11=datetime.now()		
        flat_outs=self.customGibbsSampling(flat_outs, documentSet, wordVectorsDic)
        t12=datetime.now()	  
        t_diff = t12-t11
        print("batch", self.batchNum,"time diff secs-out-customGibbsSampling=",t_diff.seconds)		
		
        t11=datetime.now()		
        flat_outs, maxPredLabel=self.assignOutlierToClusterEmbeddingCenterCalculated(flat_outs, documentSet, wordVectorsDic, dic_ClusterGroups, maxPredLabel, batchDocs, dic_cluster_noTexts)
        t12=datetime.now()	  
        t_diff = t12-t11
        print("batch", self.batchNum,"time diff secs-out-assignOutlier=",t_diff.seconds)		
		
        appendResultFile(flat_outs, 'result/mstr-enh')
        self.flat_outs.clear()

      appendResultFile(non_outlier_pred_true_text_ind_prevPreds, 'result/mstr-enh')
      #print("Evaluate-enhance-forget", self.batchNum)	  
      #Evaluate(non_outlier_pred_true_text_ind_prevPreds)  	
	
    def detectOutlierForgetEnhanceByEmbeddingSimProduct_ACL(self, documentSet, skStopWords, wordVectorsDic):
      batchDocs, pred_true_text_inds, maxPredLabel=self.populateBatchDocs_wordArr(skStopWords, documentSet)
      #pred_true_text_inds, text contains the word arr 	  
	  
      #obtain a batch no
      #delete the docs from model and update cluseter center
      if self.batchNum>2:	  
        targetbatchNo=self.batchNum-1
        print("to delete batch", self.batchNum, targetbatchNo)
        print(self.docsPerBatch.keys())	
        
        t11=datetime.now()		
        self.updateModelParametersList_DeleteDocsUpdateCenter(targetbatchNo, self.docsPerBatch[targetbatchNo], wordVectorsDic)  	  
        t12=datetime.now()	  
        t_diff = t12-t11
        print("batch", self.batchNum,"time diff secs-out-DeleteDocsUpdateCenter=",t_diff.seconds)	  
	  
      t11=datetime.now()	  
      outlier_pred_true_text_ind_prevPreds, non_outlier_pred_true_text_ind_prevPreds, avgItemsInCluster, maxPredLabel, all_pred_clusters=self.removeOutlierConnectedComponentLexicalIndex(pred_true_text_inds, batchDocs, maxPredLabel)
      print("outlier="+str(len(outlier_pred_true_text_ind_prevPreds))+", non-outlier="+str(len(non_outlier_pred_true_text_ind_prevPreds))+",=maxPredLabel="+str(maxPredLabel))
      t12=datetime.now()	  
      t_diff = t12-t11
      print("batch", self.batchNum,"time diff secs-out-Connected=",t_diff.seconds)	  
	  
      self.outsNumberPerBatch[self.batchNum]=len(non_outlier_pred_true_text_ind_prevPreds)	  

      t11=datetime.now()	      
      non_outlier_pred_true_text_ind_prevPreds=RemoveHighClusterEntropyWordsIndex(non_outlier_pred_true_text_ind_prevPreds)
      t12=datetime.now()	  
      t_diff = t12-t11
      print("batch", self.batchNum,"time diff secs-out-removeEntropyWords=",t_diff.seconds)		  

      self.populateOutliers(outlier_pred_true_text_ind_prevPreds)
      self.populateNonOutliers(non_outlier_pred_true_text_ind_prevPreds)

      flat_non_outs= self.flat_non_outs
      flat_outs= self.flat_outs

      print("outsPerCluster.values(), nonOuts", len(flat_outs), len(flat_non_outs))
	   	  
      dic_ClusterGroups, dic_cluster_noTexts=self.populateClusterCenters(flat_non_outs, documentSet, wordVectorsDic)	#need to use  
 
      if self.batchNum>0: #or self.batchNum==24:
        t11=datetime.now()		
        flat_outs=self.customGibbsSampling(flat_outs, documentSet, wordVectorsDic)
        t12=datetime.now()	  
        t_diff = t12-t11
        print("batch", self.batchNum,"time diff secs-out-customGibbsSampling=",t_diff.seconds)		
		
        t11=datetime.now()		
        flat_outs, maxPredLabel=self.assignOutlierToClusterEmbeddingCenterCalculated(flat_outs, documentSet, wordVectorsDic, dic_ClusterGroups, maxPredLabel, batchDocs, dic_cluster_noTexts)
        t12=datetime.now()	  
        t_diff = t12-t11
        print("batch", self.batchNum,"time diff secs-out-assignOutlier=",t_diff.seconds)		
		
        appendResultFile(flat_outs, 'result/mstr-enh')
        self.flat_outs.clear()

      appendResultFile(non_outlier_pred_true_text_ind_prevPreds, 'result/mstr-enh')
      #print("Evaluate-enhance-forget", self.batchNum)	  
      #Evaluate(non_outlier_pred_true_text_ind_prevPreds+flat_outs)  
	  
    def detectOutlierForgetEnhanceByEmbeddingSimProduct(self, documentSet, skStopWords, wordVectorsDic):
      batchDocs, pred_true_text_inds, maxPredLabel=self.populateBatchDocs_wordArr(skStopWords, documentSet)
      #pred_true_text_inds, text contains the word arr 	  
	  
      #obtain a batch no
      #delete the docs from model and update cluseter center
      if self.batchNum>2:	  
        targetbatchNo=self.batchNum-1
        print("to delete batch", self.batchNum, targetbatchNo)
        print(self.docsPerBatch.keys())	
        
        t11=datetime.now()		
        self.updateModelParametersList_DeleteDocsUpdateCenter(targetbatchNo, self.docsPerBatch[targetbatchNo], wordVectorsDic)  	  
        t12=datetime.now()	  
        t_diff = t12-t11
        print("batch", self.batchNum,"time diff secs-out-DeleteDocsUpdateCenter=",t_diff.seconds)	  
	  
      t11=datetime.now()	  
      outlier_pred_true_text_ind_prevPreds, non_outlier_pred_true_text_ind_prevPreds, avgItemsInCluster, maxPredLabel, all_pred_clusters=self.removeOutlierConnectedComponentLexicalIndex(pred_true_text_inds, batchDocs, maxPredLabel)
      print("outlier="+str(len(outlier_pred_true_text_ind_prevPreds))+", non-outlier="+str(len(non_outlier_pred_true_text_ind_prevPreds))+",=maxPredLabel="+str(maxPredLabel))
      t12=datetime.now()	  
      t_diff = t12-t11
      print("batch", self.batchNum,"time diff secs-out-Connected=",t_diff.seconds)	  
	  
      self.outsNumberPerBatch[self.batchNum]=len(non_outlier_pred_true_text_ind_prevPreds)	  

      t11=datetime.now()	      
      non_outlier_pred_true_text_ind_prevPreds=RemoveHighClusterEntropyWordsIndex(non_outlier_pred_true_text_ind_prevPreds)
      t12=datetime.now()	  
      t_diff = t12-t11
      print("batch", self.batchNum,"time diff secs-out-removeEntropyWords=",t_diff.seconds)		  

      self.populateOutliers(outlier_pred_true_text_ind_prevPreds)
      self.populateNonOutliers(non_outlier_pred_true_text_ind_prevPreds)

      flat_non_outs= self.flat_non_outs
      flat_outs= self.flat_outs

      print("outsPerCluster.values(), nonOuts", len(flat_outs), len(flat_non_outs))
	   	  
      dic_ClusterGroups=self.populateClusterCenters(flat_non_outs, documentSet, wordVectorsDic)	#need to use  
 
      '''if self.batchNum%8==0: #or self.batchNum==24:
      #if self.batchNum==24:	  
        print("outsPerCluster")		
        print_by_group(flat_outs)
		
        print("notOutsPerCluster")			
        print_by_group(flat_non_outs)
		
        flat_outs, maxPredLabel=self.assignOutlierToClusterEmbeddingCenterCalculated(flat_outs, documentSet, wordVectorsDic, flat_non_outs, maxPredLabel, batchDocs)
        appendResultFile(flat_outs, 'result/mstr-enh')
        self.outsPerCluster.clear()

        #self.updateModelParametersList(flat_outs, batchDocs)	'''	

      if self.batchNum>0: #or self.batchNum==24:
      #if self.batchNum==24:	  
        #print("outsPerCluster")		
        #print_by_group(flat_outs)
		
        #print("notOutsPerCluster")			
        #print_by_group(flat_non_outs)
        t11=datetime.now()		
        flat_outs=self.customGibbsSampling(flat_outs, documentSet, wordVectorsDic)
        t12=datetime.now()	  
        t_diff = t12-t11
        print("batch", self.batchNum,"time diff secs-out-customGibbsSampling=",t_diff.seconds)		
		
        t11=datetime.now()		
        flat_outs, maxPredLabel=self.assignOutlierToClusterEmbeddingCenterCalculated(flat_outs, documentSet, wordVectorsDic, dic_ClusterGroups, maxPredLabel, batchDocs)
        t12=datetime.now()	  
        t_diff = t12-t11
        print("batch", self.batchNum,"time diff secs-out-assignOutlier=",t_diff.seconds)		
		
        appendResultFile(flat_outs, 'result/mstr-enh')
        self.flat_outs.clear()

      		
      
      		  
	  
            	  
	  
      '''cleaned_non_outlier_pred_true_txt_ind_prevPreds=RemoveHighClusterEntropyWordsIndex(non_outlier_pred_true_text_ind_prevPreds)

      dic_itemGroups=groupItemsBySingleKeyIndex(cleaned_non_outlier_pred_true_txt_ind_prevPreds,0)

      for label, cluster_pred_true_txt_inds in dic_itemGroups.items():
        	  
        nparr=np.array(cluster_pred_true_txt_inds) 
        preds=list(nparr[:,0])
        trues=list(nparr[:,1])
        texts=list(nparr[:,2])
        inds=list(nparr[:,3])
        X=generate_sent_vecs_toktextdata(texts, wordVectorsDic, 300)

        centVec=np.sum(np.array(X), axis=0)
        intlabel= int(str(label))
        if intlabel in self.centerVecs:
          centVec=centVec+np.array(self.centerVecs[intlabel])
        centVec=list(centVec) 		  
        self.centerVecs[intlabel]=centVec
        clusterSizes=list(map(len, self.docsPerCluster.values()))		
        self.avgClusterSize=statistics.mean(clusterSizes) 
        self.stdClusterSize=statistics.stdev(clusterSizes)
		
      outlier_pred_true_text_ind_prevPreds=self.customGibbsSampling(outlier_pred_true_text_ind_prevPreds, documentSet, wordVectorsDic)
	  
      outlier_pred_true_text_ind_prevPreds, non_outlier_pred_true_text_ind_prevPreds, avgItemsInCluster, maxPredLabel, all_pred_clusters=self.removeOutlierConnectedComponentLexicalIndex(non_outlier_pred_true_text_ind_prevPreds+outlier_pred_true_text_ind_prevPreds, batchDocs, maxPredLabel)
      print("outlier="+str(len(outlier_pred_true_text_ind_prevPreds))+", non-outlier="+str(len(non_outlier_pred_true_text_ind_prevPreds))+",=maxPredLabel="+str(maxPredLabel))'''
	  
      appendResultFile(non_outlier_pred_true_text_ind_prevPreds, 'result/mstr-enh')
      #print("Evaluate-enhance-forget", self.batchNum)	  
      #Evaluate(non_outlier_pred_true_text_ind_prevPreds+flat_outs) 	  

    def detectOutlierAndEnhanceByEmbeddingSimProduct_COLING(self, documentSet, skStopWords, wordVectorsDic):
      batchDocs, pred_true_text_inds, maxPredLabel=self.populateBatchDocs_wordArr(skStopWords, documentSet)	
	  
      t11=datetime.now()
      outlier_pred_true_text_ind_prevPreds, non_outlier_pred_true_text_ind_prevPreds, avgItemsInCluster, maxPredLabel, all_pred_clusters=self.removeOutlierConnectedComponentLexicalIndex(pred_true_text_inds, batchDocs, maxPredLabel)

      print("outlier="+str(len(outlier_pred_true_text_ind_prevPreds))+", non-outlier="+str(len(non_outlier_pred_true_text_ind_prevPreds))+",=maxPredLabel="+str(maxPredLabel))

      t11=datetime.now()
      non_outlier_pred_true_text_ind_prevPreds=RemoveHighClusterEntropyWordsIndex(non_outlier_pred_true_text_ind_prevPreds)
      t12=datetime.now()	  
      t_diff = t12-t11
      print("batch", self.batchNum,"time diff secs-remove high entropy word=",t_diff.seconds)

      self.populateOutliers(outlier_pred_true_text_ind_prevPreds)
      self.populateNonOutliers(non_outlier_pred_true_text_ind_prevPreds)
      flat_non_outs=self.flat_non_outs
      flat_outs=self.flat_outs

      print("outsPerCluster.values(), nonOuts", len(flat_outs), len(flat_non_outs))
	  	 
      dic_ClusterGroups, dic_cluster_noTexts=self.populateClusterCenters_AutoEncoder(flat_non_outs, documentSet, wordVectorsDic)	  
      #dic_ClusterGroups[clauter1]= [{word1_freq=3, word2_freq=2}, 5]

      if self.batchNum>0:
        t11=datetime.now()
        flat_outs=self.customGibbsSampling(flat_outs, documentSet, wordVectorsDic)
        t12=datetime.now()	  
        t_diff = t12-t11
        print("batch", self.batchNum,"time diff secs-customGibbsSampling=",t_diff.seconds)    
	
        t11=datetime.now()		
        flat_outs, maxPredLabel=self.assignOutlierToClusterEmbeddingCenterCalculated(flat_outs, documentSet, wordVectorsDic, dic_ClusterGroups, maxPredLabel, batchDocs, dic_cluster_noTexts)
        t12=datetime.now()	  
        t_diff = t12-t11
        print("batch", self.batchNum,"time diff secs-assign outlier=",t_diff.seconds)		
        appendResultFile(flat_outs, 'result/mstr-enh')
        #self.outsPerCluster.clear()
        self.flat_outs.clear() 		

      appendResultFile(non_outlier_pred_true_text_ind_prevPreds, 'result/mstr-enh')
      #print("Evaluate-enhance", self.batchNum)	  
      #Evaluate(non_outlier_pred_true_text_ind_prevPreds+flat_outs)	 

    def detectOutlierAndEnhanceByEmbeddingSimProduct_Employee(self, documentSet, skStopWords, wordVectorsDic):
      batchDocs, pred_true_text_inds, maxPredLabel=self.populateBatchDocs_wordArr(skStopWords, documentSet)	
	  
      t11=datetime.now()
      outlier_pred_true_text_ind_prevPreds, non_outlier_pred_true_text_ind_prevPreds, avgItemsInCluster, maxPredLabel, all_pred_clusters=self.removeOutlierConnectedComponentLexicalIndex(pred_true_text_inds, batchDocs, maxPredLabel)

      print("outlier="+str(len(outlier_pred_true_text_ind_prevPreds))+", non-outlier="+str(len(non_outlier_pred_true_text_ind_prevPreds))+",=maxPredLabel="+str(maxPredLabel))

      t11=datetime.now()
      non_outlier_pred_true_text_ind_prevPreds=RemoveHighClusterEntropyWordsIndex(non_outlier_pred_true_text_ind_prevPreds)
      t12=datetime.now()	  
      t_diff = t12-t11
      print("batch", self.batchNum,"time diff secs-remove high entropy word=",t_diff.seconds)

      self.populateOutliers(outlier_pred_true_text_ind_prevPreds)
      self.populateNonOutliers(non_outlier_pred_true_text_ind_prevPreds)
      flat_non_outs=self.flat_non_outs
      flat_outs=self.flat_outs

      print("outsPerCluster.values(), nonOuts", len(flat_outs), len(flat_non_outs))
	   	  
      dic_ClusterGroups, dic_cluster_noTexts=self.populateClusterCenters(flat_non_outs, documentSet, wordVectorsDic)
      #dic_ClusterGroups[clauter1]= [{word1_freq=3, word2_freq=2}, 5]

      if self.batchNum>0:
        t11=datetime.now()
        flat_outs=self.customGibbsSampling(flat_outs, documentSet, wordVectorsDic)
        t12=datetime.now()	  
        t_diff = t12-t11
        print("batch", self.batchNum,"time diff secs-customGibbsSampling=",t_diff.seconds)    
	
        t11=datetime.now()		
        flat_outs, maxPredLabel=self.assignOutlierToClusterEmbeddingCenterCalculated(flat_outs, documentSet, wordVectorsDic, dic_ClusterGroups, maxPredLabel, batchDocs, dic_cluster_noTexts)
        t12=datetime.now()	  
        t_diff = t12-t11
        print("batch", self.batchNum,"time diff secs-assign outlier=",t_diff.seconds)		
        appendResultFile(flat_outs, 'result/mstr-enh')
        self.populateOutliersAfterAssign(flat_outs)		
        self.flat_outs.clear() 		

      appendResultFile(non_outlier_pred_true_text_ind_prevPreds, 'result/mstr-enh')
      self.populateNonOutliersAfterAssign(non_outlier_pred_true_text_ind_prevPreds)	    
      #print("Evaluate-enhance", self.batchNum)	  
      #Evaluate(non_outlier_pred_true_text_ind_prevPreds+flat_outs)	  
      		  
    def detectOutlierAndEnhanceByEmbeddingSimProduct_ACL(self, documentSet, skStopWords, wordVectorsDic):
      batchDocs, pred_true_text_inds, maxPredLabel=self.populateBatchDocs_wordArr(skStopWords, documentSet)	
	  
      t11=datetime.now()
      outlier_pred_true_text_ind_prevPreds, non_outlier_pred_true_text_ind_prevPreds, avgItemsInCluster, maxPredLabel, all_pred_clusters=self.removeOutlierConnectedComponentLexicalIndex(pred_true_text_inds, batchDocs, maxPredLabel)

      print("outlier="+str(len(outlier_pred_true_text_ind_prevPreds))+", non-outlier="+str(len(non_outlier_pred_true_text_ind_prevPreds))+",=maxPredLabel="+str(maxPredLabel))

      t11=datetime.now()
      non_outlier_pred_true_text_ind_prevPreds=RemoveHighClusterEntropyWordsIndex(non_outlier_pred_true_text_ind_prevPreds)
      t12=datetime.now()	  
      t_diff = t12-t11
      print("batch", self.batchNum,"time diff secs-remove high entropy word=",t_diff.seconds)

      self.populateOutliers(outlier_pred_true_text_ind_prevPreds)
      self.populateNonOutliers(non_outlier_pred_true_text_ind_prevPreds)
      flat_non_outs=self.flat_non_outs
      flat_outs=self.flat_outs

      print("outsPerCluster.values(), nonOuts", len(flat_outs), len(flat_non_outs))
	   	  
      dic_ClusterGroups, dic_cluster_noTexts=self.populateClusterCenters(flat_non_outs, documentSet, wordVectorsDic)
      #dic_ClusterGroups[clauter1]= [{word1_freq=3, word2_freq=2}, 5]

      if self.batchNum>0:
        t11=datetime.now()
        flat_outs=self.customGibbsSampling(flat_outs, documentSet, wordVectorsDic)
        t12=datetime.now()	  
        t_diff = t12-t11
        print("batch", self.batchNum,"time diff secs-customGibbsSampling=",t_diff.seconds)    
	
        t11=datetime.now()		
        flat_outs, maxPredLabel=self.assignOutlierToClusterEmbeddingCenterCalculated(flat_outs, documentSet, wordVectorsDic, dic_ClusterGroups, maxPredLabel, batchDocs, dic_cluster_noTexts)
        t12=datetime.now()	  
        t_diff = t12-t11
        print("batch", self.batchNum,"time diff secs-assign outlier=",t_diff.seconds)		
        appendResultFile(flat_outs, 'result/mstr-enh')
        #self.outsPerCluster.clear()
        self.flat_outs.clear() 		

      appendResultFile(non_outlier_pred_true_text_ind_prevPreds, 'result/mstr-enh')
      #print("Evaluate-enhance", self.batchNum)	  
      #Evaluate(non_outlier_pred_true_text_ind_prevPreds+flat_outs)	  
      	
	  
    def detectOutlierAndEnhanceByEmbeddingSimProduct(self, documentSet, skStopWords, wordVectorsDic):
      batchDocs, pred_true_text_inds, maxPredLabel=self.populateBatchDocs_wordArr(skStopWords, documentSet)

      t11=datetime.now()
      outlier_pred_true_text_ind_prevPreds, non_outlier_pred_true_text_ind_prevPreds, avgItemsInCluster, maxPredLabel, all_pred_clusters=self.removeOutlierConnectedComponentLexicalIndex(pred_true_text_inds, batchDocs, maxPredLabel)
	 
      t12=datetime.now()	  
      t_diff = t12-t11
      print("batch", self.batchNum,"time diff secs-out-connected=",t_diff.seconds)	 
      
      #outlier_pred_true_text_ind_prevPreds, non_outlier_pred_true_text_ind_prevPreds, avgItemsInCluster,maxPredLabel,all_pred_clusters=self.removeOutlierSimDistributionIndex(pred_true_text_inds,batchDocs,maxPredLabel, wordVectorsDic)
      #outlier_pred_true_text_ind_prevPreds, non_outlier_pred_true_text_ind_prevPreds, avgItemsInCluster, maxPredLabel, all_pred_clusters=self.removeOutlierXStreamIndex(pred_true_text_inds, batchDocs, maxPredLabel,wordVectorsDic)
      #outlier_pred_true_text_ind_prevPreds, non_outlier_pred_true_text_ind_prevPreds, avgItemsInCluster, maxPredLabel, all_pred_clusters=self.removeOutlierOtherIndex(pred_true_text_inds, batchDocs, maxPredLabel,wordVectorsDic)	  
      t12=datetime.now()	  
      t_diff = t12-t11
      print("batch", self.batchNum,"time diff secs-out-connected=",t_diff.seconds)	        	  
  
	  
      print("outlier="+str(len(outlier_pred_true_text_ind_prevPreds))+", non-outlier="+str(len(non_outlier_pred_true_text_ind_prevPreds))+",=maxPredLabel="+str(maxPredLabel))

      t11=datetime.now()
      non_outlier_pred_true_text_ind_prevPreds=RemoveHighClusterEntropyWordsIndex(non_outlier_pred_true_text_ind_prevPreds)
      t12=datetime.now()	  
      t_diff = t12-t11
      print("batch", self.batchNum,"time diff secs-remove high entropy word=",t_diff.seconds)	  

      self.populateOutliers(outlier_pred_true_text_ind_prevPreds)
      self.populateNonOutliers(non_outlier_pred_true_text_ind_prevPreds)
      flat_non_outs=self.flat_non_outs
      flat_outs=self.flat_outs
	  

      '''t11=datetime.now()
      flat_non_outs= list(chain.from_iterable(list(self.notOutsPerCluster.values())))
      flat_outs= list(chain.from_iterable(list(self.outsPerCluster.values())))
      t12=datetime.now()	  
      t_diff = t12-t11
      print("batch", self.batchNum,"time diff secs-flatten=",t_diff.seconds)	  '''

      print("outsPerCluster.values(), nonOuts", len(flat_outs), len(flat_non_outs))
	   	  
      dic_ClusterGroups=self.populateClusterCenters(flat_non_outs, documentSet, wordVectorsDic)	#need to use  
 
      if self.batchNum>0: #or self.batchNum==24:
      #if self.batchNum==24:	  
        #print("outsPerCluster")		
        #print_by_group(flat_outs)
		
        #print("notOutsPerCluster")			
        #print_by_group(flat_non_outs)
        t11=datetime.now()
        flat_outs=self.customGibbsSampling(flat_outs, documentSet, wordVectorsDic)
        t12=datetime.now()	  
        t_diff = t12-t11
        print("batch", self.batchNum,"time diff secs-customGibbsSampling=",t_diff.seconds)    
	
        t11=datetime.now()		
        flat_outs, maxPredLabel=self.assignOutlierToClusterEmbeddingCenterCalculated(flat_outs, documentSet, wordVectorsDic, dic_ClusterGroups, maxPredLabel, batchDocs)
        t12=datetime.now()	  
        t_diff = t12-t11
        print("batch", self.batchNum,"time diff secs-assign outlier=",t_diff.seconds)		
        appendResultFile(flat_outs, 'result/mstr-enh')
        #self.outsPerCluster.clear()
        self.flat_outs.clear() 		

      appendResultFile(non_outlier_pred_true_text_ind_prevPreds, 'result/mstr-enh')
      #print("Evaluate-enhance", self.batchNum)	  
      #Evaluate(non_outlier_pred_true_text_ind_prevPreds+flat_outs) 	  
   	 
    def clusteringByEmbedding(self, pred_true_txt_ind_prevPreds, wordVectorsDic, batchDocs, maxPredLabel, outliersPrevStage):

      pred_true_text_ind_prevPreds_to_cluster, pred_true_text_ind_prevPreds_to_not_cluster=extrcatLargeClusterItems(pred_true_txt_ind_prevPreds)

      minPredAll, maxPredAll, minTrueAll, maxTrueAll=findMinMaxLabel(pred_true_txt_ind_prevPreds+outliersPrevStage)	  
  
      all_pred_clusters=len(groupTxtByClass(pred_true_txt_ind_prevPreds, False))
      pred_clusters=len(groupTxtByClass(pred_true_text_ind_prevPreds_to_cluster, False))
      non_pred_clusters=len(groupTxtByClass(pred_true_text_ind_prevPreds_to_not_cluster, False))  
  
      print("#all pred clusters="+str(all_pred_clusters))  
      print("#pred_clusters="+str(pred_clusters))
      print("#not clusters="+str(non_pred_clusters))
      print("this clustering with embedding DCT")
      pred_clusters=int(all_pred_clusters/(len(pred_true_txt_ind_prevPreds)/len(pred_true_text_ind_prevPreds_to_cluster)))
	  #int(all_pred_clusters/3)#non_pred_clusters-pred_clusters #int(all_pred_clusters/2)
      print("#update clusters="+str(pred_clusters))  
 
      nparr=np.array(pred_true_text_ind_prevPreds_to_cluster) 
      preds=list(nparr[:,0])
      trues=list(nparr[:,1])
      texts=list(nparr[:,2])
      inds=list(nparr[:,3])
      prevPreds=list(nparr[:,4])  
    
      skStopWords=getScikitLearn_StopWords()
      texts= processTextsRemoveStopWordTokenized(texts, skStopWords)	
      
      X = generate_sent_vecs_toktextdata(texts, wordVectorsDic, 300)
      
      maxPredLabel=max(maxPredLabel, int(str(maxPredAll)))	  
      print("#spectral-avg")
      clustering = SpectralClustering(n_clusters=pred_clusters,assign_labels="discretize", random_state=0).fit(X)
      newOffsetLabels=list(np.array(clustering.labels_)+int(str(maxPredLabel))+1)      	
      list_sp_pred_true_text_ind_prevPred=np.column_stack((newOffsetLabels, trues, texts, inds, prevPreds)).tolist()
      self.updateModelParametersList(list_sp_pred_true_text_ind_prevPred, batchDocs)
	  
      pred_true_text_ind_prevPreds_to_not_cluster_spec=pred_true_text_ind_prevPreds_to_not_cluster
      #pred_true_text_ind_prevPreds_to_not_cluster_spec=change_pred_label(pred_true_text_ind_prevPreds_to_not_cluster, pred_clusters+1) no need
	  
      outlier_pred_true_text_ind_prevPreds, non_outlier_pred_true_text_ind_prevPreds, avgItemsInCluster, maxPredLabel, clusters=self.removeOutlierConnectedComponentLexical(list_sp_pred_true_text_ind_prevPred+pred_true_text_ind_prevPreds_to_not_cluster_spec, batchDocs, maxPredLabel)
      Evaluate(non_outlier_pred_true_text_ind_prevPreds)
     
	  
      dic_itemGroups=groupItemsBySingleKeyIndex(non_outlier_pred_true_text_ind_prevPreds,0)
	  
      modified_all_outliers, maxPredLabel=self.assignOutlierToClusterSimilarityEmbedding(outlier_pred_true_text_ind_prevPreds+outliersPrevStage, dic_itemGroups, batchDocs,maxPredLabel)
      
      final_list=modified_all_outliers+non_outlier_pred_true_text_ind_prevPreds	 
      #final_list=non_outlier_pred_true_text_ind_prevPreds 	  
 
    def profileClusters(self, documentSet,wordVectorsDic):
      batchDocs, pred_true_text_inds, maxPredLabel=self.populateBatchDocs(documentSet)
	  
      outlier_pred_true_text_ind_prevPreds, non_outlier_pred_true_text_ind_prevPreds, avgItemsInCluster, maxPredLabel, all_pred_clusters=self.removeOutlierConnectedComponentLexicalIndex(pred_true_text_inds, batchDocs, maxPredLabel)
      print("outlier="+str(len(outlier_pred_true_text_ind_prevPreds))+", non-outlier="+str(len(non_outlier_pred_true_text_ind_prevPreds))+",=maxPredLabel="+str(maxPredLabel))
	  
      cleaned_non_outlier_pred_true_txt_ind_prevPreds=RemoveHighClusterEntropyWordsIndex(non_outlier_pred_true_text_ind_prevPreds)

      dic_itemGroups=groupItemsBySingleKeyIndex(non_outlier_pred_true_text_ind_prevPreds,0)

      for label, cluster_pred_true_txt_inds in dic_itemGroups.items():
        	  
        nparr=np.array(cluster_pred_true_txt_inds) 
        preds=list(nparr[:,0])
        trues=list(nparr[:,1])
        texts=list(nparr[:,2])
        inds=list(nparr[:,3])
        X=generate_sent_vecs_toktextdata(texts, wordVectorsDic, 300)

        centVec=np.sum(np.array(X), axis=0)
        intlabel= int(str(label))
        if intlabel in self.centerVecs:
          centVec=centVec+np.array(self.centerVecs[intlabel])
        centVec=list(centVec) 		  
        self.centerVecs[intlabel]=centVec
        clusterSizes=list(map(len, self.docsPerCluster.values()))		
        self.avgClusterSize=statistics.mean(clusterSizes) 
        self.stdClusterSize=statistics.stdev(clusterSizes)
        #print("self.avgClusterSize", self.avgClusterSize, self.stdClusterSize)		
        		

        '''closeVec,farVec= computeClose_Far_vec(centVec, X)
        self.closetVecs[intlabel]=closeVec
        self.farthestVecs[intlabel]=farVec
        if len(centVec)!=300 or len(closeVec)!=300 or len(farVec)!=300: 		
          print("centVec", len(centVec))
          print("closeVec", len(closeVec))
          print("farVec", len(farVec))'''
       	  
        		

        		
        		
    def cluster_MStream(self, documentSet, wordVectorsDic, skStopWords, wordList, outputPath):
      t11=datetime.now()			
      self.intialize(documentSet, wordVectorsDic)
      self.gibbsSampling(documentSet, wordVectorsDic)
      t12=datetime.now()	  
      t_diff = t12-t11
      print("batch", self.batchNum,"time diff secs-kdd-gibbsSampling=",t_diff.seconds)	   			
            			
      self.detectOutlierAndEnhanceByEmbeddingSimProduct_Employee(documentSet, skStopWords, wordVectorsDic) 			
      #self.detectOutlierAndEnhanceByEmbeddingSimProduct_ACL(documentSet, skStopWords, wordVectorsDic)	
      #self.detectOutlierForgetEnhanceByEmbeddingSimProduct_ACL(documentSet, skStopWords, wordVectorsDic)

            	
      print("\tGibbs sampling successful! Start to saving results.")
      self.output(documentSet, outputPath, wordList, self.batchNum - 1)
      print("\tSaving successful!")

    def cluster_EmployeeRecruitment(self, documentSet, wordVectorsDic, skStopWords, wordList, outputPath):
      t11=datetime.now()			
      self.intialize(documentSet, wordVectorsDic)
      t12=datetime.now()	  
      t_diff = t12-t11
      print("batch", self.batchNum,"time diff subtruct=",t_diff.seconds)		
      	
	  
        		
      	  
    def run_MStream(self, documentSet, outputPath, wordList, AllBatchNum, wordVectorsDic):
        self.D_All = documentSet.D  # The whole number of documents
        self.z = {}  # Cluster assignments of each document                 (documentID -> clusterID)
        self.m_z = {}  # The number of documents in cluster z               (clusterID -> number of documents)
        self.n_z = {}  # The number of words in cluster z                   (clusterID -> number of words)
        self.n_zv = {}  # The number of occurrences of word v in cluster z  (n_zv[clusterID][wordID] = number)
        self.currentDoc = 0  # Store start point of next batch
        self.startDoc = 0  # Store start point of this batch
        self.D = 0  # The number of documents currently
        self.K_current = copy.deepcopy(self.K) # the number of cluster containing documents currently
        # self.BatchSet = {} # No need to store information of each batch
        self.word_current = {} # Store word-IDs' list of each batch
        #rakib
        self.docsPerCluster={}		
        self.centerVecs={}
        self.centerFarthestDist={}
        self.centerclosestDist={}		
        self.avgClusterSize=0
        self.stdClusterSize=0
        self.outsNumberPerBatch={}
        self.docsPerBatch={}
        self.flat_non_outs=[]
        self.flat_outs=[]
        self.docsPerCluster_outlier_afterAssign={}		
        self.docsPerCluster_nonoutlier_afterAssign={}		
        #self.dic_ClusterGroups={} #dicWords and totalWCount for lexical similarity		
        #self.batchPerDocProbList={} #calculate the probability at runtime, not store the probs		
        #self.closetVecs={}
        #self.farthestVecs={}
        		

        # Get batchNum2tweetID by AllBatchNum
        self.getAveBatch(documentSet, AllBatchNum)
        print("batchNum2tweetID is ", self.batchNum2tweetID)

        if os.path.exists("result/NewsPredTueTextMStream_WordArr.txt"):
          os.remove("result/NewsPredTueTextMStream_WordArr.txt")
        if os.path.exists('result/mstr-enh'):
          os.remove('result/mstr-enh')
        if os.path.exists('result/mstr-enh-sd'):
          os.remove('result/mstr-enh-sd')		  

        skStopWords=getScikitLearn_StopWords()

        t1=datetime.now()	

        while self.currentDoc < self.D_All:
            print("Batch", self.batchNum)
            if self.batchNum not in self.batchNum2tweetID:
                break
				
            self.cluster_MStream(documentSet, wordVectorsDic, skStopWords, wordList, outputPath)
            			
            print(len(documentSet.documents))
            
        t2=datetime.now()
        t_diff = t2-t1
        print("time diff secs=",t_diff.seconds)
        listtuple_pred_true_text=ReadPredTrueText('result/mstr-enh')
        Evaluate(listtuple_pred_true_text)
        #print("listtuple_pred_true_text-final-enhance")		
        #print_by_group(listtuple_pred_true_text)		
        #listtuple_pred_true_text=ReadPredTrueText('result/mstr-enh-sd')
        #Evaluate(listtuple_pred_true_text)
        listtuple_pred_true_text=ReadPredTrueText('result/NewsPredTueTextMStream_WordArr.txt')
        Evaluate(listtuple_pred_true_text)
        #print("listtuple_pred_true_text-final-kdd")		
        #print_by_group(listtuple_pred_true_text)		
        		

    def run_MStreamF(self, documentSet, outputPath, wordList, AllBatchNum, wordVectorsDic):
        self.D_All = documentSet.D  # The whole number of documents
        self.z = {}  # Cluster assignments of each document                 (documentID -> clusterID)
        self.m_z = {}  # The number of documents in cluster z               (clusterID -> number of documents)
        self.n_z = {}  # The number of words in cluster z                   (clusterID -> number of words)
        self.n_zv = {}  # The number of occurrences of word v in cluster z  (n_zv[clusterID][wordID] = number)
        self.currentDoc = 0  # Store start point of next batch
        self.startDoc = 0  # Store start point of this batch
        self.D = 0  # The number of documents currently
        self.K_current = copy.deepcopy(self.K) # the number of cluster containing documents currently
        self.BatchSet = {} # Store information of each batch
        self.word_current = {} # Store word-IDs' list of each batch
        #rakib
        self.docsPerCluster={}	
        self.centerVecs={}
        self.centerFarthestDist={}
        self.centerclosestDist={}		
        self.avgClusterSize=0
        self.stdClusterSize=0
         		
        		

        # Get batchNum2tweetID by AllBatchNum
        self.getAveBatch(documentSet, AllBatchNum)
        print("batchNum2tweetID is ", self.batchNum2tweetID)
		
        if os.path.exists("result/NewsPredTueTextMStream_WordArr.txt"):
          os.remove("result/NewsPredTueTextMStream_WordArr.txt")
        if os.path.exists('result/mstr-enh'):
          os.remove('result/mstr-enh')
        if os.path.exists('result/mstr-enh-sd'):
          os.remove('result/mstr-enh-sd')		  

        t1=datetime.now()		

        while self.currentDoc < self.D_All:
            print("Batch", self.batchNum)
            if self.batchNum not in self.batchNum2tweetID:
                break
            if self.batchNum <= self.Max_Batch:
                self.BatchSet[self.batchNum] = {}
                self.BatchSet[self.batchNum]['D'] = copy.deepcopy(self.D)
                self.BatchSet[self.batchNum]['z'] = copy.deepcopy(self.z)
                self.BatchSet[self.batchNum]['m_z'] = copy.deepcopy(self.m_z)
                self.BatchSet[self.batchNum]['n_z'] = copy.deepcopy(self.n_z)
                self.BatchSet[self.batchNum]['n_zv'] = copy.deepcopy(self.n_zv)
                self.intialize(documentSet, wordVectorsDic)
                self.gibbsSampling(documentSet, wordVectorsDic)
            else:
                # remove influence of batch earlier than Max_Batch
                self.D -= self.BatchSet[self.batchNum - self.Max_Batch]['D']
                for cluster in self.m_z:
                    if cluster in self.BatchSet[self.batchNum - self.Max_Batch]['m_z']:
                        self.m_z[cluster] -= self.BatchSet[self.batchNum - self.Max_Batch]['m_z'][cluster]
                        self.n_z[cluster] -= self.BatchSet[self.batchNum - self.Max_Batch]['n_z'][cluster]
                        for word in self.n_zv[cluster]:
                            if word in self.BatchSet[self.batchNum - self.Max_Batch]['n_zv'][cluster]:
                                self.n_zv[cluster][word] -= \
                                    self.BatchSet[self.batchNum - self.Max_Batch]['n_zv'][cluster][word]
                for cluster in range(self.K):
                    self.checkEmpty(cluster)
                self.BatchSet.pop(self.batchNum - self.Max_Batch)
                self.BatchSet[self.batchNum] = {}
                self.BatchSet[self.batchNum]['D'] = copy.deepcopy(self.D)
                self.BatchSet[self.batchNum]['z'] = copy.deepcopy(self.z)
                self.BatchSet[self.batchNum]['m_z'] = copy.deepcopy(self.m_z)
                self.BatchSet[self.batchNum]['n_z'] = copy.deepcopy(self.n_z)
                self.BatchSet[self.batchNum]['n_zv'] = copy.deepcopy(self.n_zv)
                self.intialize(documentSet, wordVectorsDic)
                self.gibbsSampling(documentSet, wordVectorsDic)
            # get influence of only the current batch (remove other influence)
            self.BatchSet[self.batchNum-1]['D'] = self.D - self.BatchSet[self.batchNum-1]['D']
            for cluster in self.m_z:
                if cluster not in self.BatchSet[self.batchNum - 1]['m_z']:
                    self.BatchSet[self.batchNum - 1]['m_z'][cluster] = 0
                if cluster not in self.BatchSet[self.batchNum - 1]['n_z']:
                    self.BatchSet[self.batchNum - 1]['n_z'][cluster] = 0
                self.BatchSet[self.batchNum - 1]['m_z'][cluster] = self.m_z[cluster] - self.BatchSet[self.batchNum - 1]['m_z'][cluster]
                self.BatchSet[self.batchNum - 1]['n_z'][cluster] = self.n_z[cluster] - self.BatchSet[self.batchNum - 1]['n_z'][cluster]
                if cluster not in self.BatchSet[self.batchNum - 1]['n_zv']:
                    self.BatchSet[self.batchNum - 1]['n_zv'][cluster] = {}
                for word in self.n_zv[cluster]:
                    if word not in self.BatchSet[self.batchNum - 1]['n_zv'][cluster]:
                        self.BatchSet[self.batchNum - 1]['n_zv'][cluster][word] = 0
                    self.BatchSet[self.batchNum - 1]['n_zv'][cluster][word] = self.n_zv[cluster][word] - self.BatchSet[self.batchNum - 1]['n_zv'][cluster][word]
            print("\tGibbs sampling successful! Start to saving results.")

            self.output(documentSet, outputPath, wordList, self.batchNum - 1)
            print("\tSaving successful!")
			
        t2=datetime.now()
        t_diff = t2-t1
        print("time diff secs=",t_diff.seconds) 
        #listtuple_pred_true_text=ReadPredTrueText('result/mstr-enh')
        #Evaluate(listtuple_pred_true_text)
        #print("listtuple_pred_true_text-final-enhance")		
        #print_by_group(listtuple_pred_true_text)		
		
        listtuple_pred_true_text=ReadPredTrueText('result/NewsPredTueTextMStream_WordArr.txt')
        Evaluate(listtuple_pred_true_text)		
    # Compute beta0 for every batch
    def getBeta0(self):
        Words = []
        if self.batchNum < 5:
            for i in range(1, self.batchNum + 1):
                Words = list(set(Words + self.word_current[i]))
        if self.batchNum >= 5:
            for i in range(self.batchNum - 4, self.batchNum + 1):
                Words = list(set(Words + self.word_current[i]))
        return (float(len(list(set(Words)))) * float(self.beta))

    def intialize(self, documentSet, wordVectorsDic):
        self.word_current[self.batchNum] = []
        for d in range(self.currentDoc, self.D_All):
            document = documentSet.documents[d]
            documentID = document.documentID
            # This method is getting beta0 at the beginning of initialization considering the whole words in current batch
            # for w in range(document.wordNum):
            #     wordNo = document.wordIdArray[w]
            #     if wordNo not in self.word_current[self.batchNum]:
            #         self.word_current[self.batchNum].append(wordNo)
            if documentID != self.batchNum2tweetID[self.batchNum]:
                self.D += 1
            else:
                break
        self.beta0 = self.getBeta0()
        self.alpha = self.alpha0 * self.D
        print("\t" + str(self.D) + " documents will be analyze. alpha is" + " %.2f." % self.alpha + "\n\tInitialization.", end='\n')
        #print("rakib docs="+str(len(documentSet.documents)))
        for d in range(self.currentDoc, self.D_All):
            document = documentSet.documents[d]
            documentID = document.documentID

            # This method is getting beta0 before each document is initialized
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                if wordNo not in self.word_current[self.batchNum]:
                    self.word_current[self.batchNum].append(wordNo)
            self.beta0 = self.getBeta0()
            if self.beta0 <= 0:
                print("Wrong V!")
                exit(-1)

            if documentID != self.batchNum2tweetID[self.batchNum]:
                cluster = self.sampleCluster(d, document, documentID, 0, wordVectorsDic)
                self.z[documentID] = cluster
                if cluster not in self.m_z:
                    self.m_z[cluster] = 0
                self.m_z[cluster] += 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    if cluster not in self.n_zv:
                        self.n_zv[cluster] = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    self.n_zv[cluster][wordNo] += wordFre
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    self.n_z[cluster] += wordFre
                if d == self.D_All - 1:
                    self.startDoc = self.currentDoc
                    self.currentDoc = self.D_All
                    self.batchNum += 1
            else:
                self.startDoc = self.currentDoc
                self.currentDoc = d
                self.batchNum += 1
                break

    def customGibbsSampling(self, pred_true_text_ind_prevPreds, documentSet, wordVectorsDic):
      #for i in range(self.iterNum):
      for i in range(10):	  
        print("\titer is ", i+1, end='\n')
        j=-1		
        for pred_true_text_ind_prevPred in pred_true_text_ind_prevPreds:
          j=j+1	
          pred=int(str(pred_true_text_ind_prevPred[0]))
          true=int(str(pred_true_text_ind_prevPred[1]))	
          word_arr=pred_true_text_ind_prevPred[2]
          ind=int(str(pred_true_text_ind_prevPred[3]))
          prevPred=int(str(pred_true_text_ind_prevPred[4]))
          document = documentSet.documents[ind]
          documentID = document.documentID
          cluster = self.z[documentID]
          if cluster in self.m_z and cluster in self.n_zv and cluster in self.n_z:		  
            self.m_z[cluster] -= 1
            for w in range(document.wordNum):
              wordNo = document.wordIdArray[w]
              wordFre = document.wordFreArray[w]
              if wordNo in self.n_zv[cluster]: 			  
                self.n_zv[cluster][wordNo] -= wordFre
              self.n_z[cluster] -= wordFre
			  
          self.checkEmpty(cluster)
          if i != self.iterNum - 1:  # if not last iteration
            cluster = self.customSampleCluster(document, documentID, 0, wordVectorsDic)
          elif i == self.iterNum - 1:  # if last iteration
            cluster = self.customSampleCluster(document, documentID, 1, wordVectorsDic)
	
          pred_true_text_ind_prevPreds[j][0]=str(cluster)	
          self.z[documentID] = cluster
          if cluster not in self.m_z:
            self.m_z[cluster] = 0
          self.m_z[cluster] += 1
          for w in range(document.wordNum):
            wordNo = document.wordIdArray[w]
            wordFre = document.wordFreArray[w]
            if cluster not in self.n_zv:
              self.n_zv[cluster] = {}
            if wordNo not in self.n_zv[cluster]:
              self.n_zv[cluster][wordNo] = 0
            if cluster not in self.n_z:
              self.n_z[cluster] = 0
            self.n_zv[cluster][wordNo] += wordFre
            self.n_z[cluster] += wordFre
			
      return pred_true_text_ind_prevPreds		  
        		
    def gibbsSampling(self, documentSet, wordVectorsDic):
        
        for i in range(self.iterNum):
            print("\titer is ", i+1, end='\n')
            for d in range(self.startDoc, self.currentDoc):
                document = documentSet.documents[d]
                documentID = document.documentID		
                # rakib if i==0:
                # print(str(documentID)+","+str(document.text))
                cluster = self.z[documentID]
                self.m_z[cluster] -= 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    self.n_zv[cluster][wordNo] -= wordFre
                    self.n_z[cluster] -= wordFre
                self.checkEmpty(cluster)
                if i != self.iterNum - 1:  # if not last iteration
                    cluster = self.sampleCluster(d, document, documentID, 0, wordVectorsDic)
                elif i == self.iterNum - 1:  # if last iteration
                    cluster = self.sampleCluster(d, document, documentID, 1, wordVectorsDic)
                self.z[documentID] = cluster
                if cluster not in self.m_z:
                    self.m_z[cluster] = 0
                self.m_z[cluster] += 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    if cluster not in self.n_zv:
                        self.n_zv[cluster] = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    self.n_zv[cluster][wordNo] += wordFre
                    self.n_z[cluster] += wordFre
        return

    #may need to multiply common word count with similarity, then normalize, and take the maximum
    def customSampleCluster(self, document, documentID, isLast, wordVectorsDic):
        prob = [float(0.0)] * (self.K + 1)
        for cluster in range(self.K):
            if cluster not in self.m_z or self.m_z[cluster] == 0:
                prob[cluster] = 0
                continue
            prob[cluster] = self.m_z[cluster] #/ (self.D - 1 + self.alpha)
            valueOfRule2 = 1.0
            i = 0
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                for j in range(wordFre):
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    valueOfRule2 *= (self.n_zv[cluster][wordNo] + self.beta + j) / (self.n_z[cluster] + self.beta0 + i)
                    i += 1
            prob[cluster] *= valueOfRule2
        prob[self.K] = self.alpha #/ (self.D - 1 + self.alpha)
        valueOfRule2 = 1.0
        i = 0
        for w in range(document.wordNum):
            wordFre = document.wordFreArray[w]
            for j in range(wordFre):
                valueOfRule2 *= (self.beta + j) / (self.beta0 + i)
                i += 1
        prob[self.K] *= valueOfRule2
        #rakib		
        '''if self.batchNum>1:
          for i in range(len(prob)):
            if i in self.centerVecs.keys() and i in self.docsPerCluster.keys(): # and self.m_z[i]>=self.avgClusterSize :# +self.stdClusterSize:
              #print("find center key=", i)			
              X=generate_sent_vecs_toktextdata([document.text], wordVectorsDic, 300)      
              text_Vec= X[0]	  
              simVal= 1-cosine(text_Vec, list(np.array(self.centerVecs[i])/len(self.docsPerCluster[i])))
              #print("find center key=", i, "sim=", simVal)			  
              #prob[i]= simVal 
              			  
              prob[i]= prob[i]*simVal #/len(self.docsPerCluster[i]) #/sum(map(len, self.docsPerCluster.values()))
            #else:
            #  print("missing cluster center="+str(i))	'''		

        
        allProb = 0 # record the amount of all probabilities
        prob_normalized = [] # record normalized probabilities
        for k in range(self.K+1):
            allProb += prob[k]
        for k in range(self.K + 1):
            prob_normalized.append(prob[k]/allProb)
        #rakib	
        kChoosed=prob_normalized.index(max(prob_normalized))
        if kChoosed == self.K:
          self.K += 1
          self.K_current += 1

        '''kChoosed = 0
        if isLast == 0:
            for k in range(1, self.K + 1):
                prob[k] += prob[k - 1]
            thred = random.random() * prob[self.K]
            while kChoosed < self.K + 1:
                if thred < prob[kChoosed]:
                    break
                kChoosed += 1
            
        else:
            bigPro = prob[0]
            for k in range(1, self.K + 1):
                if prob[k] > bigPro:
                    bigPro = prob[k]
                    kChoosed = k'''
           

        '''kChoosed = 0
        if isLast == 0:
            for k in range(1, self.K + 1):
                prob_normalized[k] += prob_normalized[k - 1]
            thred = random.random() * prob_normalized[self.K]
            while kChoosed < self.K + 1:
                if thred < prob_normalized[kChoosed]:
                    break
                kChoosed += 1
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1
        else:
            bigPro = prob_normalized[0]
            for k in range(1, self.K + 1):
                if prob_normalized[k] > bigPro:
                    bigPro = prob_normalized[k]
                    kChoosed = k
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1'''

        return kChoosed	
		
    def sampleCluster(self, d, document, documentID, isLast, wordVectorsDic):
        prob = [float(0.0)] * (self.K + 1)
        for cluster in range(self.K):
            if cluster not in self.m_z or self.m_z[cluster] == 0:
                prob[cluster] = 0
                continue
            prob[cluster] = self.m_z[cluster] #/ (self.D - 1 + self.alpha)
            valueOfRule2 = 1.0
            i = 0
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                for j in range(wordFre):
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    valueOfRule2 *= (self.n_zv[cluster][wordNo] + self.beta + j) / (self.n_z[cluster] + self.beta0 + i)
                    i += 1
            prob[cluster] *= valueOfRule2
        prob[self.K] = self.alpha #/ (self.D - 1 + self.alpha)
        valueOfRule2 = 1.0
        i = 0
        for w in range(document.wordNum):
            wordFre = document.wordFreArray[w]
            for j in range(wordFre):
                valueOfRule2 *= (self.beta + j) / (self.beta0 + i)
                i += 1
        prob[self.K] *= valueOfRule2
        #rakib		
        '''if self.batchNum>1 and (self.batchNum%4==0)==0:
          for i in range(len(prob)):
            if i in self.centerVecs.keys() and i in self.docsPerCluster.keys() and self.m_z[i]>=self.avgClusterSize :# +self.stdClusterSize:
              #print("find center key=", i)			
              X=generate_sent_vecs_toktextdata([document.text], wordVectorsDic, 300)      
              text_Vec= X[0]	  
              simVal= 1-cosine(text_Vec, list(np.array(self.centerVecs[i])/len(self.docsPerCluster[i])))
              #print("find center key=", i, "sim=", simVal)			  
              prob[i]= prob[i]*simVal/len(self.docsPerCluster[i]) #/sum(map(len, self.docsPerCluster.values()))
            #else:
            #  print("missing cluster center="+str(i))'''			

        
        allProb = 0 # record the amount of all probabilities
        prob_normalized = [] # record normalized probabilities
        for k in range(self.K+1):
            allProb += prob[k]
        for k in range(self.K + 1):
            prob_normalized.append(prob[k]/allProb)
        #rakib	
        kChoosed=prob_normalized.index(max(prob_normalized))
        if kChoosed == self.K:
          self.K += 1
          self.K_current += 1

        '''kChoosed = 0
        if isLast == 0:
            for k in range(1, self.K + 1):
                prob[k] += prob[k - 1]
            thred = random.random() * prob[self.K]
            while kChoosed < self.K + 1:
                if thred < prob[kChoosed]:
                    break
                kChoosed += 1
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1
        else:
            bigPro = prob[0]
            for k in range(1, self.K + 1):
                if prob[k] > bigPro:
                    bigPro = prob[k]
                    kChoosed = k
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1'''

        '''kChoosed = 0
        if isLast == 0:
            for k in range(1, self.K + 1):
                prob_normalized[k] += prob_normalized[k - 1]
            thred = random.random() * prob_normalized[self.K]
            while kChoosed < self.K + 1:
                if thred < prob_normalized[kChoosed]:
                    break
                kChoosed += 1
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1
        else:
            bigPro = prob_normalized[0]
            for k in range(1, self.K + 1):
                if prob_normalized[k] > bigPro:
                    bigPro = prob_normalized[k]
                    kChoosed = k
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1'''

        return kChoosed		

    # Clear the useless cluster
    def checkEmpty(self, cluster):
        if cluster in self.n_z and self.m_z[cluster] == 0:
            self.K_current -= 1
            self.m_z.pop(cluster)
            if cluster in self.n_z:
                self.n_z.pop(cluster)
                self.n_zv.pop(cluster)

    def output(self, documentSet, outputPath, wordList, batchNum):
        outputDir = outputPath + self.dataset + self.ParametersStr + "Batch" + str(batchNum) + "/"
        try:
            isExists = os.path.exists(outputDir)
            if not isExists:
                os.mkdir(outputDir)
                print("\tCreate directory:", outputDir)
        except:
            print("ERROR: Failed to create directory:", outputDir)
        self.outputClusteringResult(outputDir, documentSet, batchNum)
        self.estimatePosterior()
        try:
            self.outputPhiWordsInTopics(outputDir, wordList, self.wordsInTopicNum)
        except:
            print("\tOutput Phi Words Wrong!")
        self.outputSizeOfEachCluster(outputDir, documentSet)

    def estimatePosterior(self):
        self.phi_zv = {}
        for cluster in self.n_zv:
            n_z_sum = 0
            if self.m_z[cluster] != 0:
                if cluster not in self.phi_zv:
                    self.phi_zv[cluster] = {}
                for v in self.n_zv[cluster]:
                    if self.n_zv[cluster][v] != 0:
                        n_z_sum += self.n_zv[cluster][v]
                for v in self.n_zv[cluster]:
                    if self.n_zv[cluster][v] != 0:
                        self.phi_zv[cluster][v] = float(self.n_zv[cluster][v] + self.beta) / float(n_z_sum + self.beta0)

    def getTop(self, array, rankList, Cnt):
        index = 0
        m = 0
        while m < Cnt and m < len(array):
            max = 0
            for no in array:
                if (array[no] > max and no not in rankList):
                    index = no
                    max = array[no]
            rankList.append(index)
            m += 1

    def outputPhiWordsInTopics(self, outputDir, wordList, Cnt):
        outputfiledir = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "PhiWordsInTopics.txt"
        writer = open(outputfiledir, 'w')
        for k in range(self.K):
            rankList = []
            if k not in self.phi_zv:
                continue
            topicline = "Topic " + str(k) + ":\n"
            writer.write(topicline)
            self.getTop(self.phi_zv[k], rankList, Cnt)
            for i in range(rankList.__len__()):
                tmp = "\t" + wordList[rankList[i]] + "\t" + str(self.phi_zv[k][rankList[i]])
                writer.write(tmp + "\n")
        writer.close()

    def outputSizeOfEachCluster(self, outputDir, documentSet):
        outputfile = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "SizeOfEachCluster.txt"
        writer = open(outputfile, 'w')
        topicCountIntList = []
        for cluster in range(self.K):
            if cluster in self.m_z and self.m_z[cluster] != 0:
                topicCountIntList.append([cluster, self.m_z[cluster]])
        line = ""
        for i in range(topicCountIntList.__len__()):
            line += str(topicCountIntList[i][0]) + ":" + str(topicCountIntList[i][1]) + ",\t"
        writer.write(line + "\n\n")
        line = ""
        topicCountIntList.sort(key = lambda tc: tc[1], reverse = True)
        for i in range(topicCountIntList.__len__()):
            line += str(topicCountIntList[i][0]) + ":" + str(topicCountIntList[i][1]) + ",\t"
        writer.write(line + "\n")
        writer.close()

    def docWordsExistInTrainingClusters(self, doc, allTrainingWords):
        wordarr = doc.split()
        for word in wordarr:
           if word in allTrainingWords:
               return True
        
        return False 
    
    def getSuitableClusterIndexSimilarity(self, dic_itemGroups, text, pred, maxPredLabel):
      clustInd=-100
      maxSim=0	  
      for cleanedPredLabel, cleaned_pred_true_txt_inds in dic_itemGroups.items():
        if cleanedPredLabel==pred:
          continue
        listStrs=extractBySingleIndex(cleaned_pred_true_txt_inds, 2)		  
        comText=combineDocsToSingle(listStrs)
        txtSim, commonCount=computeTextSimCommonWord(text, comText)
        if txtSim>0 and txtSim> maxSim:
          maxSim=txtSim
          clustInd=int(cleanedPredLabel)		  
          #print("text="+text+",cleanedPredLabel="+str(cleanedPredLabel)+",comText="+comText) 		  
          		  
        		
      #now we need to check, whether text should be inside the closet group or should be outside (clustInd=maxPredLabel+1)
      if clustInd==-100:
        clustInd=int(str(maxPredLabel))+1	  
        #clustInd=int(str(pred)) 		
      #print("new clustInd="+str(clustInd)+", old pred="+str(clustIndpred)+", text="+text)
      return 	  
	
    #we need to pass pred_label wise center vectror to avoid repeated computation
    def getSuitableClusterIndexSimilarityEmbedding(self, dic_itemGroups, text, pred, maxPredLabel, wordVectorsDic):
      clustInd=-100
      maxSim=0
      text=text.strip()	  
      if len(text)==0:
        return int(str(pred))	  
      X = generate_sent_vecs_toktextdata([text], wordVectorsDic, 300)
      text_Vec=X[0]
      vec_sum=sum(text_Vec)
      if vec_sum==0.0:
        return int(str(pred))
      	  
      for pred_label, pred_true_text_ind_prevPreds in dic_itemGroups.items():
        nparr=np.array(pred_true_text_ind_prevPreds) 
        preds=list(nparr[:,0])
        trues=list(nparr[:,1])
        texts=list(nparr[:,2])
        inds=list(nparr[:,3])
        prevInds=list(nparr[:,4])
        textVecs= generate_sent_vecs_toktextdata(texts, wordVectorsDic, 300)
        npTextVecs=np.array(textVecs)
        sumVec=np.sum(npTextVecs, axis=0)
        clusterSim=compute_sim_value(text_Vec, sumVec)
        if clusterSim> maxSim:
          maxSim=clusterSim
          clustInd=int(str(pred_label))		  
                
      		
	  
      if clustInd==-100:
        clustInd=int(str(maxPredLabel))+1
        print("clustInd (-100)="+str(clustInd))		
 		
      print("new clustInd="+str(clustInd)+", old pred="+str(pred)+", text="+text+", maxSim="+str(maxSim))
      return clustInd	  
	
    def updateModelParametersBySimilarity(self, outlier_pred_true_text_inds, cleaned_non_outlier_pred_true_txt_inds, batchDocs, maxPredLabel):

      dic_itemGroups=groupItemsBySingleKeyIndex(cleaned_non_outlier_pred_true_txt_inds, 0)	

      for pred_true_text_ind in outlier_pred_true_text_inds:
        pred=pred_true_text_ind[0]
        true=pred_true_text_ind[1]
        text=pred_true_text_ind[2]
        ind=pred_true_text_ind[3]
        document=batchDocs[ind]
        documentID = document.documentID
        cluster=self.z[documentID] #cluster=old cluster
        self.m_z[cluster] -= 1
        for w in range(document.wordNum):
          wordNo = document.wordIdArray[w]
          wordFre = document.wordFreArray[w]	  
          self.n_zv[cluster][wordNo] -= wordFre
          self.n_z[cluster] -= wordFre
        
        self.checkEmpty(cluster)
        
        #cluster=int(new_p_t_txt[0]) #assign new cluster		
        #call function to get new cluster index except pred
        cluster=self.getSuitableClusterIndexSimilarity(dic_itemGroups, text, pred, maxPredLabel)
    		
        if int(str(maxPredLabel)) < int(str(cluster)):
          maxPredLabel=cluster 		
		
        self.z[documentID] = cluster
        if cluster not in self.m_z:
          self.m_z[cluster] = 0
        self.m_z[cluster] += 1
        for w in range(document.wordNum):
          wordNo = document.wordIdArray[w]
          wordFre = document.wordFreArray[w]
          if cluster not in self.n_zv:
            self.n_zv[cluster] = {}
          if wordNo not in self.n_zv[cluster]:
            self.n_zv[cluster][wordNo] = 0
          if cluster not in self.n_z:
            self.n_z[cluster] = 0
          self.n_zv[cluster][wordNo] += wordFre
          self.n_z[cluster] += wordFre		  	
	
    def updateModelParameters(self, outlier_pred_true_texts, newOutlier_pred_true_txts, outlier_indecies, batchDocs):
      #print(self)
      for i in range(len(outlier_pred_true_texts)):
        old_p_t_txt=outlier_pred_true_texts[i]
        new_p_t_txt=newOutlier_pred_true_txts[i]
        ind_in_mainList=int(outlier_indecies[i])
        document=batchDocs[ind_in_mainList]
        documentID = document.documentID
        #print("self.z[documentID], old_p_t_txt[0] should be same")
        #print(str(self.z[documentID])==old_p_t_txt[0])
        
        cluster=self.z[documentID] #cluster=old cluster
        self.m_z[cluster] -= 1
        for w in range(document.wordNum):
          wordNo = document.wordIdArray[w]
          wordFre = document.wordFreArray[w]
          #print("wordNo, wordFre")		  
          #print(wordNo, wordFre)		  
          #print(cluster in self.n_zv, wordNo in self.n_zv[cluster])	  
          self.n_zv[cluster][wordNo] -= wordFre
          self.n_z[cluster] -= wordFre
        
        self.checkEmpty(cluster)
		
        cluster=int(new_p_t_txt[0]) #assign new cluster		
        self.z[documentID] = cluster
        if cluster not in self.m_z:
          self.m_z[cluster] = 0
        self.m_z[cluster] += 1
        for w in range(document.wordNum):
          wordNo = document.wordIdArray[w]
          wordFre = document.wordFreArray[w]
          if cluster not in self.n_zv:
            self.n_zv[cluster] = {}
          if wordNo not in self.n_zv[cluster]:
            self.n_zv[cluster][wordNo] = 0
          if cluster not in self.n_z:
            self.n_z[cluster] = 0
          self.n_zv[cluster][wordNo] += wordFre
          self.n_z[cluster] += wordFre

    def updateModelParametersForSingleDelete(self, oldPredLabel, document):
      oldPredLabel=int(str(oldPredLabel))
      documentID = document.documentID
      cluster=self.z[documentID] #cluster=old cluster
      if cluster not in self.m_z:
        return	  
      self.m_z[cluster] -= 1
      if str(oldPredLabel)!=str(cluster):
        print("#updateModelParametersForSingleDelete:oldPredLabel not cluster=old"+str(oldPredLabel)+", cluster="+str(cluster))
      for w in range(document.wordNum):
        wordNo = document.wordIdArray[w]
        wordFre = document.wordFreArray[w]
        if cluster not in self.n_zv or wordNo not in self.n_zv[cluster] or cluster not in self.n_z:
            continue		  
        self.n_zv[cluster][wordNo] -= wordFre
        self.n_z[cluster] -= wordFre
		
      self.checkEmpty(cluster)	

    def updateModelParametersList_DeleteDocsUpdateCenter(self, targetbatchNo, documents, wordVectorsDic):
      print("targetbatchNo", targetbatchNo)
      print("len(docsPerBatch)", len(documents))

      for document in documents:
        documentID = document.documentID
        cluster=self.z[documentID]
        self.updateModelParametersForSingleDelete(cluster, document)

        text=document.text
        X = generate_sent_vecs_toktextdata([text], wordVectorsDic, 300)
        text_Vec=X[0]
        if cluster in self.centerVecs:
          self.centerVecs[cluster]=np.array(self.centerVecs[cluster])-np.array(text_Vec)
        elif str(cluster) in self.centerVecs:
          self.centerVecs[str(cluster)]=np.array(self.centerVecs[str(cluster)])-np.array(text_Vec)		
       
    def updateModelParametersForSingle(self, oldPredLabel, chngPredLabel, document):
      oldPredLabel=int(str(oldPredLabel))
      chngPredLabel=int(str(chngPredLabel))
      documentID = document.documentID
      cluster=self.z[documentID] #cluster=old cluster

      if str(oldPredLabel)!=str(cluster):
        print("oldPredLabel not cluster=old"+str(oldPredLabel)+", cluster="+str(cluster))

      if cluster in self.m_z:
        self.m_z[cluster] -= 1
      
      if cluster in self.n_zv and cluster in self.n_z:
        for w in range(document.wordNum):
          wordNo = document.wordIdArray[w]
          wordFre = document.wordFreArray[w]
          if wordNo in self.n_zv[cluster]:		  
            self.n_zv[cluster][wordNo] -= wordFre
          self.n_z[cluster] -= wordFre

      self.checkEmpty(cluster)
      cluster=chngPredLabel #assign new cluster		
      self.z[documentID] = cluster
      if cluster not in self.m_z:
        self.m_z[cluster] = 0
      self.m_z[cluster] += 1
      for w in range(document.wordNum):
        wordNo = document.wordIdArray[w]
        wordFre = document.wordFreArray[w]
        if cluster not in self.n_zv:
          self.n_zv[cluster] = {}
        if wordNo not in self.n_zv[cluster]:
          self.n_zv[cluster][wordNo] = 0
        if cluster not in self.n_z:
          self.n_z[cluster] = 0
        self.n_zv[cluster][wordNo] += wordFre
        self.n_z[cluster] += wordFre	
	  
      '''if cluster >= self.K:
        self.K = cluster
        self.K_current = cluster'''	  

    def updateModelParametersList(self,list_sp_pred_true_text_ind_prevPred, batchDocs):
       for pred_true_text_ind_prevPred in list_sp_pred_true_text_ind_prevPred:
         newPred=str(pred_true_text_ind_prevPred[0])
         oldPred=str(pred_true_text_ind_prevPred[4])	 
         ind=int(str(pred_true_text_ind_prevPred[3]))		 
         document=batchDocs[ind]         
         self.updateModelParametersForSingle(oldPred, newPred, document)		 
  
    def outputClusteringResult(self, outputDir, documentSet, batchNum):
        outputPath = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "ClusteringResult" + ".txt"
        writer = open(outputPath, 'w')
        f = open("result/NewsPredTueTextMStream_WordArr.txt", 'a')
		
        #docTexts = set()
        batchDocs = []	##contains only the docs in a single batch	
        print("start doc="+str(self.startDoc)+", end doc="+str(self.currentDoc))			
        for d in range(self.startDoc, self.currentDoc):           	
            documentID = documentSet.documents[d].documentID			
            cluster = self.z[documentID]
            #rakib			
            #docTexts.add(documentSet.documents[d].text)
            documentSet.documents[d].predLabel=cluster             
            batchDocs.append(documentSet.documents[d])			
            writer.write(str(documentID) + " " + str(cluster) + "\n")
            f.write(str(cluster)+"	"+str(documentSet.documents[d].clusterNo)+"	"+documentSet.documents[d].text+"\n")			
        writer.close()
        f.close()
        
        #rakib: writing each batch docs
        fileBatchWrire = open("result/batchId_PredTrueText"+str(batchNum), 'w')
		
        for batchDoc in batchDocs:
            documentID = batchDoc.documentID			
            cluster = self.z[documentID]           			
            fileBatchWrire.write(str(cluster)+"\t"+str(batchDoc.clusterNo)+"\t"+batchDoc.text+"\n")		
        fileBatchWrire.close() 	
        #rakib: writing each batch docs. end
        #print("Evaluate-kdd", self.batchNum)	  
        #listtuple_pred_true_text=ReadPredTrueText("result/batchId_PredTrueText"+str(batchNum))
        #Evaluate(listtuple_pred_true_text)	
         
        ctr = sum(map(len, self.docsPerCluster.values()))
        print("total texts="+str(ctr)+", total clusters="+str(len(self.docsPerCluster)))

        '''outlier_pred_true_texts, non_outlier_pred_true_txts, avgItemsInCluster=removeOutlierConnectedComponentLexical(pred_true_texts)	
		
        outlier_indecies=findIndexByItems(outlier_pred_true_texts, pred_true_texts)
        print("len(outlier_indecies), len(set(outlier_indecies))")
        print(len(outlier_indecies), len(set(outlier_indecies)))
      
        #assign new labels to the outliers    		
        newOutlier_pred_true_txts=change_pred_label(outlier_pred_true_texts, maxPredLabel)

        #change the model.py variables
        self.updateModelParameters(outlier_pred_true_texts, newOutlier_pred_true_txts, outlier_indecies, batchDocs)  		
		
        ##############rakib semantic classification##################
		#if self.docsPerCluster.setdefault has something
		#if nothing is initialized, then assign it to according to FStream
		##############end######################### 
		
        #add the docs to dic for GSDPMM, rakib 
        docsPerClusterToBeUsedForTraining = {} #not using
        wordsPerClusterToBeUsedForTraining = {}
        singleClusterTexts = set() ##contains all the texts that are in their own clusters		
        allTrainingWords= set()
	
        for clusterid, docs in self.docsPerCluster.items():
            if len(docs)==1:
                singleClusterTexts.add(docs[0].text)            
            elif len(docs)>1:
                mergeDic = Counter()			
                for document in docs:
                    docTxtArr = document.text.split()
                    #allTrainingWords.update(docTxtArr) 
                    mergeDic= mergeDic + Counter(docTxtArr)
                    #document.predLabel=clusterid                                       
                    #if document.text in docTexts:					
                    #    otherDocsToKeepAsItIs.append(document)                    				   
				   
                mergeDic=Counter(el for el in mergeDic.elements() if mergeDic[el] >= 1)				                   
                #if len(mergeDic)==0:
                #    for doc in docs:
                #        docsPerClusterToBeReclassified.setdefault(clusterid,[]).append(doc)                        				     
                #    print("0 mergeDic")	
                #else:
                allTrainingWords.update(list(mergeDic.elements()))
                docsPerClusterToBeUsedForTraining[clusterid]=docs
                wordsPerClusterToBeUsedForTraining[clusterid]=mergeDic					     
        
        docsToBeReclassified = []
        otherDocsToKeepAsItIs = []

        ############main part to generate test dataset############
        for batchDoc in batchDocs:
            #if len(set(batchDoc.text.split()).intersection(allTrainingWords))==0:
            if batchDoc.text in singleClusterTexts: # and len(set(batchDoc.text.split()).intersection(allTrainingWords))==0: # not docWordsExistInTrainingClusters(batchDoc.text, allTrainingWords):
                docsToBeReclassified.append(batchDoc)
            else:
                otherDocsToKeepAsItIs.append(batchDoc)			
 
                
				
        print("dic len="+str(len(self.docsPerCluster)))
        print("#docsToBeReclassified="+str(len(docsToBeReclassified))+", #otherDocsToKeepAsItIs="+str(len(otherDocsToKeepAsItIs)))
        print("#docsPerClusterToBeUsedForTraining="+str(len(docsPerClusterToBeUsedForTraining)))	
        print("#wordsPerClusterToBeUsedForTraining="+str(len(wordsPerClusterToBeUsedForTraining)))
        ###########rakib start readjusting#######
        #rakib popultate word cluster vectors, populate doc cluster vectror for reclassify
        wordsVectorPerClusterForTraining = {}
        docsVectorToBeReclassified = {}
        for clusterid, wordCounters in wordsPerClusterToBeUsedForTraining.items():
            combinedWordVector = [0]*300            
            onlyWords = list(wordCounters)
            #print("clusterid="+str(clusterid)+",onlyWords="+str(onlyWords))
            for word in onlyWords:
                combinedWordVector= list(map(add, combinedWordVector, getWordEmbedding(word)))
            wordsVectorPerClusterForTraining[clusterid]= list(numpy.divide(combinedWordVector, len(onlyWords)))
            #print(wordsVectorPerClusterForTraining[clusterid])
        
        globalOutPathSemantic="result/NewsPredTueTextMStreamSemantic_WordArr.txt"		
        fSemantic = open(globalOutPathSemantic, 'a')
	
        for doc in otherDocsToKeepAsItIs:
            fSemantic.write(str(doc.predLabel)+"	"+str(doc.clusterNo)+"	"+doc.text+"\n") 			
		
        for doc in docsToBeReclassified:
            wordArr = doc.text.strip().split()
            singleDocVector = [0]*300
            for word in wordArr:
                singleDocVector= list(map(add, singleDocVector, getWordEmbedding(word)))
            singleDocVector=list(numpy.divide(singleDocVector, len(wordArr)))
            predLabel=classifyBySimilaritySingle(singleDocVector, wordsVectorPerClusterForTraining)
            fSemantic.write(str(predLabel)+"	"+str(doc.clusterNo)+"	"+doc.text+"\n")			

        fSemantic.close()'''			
		
