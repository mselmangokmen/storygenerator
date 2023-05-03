import re
import string
import spacy
from transformers import pipeline
import math  

import torch
from torch.utils.data import Dataset, DataLoader
#classifier = pipeline("token-classification", model = "vblagoje/bert-english-uncased-finetuned-pos")
#nlp = spacy.load("en_core_web_sm")
import random
import numpy as np
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from sklearn.model_selection import train_test_split

 
 


class PrepareStoriesForGeneration:   
    def __init__(self )-> None:
        f = open('datasets/predicted_prompts_with_nouns.json')
        data = json.load(f)
        
        self.story_list=  data['prompts'] 
        self.x_train = self.create_train_set(self.story_list) 

    def getDataset(self):
        return self.x_train

    def create_train_set(self,stories):
        x_train= [] 
        for d in stories:
            train_text = '' 
            for i in range(len(d['input_nouns'])): # input nouns yapmak icin generate_prompt calistir
                train_text += self.cleanpunctuation(d['input_nouns'][i]) 
            t = dict()
            #train_text += self.cleanpunctuation(d['next_noun'])
            t['input_nouns'] = train_text
            t['predicted_prompt']=  self.cleanpunctuation(d['predicted_prompt'])
            t['target_prompt']=  self.cleanpunctuation(d['target_prompt'])
            t['index']=  d['index']
            #t['target_story']=  self.cleanpunctuation(d['target_story']) 
            x_train.append(t)  
        return x_train 
    
    def cleanpunctuation(self,s):
        for p in '!,:;?`\'_*':
            s=s.replace(p,'')
        s=s.replace( '\n',' ')
        s=s.replace( 'n\'t',' ')
        s=s.replace(' '+'\'s','\'s')
        s=s.replace(' '+'\'re','\'re')
        s=s.replace(' '+'\'ve','\'ve')
        s=s.replace(' '+'\'ll','\'ll')
        s=s.replace(' '+'\'am','\'am')
        s=s.replace(' '+'\'m','\'m')
        s=s.replace(' '+'\' m','\'m')
        s=s.replace(' '+'\'m','\'m')
        s=s.replace(' '+'\' ve','\'ve')
        s=s.replace(' '+'\' s','\'s')
        s=s.replace('<newline>',' ')
        s=s.replace('[ WP ]','') 
        #s=' '.join(s.split())
        printable = set(string.printable)
        s=''.join(filter(lambda x: x in printable, s))
        s= re.sub(r'\s+', ' ', s)
        #s= re.sub(r'\*+', '', s)
        #s = re.sub(r'\b\w{1,2}\b', '', s)
        s = re.sub(r'\.{2,}', '.', s)
        s = re.sub(r'[^\.\s\w]|_', '', s)
        s= re.sub(r'\s+', ' ', s)
        s = re.sub(r'\s+\.', '.', s) #noktadan once bosluk
        s = re.sub(r'^\s*[\.]+', '', s) #birden fazla bosluklu nokta vs var ise

        return s   
 
'''
dataset_prep= prepareDatasetForGPT2( batch_size=1)

dataloader =dataset_prep.getDataSetLoaders()
batch = next(iter(dataloader['train']))
verb_list  = batch
print( verb_list[0].shape) '''