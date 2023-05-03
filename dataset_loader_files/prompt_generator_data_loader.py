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
import re
import string
 
 


class PreparePromptsForGeneration:   
    def __init__(self , type)-> None:
        self.type= type
        data =None 
        if type=='noun':
            f = open('datasets/next_noun_with_prompts.json')
            data = json.load(f)
        elif type =='verb':
            f = open('datasets/next_verb_with_prompts.json')
            data = json.load(f)

        self.story_list=  data['prompts'] 
        self.x_train = self.create_train_set(self.story_list) 

    def getDataset(self):
        return self.x_train

    def create_train_set(self,stories):
        x_train= [] 
        input_words = ''
        next_word = ''
        if self.type == 'verb':
            input_words= 'input_verbs'
            next_word= 'next_verb'
        elif self.type == 'noun':
            input_words= 'input_nouns'
            next_word= 'next_noun'
        for d in stories: 
                train_text = ''  
                for i in range(len(d[input_words])):
                    train_text += self.cleanpunctuation(d[input_words][i])+' '
                t = dict()
                train_text += self.cleanpunctuation(d[next_word])
                t['words'] = train_text
                t['prompt']=  self.cleanpunctuation(d['prompt'][0])
                t['index']=  d['index']
                #t['target_story']=  self.cleanpunctuation(d['target_story'][0]) 
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