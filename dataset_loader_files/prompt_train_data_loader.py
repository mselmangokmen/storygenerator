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


class PromptDataLoaderForGPT2(Dataset):
    def __init__(self, inputs):
        self.ids = inputs['input_ids']
        self.attention_mask = inputs['attention_mask']
        self.labels=inputs['labels']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):

        return [torch.tensor(self.ids[item], dtype=torch.long),
                torch.tensor(self.attention_mask[item], dtype=torch.long),
                torch.tensor(self.labels[item], dtype=torch.long)]
            
 


class PreparePromptsForGPT2:  

    def __init__(self,type, batch_size = 10, max_seq_length= 1024)-> None:
        self.data = None
        self.type = type
        if type == 'verb':
            f = open('datasets/next_verb_with_prompts.json')
            data = json.load(f)
        elif type == 'noun':
            f = open('datasets/next_noun_with_prompts.json')
            data = json.load(f)
        #data= json.load()
        self.max_seq_length= max_seq_length
        self.story_list=  data['prompts']
        self.batch_size = batch_size
        self.x_train, self.x_val = self.create_train_set(self.story_list)
        #print(self.x_train[0])  
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token=self.tokenizer.eos_token

        self.x_train_tokenized = self.tokenizer(self.x_train, padding=True,truncation=True,max_length=self.max_seq_length)
        self.x_val_tokenized=self.tokenizer(self.x_val, padding=True,truncation=True,max_length=self.max_seq_length)
        
        self.create_labels(self.x_train_tokenized)
        self.create_labels(self.x_val_tokenized)
        self.x_train_loader =  PromptDataLoaderForGPT2( self.x_train_tokenized, )
        self.x_val_loader =  PromptDataLoaderForGPT2( self.x_val_tokenized ) 
        self.dataloader = {'train': DataLoader(self.x_train_loader, batch_size=self.batch_size, shuffle=False,  ) 
                           , 'val': DataLoader(self.x_val_loader, batch_size=self.batch_size, shuffle=False,  ) }
    def getDataSetLoaders(self):
        return self.dataloader
    def getTrainTextDataset(self):
        return self.x_train
    def getValidationTextDataset(self):
        return self.x_train
    
    def getDataSet(self):
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
        
            train_text += self.cleanpunctuation(d[next_word])
            train_text += ' <sep> '+ self.cleanpunctuation(d['prompt'][0])
            x_train.append(train_text) 
        
        x_train, x_val  = train_test_split(x_train  , train_size = .75)
        return x_train, x_val 
    
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
    def create_labels(self,inputs):
        labels=[]
        for ids,attention_mask in zip(inputs['input_ids'],inputs['attention_mask']):
            label=ids.copy()
            real_len=sum(attention_mask)
            padding_len=len(attention_mask)-sum(attention_mask)
            label[:]=label[:real_len]+[-100]*padding_len
            labels.append(label)
        inputs['labels']=labels
'''
dataset_prep= prepareDatasetForGPT2( batch_size=1)

dataloader =dataset_prep.getDataSetLoaders()
batch = next(iter(dataloader['train']))
verb_list  = batch
print( verb_list[0].shape) '''