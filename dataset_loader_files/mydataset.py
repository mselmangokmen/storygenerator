import spacy
from transformers import pipeline
import math   
import torch
from torch.utils.data import Dataset, DataLoader
#classifier = pipeline("token-classification", model = "vblagoje/bert-english-uncased-finetuned-pos")
#nlp = spacy.load("en_core_web_sm")
import random
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
 
class StoryDataset(Dataset):
    def __init__(self, corpus,word_to_id_verb,word_to_id_noun,  sample_size=5):
        self.corpus = corpus
        self.word_to_id_verb= word_to_id_verb 
        self.word_to_id_noun= word_to_id_noun 
        self.sample_size = sample_size

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx): 
        story = self.corpus[idx]
        verbs = story['verbs']
        nouns = story['nouns'] 
        start_index_verbs = random.randint(0, len(verbs) - self.sample_size)
        start_index_nouns = random.randint(0, len(nouns) - self.sample_size)

        # Sırasını koruyarak rastgele alt küme seçme
        selected_verbs = verbs[start_index_verbs:start_index_verbs + self.sample_size]
        selected_nouns = nouns[start_index_nouns:start_index_nouns + self.sample_size]
 
        selected_verbs = torch.tensor([self.word_to_id_verb[word] for word in selected_verbs])
        selected_nouns = torch.tensor([self.word_to_id_noun[word] for word in selected_nouns]) 
        return   selected_verbs,selected_nouns, self.corpus[idx]['prompt'], self.corpus[idx]['index'], self.corpus[idx]['target_story']
#print(len(word_to_id))
 
#doc = nlp(story_text2)
#doc.clean(texts) 


class prepareDataset(): 
    def __init__(self, batch_size = 10,sample_size=8,num_stories= 100):
        self.prompt_list = [] 
        self.story_list = [] 
        self.batch_size= batch_size
        self.sample_size= sample_size
        self.corpus=[] 
        self.corpus_val=[]  
        self.nlp = spacy.load("en_core_web_sm") 
        #self.word_to_id = dict()
        #self.id_to_word = dict()
        self.dataloader= None
        self.dataloaderForGPT2= None
        self.num_stories =num_stories
 
    def loadPrompts(self):
        self.prompt_list = self.getPromptList()[:self.num_stories]
        self.story_list = self.getStoryList()[:self.num_stories]
        #self.story_list = self.getStoryList() 
    
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

    def createCorpus(self):
        for idx, p in enumerate(self.prompt_list):
            if '[ WP ]' in p: 
                p= self.cleanpunctuation(p.strip())
                s= self.cleanpunctuation(self.story_list[idx])
                nouns,verbs = self.get_nouns_verbs(p.strip(),self.nlp)
                s_dict = dict()
                s_dict['index'] = idx
                s_dict['nouns'] = nouns
                s_dict['verbs'] = verbs 
                s_dict['prompt'] = p.strip()
                s_dict['target_story'] = s.strip()
                if len(nouns)>= self.sample_size and len(verbs)>= self.sample_size: 
                    self.corpus.append(s_dict) 
                    #print(s_dict)
        self.word_to_id_verbs, self.id_to_word_verbs  = self.create_vocabulary_verbs()
        self.word_to_id_nouns, self.id_to_word_nouns  = self.create_vocabulary_nouns()
        #self.word_to_id_nouns, self.id_to_word_nouns  = self.create_vocabulary_nouns()
        #self.word_to_id, self.id_to_word  = self.create_vocabulary_words()
        #print(len(self.word_to_id_verbs))
        print(len(self.word_to_id_verbs))
        print(len(self.word_to_id_nouns))
 
 
    def createDatasetLoader(self): 
        
        self.corpus , self.corpus_val  = train_test_split(self.corpus  , train_size = .80)
        self.dataset = StoryDataset(self.corpus, self.word_to_id_verbs, self.word_to_id_nouns, sample_size=self.sample_size)
        self.dataloader = DataLoader( self.dataset , batch_size=self.batch_size, shuffle=False)

        self.dataset_val = StoryDataset(self.corpus_val, self.word_to_id_verbs, self.word_to_id_nouns, sample_size=self.sample_size)
        self.dataloader_val = DataLoader( self.dataset_val , batch_size=self.batch_size, shuffle=False)

    def getDatasetLoader(self):
        return {'train': self.dataloader , 'val': self.dataloader_val}

    def getDatasetValLoader(self):
        return self.dataloader_val
    
    def getVerbsID(self):
        return self.word_to_id_verbs, self.id_to_word_verbs 
    
    def getNounsID(self):
        return self.word_to_id_nouns, self.id_to_word_nouns
    def getWordID(self):
        return self.word_to_id, self.id_to_word
    
    def create_vocabulary_verbs(self): 
        unique_words = set() 
        for c in self.corpus:
            
            unique_words.update(c['verbs'] ) 

        word_to_id = {word: idx for idx, word in enumerate(unique_words)}
        id_to_word = {idx: word for word, idx in word_to_id.items()}
        
        return word_to_id, id_to_word 
 
    def create_vocabulary_nouns(self): 
        unique_words = set() 
        for c in self.corpus:
            
            unique_words.update(c['nouns'] ) 

        word_to_id = {word: idx for idx, word in enumerate(unique_words)}
        id_to_word = {idx: word for word, idx in word_to_id.items()}
        
        return word_to_id, id_to_word 
    
    def create_vocabulary_words(self): 
        unique_words = set() 
        for c in self.corpus:
            
            unique_words.update(c['nouns'] ) 
            unique_words.update(c['verbs'] ) 

        word_to_id = {word: idx for idx, word in enumerate(unique_words)}
        id_to_word = {idx: word for word, idx in word_to_id.items()}
        
        return word_to_id, id_to_word 
    
    def get_subject_phrase(self,doc):
        for token in doc:
            if ("subj" in token.dep_ or  "nsubj" in token.dep_ ):
                subtree = list(token.subtree)
                start = subtree[0].i
                end = subtree[-1].i + 1
                return doc[start:end] 
            
    def get_object_phrase(self,doc):
        for token in doc:
            if ("dobj" in token.dep_):
                subtree = list(token.subtree)
                start = subtree[0].i
                end = subtree[-1].i + 1
                return doc[start:end]
            
    def getPromptList(self):
        f = open('datasets/valid.wp_source', "r")
        all_prompts = f.read()
        all_prompts = all_prompts.split('\n')

        all_prompts = [s.strip() for s in all_prompts]
        print('all_prompts len : '+ str(len(all_prompts)))
        return all_prompts

    def getStoryList(self):
        f = open('datasets/valid.wp_target', "r")
        all_stories = f.read()
        all_stories = all_stories.split('\n')

        all_stories = [s.strip() for s in all_stories]
        print('stories len : '+ str(len(all_stories)))
        return all_stories
    
    
    def get_nouns_verbs(self,story,nlp):
        #story= self.cleanpunctuation(story)
        sentence=story
        if '.' in story:  
            sentence = story.split('.')[0]
        
        sentence += '.' 
        #sentence= self.cleanpunctuation(sentence)
        verb_list = []
        noun_list = [] 
        sent_tokens =nlp(sentence) 
    
        for t in sent_tokens:
                if t.pos_ == 'NOUN' or  t.pos_ == 'PROPN':
                    if t.lemma_ not in noun_list and len(t.lemma_ )>1 :
                        noun_list.append(t.lemma_)
                elif  (t.pos_ == 'VERB' or t.tag_ =='VBN' or t.tag_ =='VBG') and len(t.lemma_ )>1:
                    if t.lemma_ not in verb_list:
                        verb_list.append(t.lemma_) 
        return noun_list,verb_list

'''
datasetPrep = prepareDataset(sample_size=3, num_stories=1500)
datasetPrep.loadStories()
datasetPrep.createCorpus()
datasetPrep.createDatasetLoader()
dataloader =datasetPrep.getDatasetLoader()['train']

batch = next(iter(dataloader))
verb_list, noun_list , story= batch
print(np.array(verb_list).shape)
print(np.array(noun_list).shape)  
print("Verbs:")
for i, verb_batch in enumerate(verb_list):
    print(f"Story {i}:")
    for verb in verb_batch:
        print(f"  Word: {verb}")
    print()

print("Nouns:")
for i, noun_batch in enumerate(noun_list):
    print(f"Story {i}:")
    for noun in noun_batch:
        print(f"  Word: {noun}")
    print() 
'''