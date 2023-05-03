import torch
from dataset_loader_files.mydataset import prepareDataset 
from models.lstm_model_noun import LSTMModelNoun 
import json 

# LSTM Model 
import numpy as np
datasetPrep = prepareDataset( batch_size = 1,sample_size=3,num_stories=100000)
datasetPrep.loadPrompts()
datasetPrep.createCorpus()
datasetPrep.createDatasetLoader()
dataloader =datasetPrep.getDatasetLoader()['train'] 
word_to_id_nouns ,  id_to_word_nouns  = datasetPrep.getNounsID() 
# Hiperparametreler 
vocab_size_nouns = len(word_to_id_nouns)
print('vocab_size_nouns : ' + str(vocab_size_nouns))  
embedding_dim = 512# 256: 6.62  hidden_dim = 128  num_layers = 5
hidden_dim = 512  
num_layers = 5  
device = torch.device("mps") 
sample_size=3
# Model, Optimizer ve Loss Fonksiyonu 

modelNoun = LSTMModelNoun(vocab_size_nouns=vocab_size_nouns, 
                   embedding_dim= embedding_dim, hidden_dim= hidden_dim, num_layers= num_layers).to(device=device)
 
 
modelNoun.load_state_dict(torch.load("pretrained_models/modelNoun.pt"))
i=0
story_dict_list =list()
for batch in dataloader:
        if i<3000:
            verbs, nouns,prompt, idx , target_story= batch  
            print(prompt)  

            #print('size of logits_verb: '+ str(logits_verb.shape))
            input_tensor_nouns = nouns[:, :-1].to(device = device)
            target_tensor_nouns = nouns[:, 1:].to(device = device) 
            logits_noun    = modelNoun(input_tensor_nouns)
 
            next_word_noun_id = torch.argmax(logits_noun[:, -1, :])  
 
            next_word_noun = id_to_word_nouns[next_word_noun_id.item()]  

            input_nouns= [id_to_word_nouns[id.item()] for id in input_tensor_nouns[0]]
            print('Input Nouns: ',str(input_nouns), 'next noun: '+ next_word_noun )  
            if len(input_nouns) == sample_size -1:
                story_dict= dict() 
                story_dict['input_nouns']= input_nouns
                story_dict['prompt']= prompt 
                story_dict['next_noun']= next_word_noun
                story_dict['index']= idx.item()
                #story_dict['target_story']= target_story 
                story_dict_list.append(story_dict)
        else:
              break
        
        i=i+1
json_dict= dict()
json_dict['prompts'] = story_dict_list
with open("datasets/next_noun_with_prompts.json", "w+") as outfile:
    json.dump(json_dict, outfile)
 
