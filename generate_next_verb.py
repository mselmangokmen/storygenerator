import torch
from dataset_loader_files.mydataset import prepareDataset  
import json 

# LSTM Model 
import numpy as np

from models.lstm_model_verb import LSTMModelVerb

datasetPrep = prepareDataset( batch_size = 1,sample_size=3,num_stories=50000)
datasetPrep.loadPrompts()
datasetPrep.createCorpus()
datasetPrep.createDatasetLoader()
dataloader =datasetPrep.getDatasetLoader()['train'] 
word_to_id_verb  ,  id_to_word_verb    = datasetPrep.getVerbsID()  
# Hiperparametreler 
vocab_size_verb  = len(word_to_id_verb)  
print('vocab_size_verb  : ' + str(vocab_size_verb))
embedding_dim = 512# 256: 6.62  hidden_dim = 128  num_layers = 5
hidden_dim = 512 # 256: 6.27  num_layers = 5  embedding_dim = 128 # 512: 5.55 num_layers = 5  embedding_dim = 128
num_layers = 5# 10: 6.55  hidden_dim = 256 embedding_dim = 128  # 10: 6.13  hidden_dim = 512 embedding_dim = 128 
# num_layers = 5 hidden_dim = 512 embedding_dim = 128  ==>  5.55
# num_layers = 5 hidden_dim = 512 embedding_dim = 128  ==>  ? 
 

device = torch.device("mps") 
# Model, Optimizer ve Loss Fonksiyonu 

modelVerb = LSTMModelVerb(vocab_size_verbs=vocab_size_verb, 
                   embedding_dim= embedding_dim, hidden_dim= hidden_dim, num_layers= num_layers).to(device=device) 
 
sample_size=3
modelVerb.load_state_dict(torch.load("pretrained_models/modelVerb.pt"))
i=0
prompt_dict_list =list()
for batch in dataloader:
        if i<3000:
            verbs, nouns,prompt, idx , target_story= batch  
            #print(verbs.item())  

            #print('size of logits_verb: '+ str(logits_verb.shape))
            input_tensor_verbs = verbs[:, :-1].to(device = device)
            target_tensor_verbs = verbs[:, 1:].to(device = device) 
            logits_verb    = modelVerb(input_tensor_verbs)
 
            next_word_verb_id = torch.argmax(logits_verb[:, -1, :])  
 
            next_word_verb = id_to_word_verb[next_word_verb_id.item()]  

            input_verbs= [id_to_word_verb[id.item()] for id in input_tensor_verbs[0]]
            print('Input Verbs: ',str(input_verbs), 'next verb: '+ next_word_verb )  
            if len(input_verbs) == sample_size -1:
                prompt_dict= dict() 
                prompt_dict['input_verbs']= input_verbs
                prompt_dict['prompt']= prompt 
                prompt_dict['next_verb']= next_word_verb
                prompt_dict['index']= idx.item()
                #story_dict['target_story']= target_story 
                prompt_dict_list.append(prompt_dict)
        else:
              break
        
        i=i+1
json_dict= dict()
json_dict['prompts'] = prompt_dict_list
with open("datasets/next_verb_with_prompts.json", "w+") as outfile:
    json.dump(json_dict, outfile)
 
