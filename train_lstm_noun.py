import torch
import torch.nn as nn
import torch.optim as optim

from dataset_loader_files.mydataset import prepareDataset
from models.lstm_model_noun import LSTMModelNoun 
# LSTM Model
 


datasetPrep = prepareDataset( batch_size = 500,sample_size=3,num_stories=100000)
datasetPrep.loadPrompts()
datasetPrep.createCorpus()
datasetPrep.createDatasetLoader()
dataloader =datasetPrep.getDatasetLoader() 
word_to_id_noun  ,  id_to_word_noun    = datasetPrep.getNounsID()  
# Hiperparametreler 
vocab_size_noun  = len(word_to_id_noun)  
print('vocab_size_noun  : ' + str(vocab_size_noun))
embedding_dim = 512# 256: 6.62  hidden_dim = 128  num_layers = 5
hidden_dim = 512 # 256: 6.27  num_layers = 5  embedding_dim = 128 # 512: 5.55 num_layers = 5  embedding_dim = 128
num_layers = 5# 10: 6.55  hidden_dim = 256 embedding_dim = 128  # 10: 6.13  hidden_dim = 512 embedding_dim = 128 
# num_layers = 5 hidden_dim = 512 embedding_dim = 128  ==>  5.55
# num_layers = 5 hidden_dim = 512 embedding_dim = 128  ==>  ? 
learning_rate = 0.0001
num_epochs = 1000

device = torch.device("mps") 
# Model, Optimizer ve Loss Fonksiyonu 

modelNoun = LSTMModelNoun(vocab_size_nouns=vocab_size_noun, 
                   embedding_dim= embedding_dim, hidden_dim= hidden_dim, num_layers= num_layers).to(device=device) 
optimizerNoun = optim.Adam(modelNoun.parameters(), lr=learning_rate,weight_decay=1e-5  )

loss_fn_noun = nn.CrossEntropyLoss()  
# Eğitim döngüsü 
modelNoun.train() 
best_loss_nouns =999999
#eps= 0.01  
for epoch in range(num_epochs):         

    for phase in ['train', 'val']: 
        if phase == 'train':   
            modelNoun.train() 
        else: 
            modelNoun.eval()  
 
        loss_list_nouns  =[]  
        for batch in dataloader[phase]:
            _, nouns,_ ,_,_= batch   
            #print(verbs) 
            # Giriş ve hedef tensörlerini hazırlama 
 

            input_tensor_nouns = nouns[:, :-1].to(device = device)
            target_tensor_nouns = nouns[:, 1:].to(device = device)
 
            optimizerNoun.zero_grad()
                #print('input : ' + str(input_tensor.shape))
                #print('target : ' + str(target_tensor.shape)) 
                # Model tahminleri

            with torch.set_grad_enabled(phase == 'train'):         
                logits_noun    = modelNoun(target_tensor_nouns)
                      #print('outputs size: '+ str(outputs.size()) ) 
                loss_nouns = loss_fn_noun(logits_noun.reshape(-1, vocab_size_noun), target_tensor_nouns.reshape(-1))
                      # backward + optimize only if in training phase
                if phase == 'train': 
                    loss_nouns.backward() 
                    optimizerNoun.step()
                    
                 
 
            loss_list_nouns.append(loss_nouns.item()) 
 
        avg_loss_nouns = sum(loss_list_nouns) / len(loss_list_nouns)
 
  
        if phase =='train':
            print(f"Epoch {epoch + 1}/{num_epochs},   Loss Train Noun: {avg_loss_nouns}")
        else:
            print(f"Epoch {epoch + 1}/{num_epochs},   Loss Val Noun: {avg_loss_nouns}")
  
 
    torch.save(modelNoun.state_dict(), 'pretrained_models/modelNoun.pt') 

