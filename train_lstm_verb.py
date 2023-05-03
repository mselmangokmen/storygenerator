import torch
import torch.nn as nn
import torch.optim as optim

from dataset_loader_files.mydataset import prepareDataset 
from models.lstm_model_verb import LSTMModelVerb 
# LSTM Model
 


datasetPrep = prepareDataset( batch_size = 500,sample_size=3,num_stories=50000)
datasetPrep.loadPrompts()
datasetPrep.createCorpus()
datasetPrep.createDatasetLoader()
dataloader =datasetPrep.getDatasetLoader() 
word_to_id_verb  ,  id_to_word_verb    = datasetPrep.getVerbsID()  
# Hiperparametreler 
vocab_size_verb  = len(word_to_id_verb)  
print('vocab_size_verb  : ' + str(vocab_size_verb))
embedding_dim = 512# 256: 6.62  hidden_dim = 128  num_layers = 5
hidden_dim = 512 # 256: 6.27  num_layers = 5  embedding_dim = 128 # 512: 5.55 num_layers = 5  embedding_dim = 128
num_layers = 5# 10: 6.55  hidden_dim = 256 embedding_dim = 128  # 10: 6.13  hidden_dim = 512 embedding_dim = 128 
# num_layers = 5 hidden_dim = 512 embedding_dim = 128  ==>  5.55
# num_layers = 5 hidden_dim = 512 embedding_dim = 128  ==>  ? 
learning_rate = 0.0001
num_epochs = 1000

device = torch.device("mps") 
# Model, Optimizer ve Loss Fonksiyonu 

modelVerb = LSTMModelVerb(vocab_size_verbs=vocab_size_verb, 
                   embedding_dim= embedding_dim, hidden_dim= hidden_dim, num_layers= num_layers).to(device=device) 
optimizerVerb = optim.Adam(modelVerb.parameters(), lr=learning_rate,weight_decay=1e-5  )

loss_fn_verb = nn.CrossEntropyLoss()  
# Eğitim döngüsü 
modelVerb.train() 
best_loss_verbs =999999
#eps= 0.01  
for epoch in range(num_epochs):         

    for phase in ['train', 'val']: 
        if phase == 'train':   
            modelVerb.train() 
        else: 
            modelVerb.eval()  
 
        loss_list_verb  =[]  
        for batch in dataloader[phase]:
            verbs, _,_ ,_,_= batch   
            #print(verbs) 
            # Giriş ve hedef tensörlerini hazırlama 
 

            input_tensor_verbs = verbs[:, :-1].to(device = device)
            target_tensor_verbs = verbs[:, 1:].to(device = device)
 
            optimizerVerb.zero_grad()
                #print('input : ' + str(input_tensor.shape))
                #print('target : ' + str(target_tensor.shape)) 
                # Model tahminleri

            with torch.set_grad_enabled(phase == 'train'):         
                logits_verb    = modelVerb(target_tensor_verbs)
                      #print('outputs size: '+ str(outputs.size()) ) 
                loss_verbs = loss_fn_verb(logits_verb.reshape(-1, vocab_size_verb), target_tensor_verbs.reshape(-1))
                      # backward + optimize only if in training phase
                if phase == 'train': 
                    loss_verbs.backward() 
                    optimizerVerb.step()
                    
                 
 
            loss_list_verb.append(loss_verbs.item()) 
 
        avg_loss_verbs = sum(loss_list_verb) / len(loss_list_verb)
 
  
        if phase =='train':
            print(f"Epoch {epoch + 1}/{num_epochs},   Loss Train Verb: {avg_loss_verbs}")
        else:
            print(f"Epoch {epoch + 1}/{num_epochs},   Loss Val Verb: {avg_loss_verbs}")
  
 
    torch.save(modelVerb.state_dict(), 'pretrained_models/modelVerb.pt') 

