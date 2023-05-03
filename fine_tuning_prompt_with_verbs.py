import torch
import torch.nn as nn 
from transformers import   GPT2LMHeadModel
import json 
import numpy as np
from tqdm import tqdm
import math

import gc
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from dataset_loader_files.prompt_train_data_loader import PreparePromptsForGPT2
 
 


def train_model(model, optimizer, scheduler,train_dataloader, valid_dataloader,train_batch_size,valid_batch_size, device): 
    print("***** Running training *****")
    print("  Total_num_training_step = {}".format(total_num_training_steps))
    print("  Num Epochs = {}".format(num_train_epochs))
    print(f"  Train_batch_size per device = {train_batch_size}")
    print(f"  Valid_batch_size per device = {valid_batch_size}")
    model.to(device= device)
    for epoch in range(num_train_epochs):
        print(f"Start epoch{epoch+1} of {num_train_epochs}")
        train_loss=0
        epoch_iterator = tqdm(train_dataloader,desc='Iteration')
        model.train()
        model.zero_grad()    
        for _, inputs in enumerate(epoch_iterator):        
            d1,d2,d3=inputs
            d1=d1.to(device= device)
            d2=d2.to(device= device)
            d3=d3.to(device= device)
            output = model(input_ids=d1, attention_mask=d2,labels=d3)
            batch_loss=output[0]
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            train_loss+=batch_loss.item()
            epoch_iterator.set_description('(batch loss=%g)' % batch_loss.item())
            del batch_loss
        print(f'Average train loss per example={train_loss/training_steps_per_epoch} in epoch{epoch+1}')    
        print(f'Starting evaluate after epoch {epoch+1}')
        eval_loss=[]    
        model.eval()    
        for inputs in tqdm(valid_dataloader, desc="eval"):
            d1,d2,d3=inputs
            d1=d1.to(device= device)       
            d2=d2.to(device= device)
            d3=d3.to(device= device)
            with torch.no_grad():
                output = model(input_ids=d1, attention_mask=d2,labels=d3)
                batch_loss=output[0]
            eval_loss+=[batch_loss.cpu().item()]
            del batch_loss
        eval_loss=np.mean(eval_loss)
        perplexity=math.exp(eval_loss)
        print(f'Average valid loss per example={eval_loss} in epoch{epoch+1}')    
        print(f'Perplextiy for valid dataset in epoch{epoch+1} is {perplexity}')

    torch.save(model.state_dict(), "pretrained_models/gpt2_prompt_generator_verbs.pt")
 
gc.collect()
torch.cuda.empty_cache() 
device = torch.device("mps") 
num_train_epochs = 20
train_batch_size= 30
valid_batch_size= 30
model = GPT2LMHeadModel.from_pretrained('gpt2')
dataset_prep= PreparePromptsForGPT2(type='verb', batch_size=train_batch_size)

dataloader =dataset_prep.getDataSetLoaders()

training_steps_per_epoch=len(dataloader['train'])
total_num_training_steps = int(training_steps_per_epoch*num_train_epochs)
#weight_decay=0
#learning_rate=0.001
#adam_epsilon=0.01
#warmup=0.1
weight_decay=0.1
learning_rate=0.01
adam_epsilon=0.01
warmup=0.1
warmup_steps=int(total_num_training_steps*warmup)
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_num_training_steps
)
train_model(model=model,optimizer=optimizer,scheduler=scheduler,train_batch_size=train_batch_size,valid_batch_size=valid_batch_size
            ,train_dataloader=dataloader['train'], valid_dataloader=dataloader['val'],device=device)