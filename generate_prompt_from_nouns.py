import json
import torch  
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from dataset_loader_files.prompt_generator_data_loader import PreparePromptsForGeneration 
 
 
def generate_prompt(prompt,k=0,p=0.9,output_length=50,temperature=1,num_return_sequences=1,repetition_penalty=1.0):
    #print("====prompt====\n")
    #print(prompt+"\n")
    #print('====target story is as below===\n')
    #print(target+"\n")
    
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    model.to('cpu')
    model.eval()
    output_sequences = model.generate(
        input_ids=encoded_prompt.to('cpu') ,
        max_length=output_length,
        temperature=temperature,
        top_k=k,
        top_p=p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        num_return_sequences=num_return_sequences
    )
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        #print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()
        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        # Remove all text after eos token 
        #text = text[: text.find(tokenizer.eos_token)]
        text= text.split('<sep>') 

        if len(text)>1:
            nouns = text[0].strip()
            predicted_prompt = text[1].strip()
            predicted_prompt = predicted_prompt.split('.')[0] + '.'
            nouns = nouns.strip().split(' ')
            return nouns, predicted_prompt 
        else:
            return None,None

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.load_state_dict(torch.load("pretrained_models/gpt2_prompt_generator_nouns.pt"))

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token=tokenizer.eos_token
dataset_prep= PreparePromptsForGeneration( type='noun')

trains = dataset_prep.getDataset()

#for i in len(trains):

prompt_dict_list= []
json_dict= dict()
#json_dict['prompts'] = prompt_dict_list
#for i in range(len(trains)):
for i in range(100):
    #random_sample = randrange(len(trains)) 
    sample = trains[i]
    #target_story = sample['target_story'] 
    prompt = sample['words']
    target_prompt=sample['prompt']  
    index=sample['index']  
    
    nouns, predicted_prompt = generate_prompt(prompt)
    print('generated')
    if nouns is not None and predicted_prompt is not None:
        predicted_prompt= dataset_prep.cleanpunctuation(predicted_prompt)
        print('index : '+ str(i))
        print('nouns : '+ str(nouns))
        print('predicted_prompt : '+ str(predicted_prompt))
        prompt_dict= {'input_nouns': nouns  , 'predicted_prompt': predicted_prompt, 'target_prompt': target_prompt,   'index': index}
        prompt_dict_list.append(prompt_dict)

json_dict['prompts'] = prompt_dict_list
with open("datasets/predicted_prompts_with_nouns.json", "w") as outfile:
    json.dump(json_dict, outfile)