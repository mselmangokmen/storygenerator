import json
import torch  
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import re
from dataset_loader_files.story_generator_data_loader import PrepareStoriesForGeneration 
 
 
def generate_prompt(prompt,k=0,p=0.9,output_length=150,temperature=1,num_return_sequences=1,repetition_penalty=1):
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
        #print('text text : '+ str(text) )
        #print('text: ' + text)
        #print('end text' )
        #text= text.split('<sep>') 
        if len(text)>1:
            #predicted_prompt = text[0].strip()
            #predicted_prompt = re.sub(r'\s*[^\.\?!]+$', '', predicted_prompt)

            predicted_story = text.strip()
            predicted_story = re.sub(r'\s*[^\.\?!]+$', '', predicted_story)
            return   predicted_story
        else:
            return None,None

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.load_state_dict(torch.load("pretrained_models/gpt2_complete_story.pt"))

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token=tokenizer.eos_token
dataset_prep= PrepareStoriesForGeneration( )

trains = dataset_prep.getDataset()

#for i in len(trains):
 
story_dict_list= []
json_dict= dict()
#json_dict['prompts'] = prompt_dict_list
for i in range(len(trains)):
#for i in range(1000):
    #random_sample = randrange(len(trains)) 
    sample = trains[i]
    #target_story = sample['target_story']  
    input_nouns = sample['input_nouns']  
    target_prompt=sample['target_prompt']  
    predicted_prompt=sample['predicted_prompt']  
    index=sample['index']  
    predicted_story = generate_prompt(predicted_prompt)
    #print('predicted_prompt: '+ str(predicted_prompt) )
    #print('predicted_story: '+ str(predicted_story) )
    if predicted_story is not None:
        print('index : '+ str(i))
        print('predicted_prompt : '+ str(predicted_prompt))
        print('predicted_story : '+ str(predicted_story))
        story_dict= {'input_nouns': input_nouns  , 
        'predicted_prompt': predicted_prompt, 'target_prompt': target_prompt, 
          'predicted_story': dataset_prep.cleanpunctuation(predicted_story), 'index': index}
        story_dict_list.append(story_dict)

json_dict['stories'] = story_dict_list
with open("datasets/predicted_stories.json", "w") as outfile:
    json.dump(json_dict, outfile)

    ##########fine tuning yap ################