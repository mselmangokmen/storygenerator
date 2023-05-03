import json
import openai
def generateStory(prompt):
    openai.api_key = "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"
    
    json_dict= dict()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a story creator. Generate a story that is at least 400 words long using the sentences I provide. The story should be adventurous and adhere to the definition of a story. Please send the story as your response."},
                {"role": "user", "content": prompt},
                #{"role": "user", "content": "The gods wish they can get past the cryogenic walls and finish their amnesia test."},
            ]
    ) 
    result = ''
    for choice in response.choices:
        result += choice.message.content

    story = result 
    json_dict['story'] = story

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a story creator. I will send you a story and you send me a list of two main goals and three subgoals of the story. Just send me one list."},
                {"role": "user", "content": story},
            ]
    )  
    result = ''
    for choice in response.choices:
        result += choice.message.content


    #print(result)
    result = "\n".join([line for line in result.split("\n") if line.strip()])

    result = result.split('\n')
    main_goals = list( )
    sub_goals = list( )
    main_goals.append(result[1][3:])
    main_goals.append(result[2][3:])

    sub_goals.append(result[-3][3:])
    sub_goals.append(result[-2][3:])
    sub_goals.append(result[-1][3:])

    json_dict['main_goals'] = main_goals
    json_dict['sub_goals'] = sub_goals 

    with open("datasets/story.json", "w+") as outfile:
        json.dump(json_dict, outfile)
generateStory('We finally get men on Mars and they discover an old Soviet flag placed down decades ago.')
