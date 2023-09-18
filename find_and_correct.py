import sys
import re
import torch
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM
import json
from fuzzywuzzy import fuzz
import numpy
import argparse

def regex_catch_predict_word(answer:str):
    '''
    extract answer words from LLMs's OUTPUT since they don't always follow the instruction 
    '''
    if answer.find('\"') > -1:
        predict_word = re.findall(r"\"(.+?)\"",answer)
    elif answer.find('\'') > -1:
        predict_word = re.findall(r"\'(.+?)\'",answer)
    else:
        return []
    result = []
    for item in predict_word:
        if item.find(','):
            result = result + item.strip().split(',')
        else:
            result.append(item.strip().strip('.'))
    return [i for i in result if i != '' and i != ',']

def correct_the_mistake(target_words:list,original_answer:str,model,tokenizer):
    '''
    use the target words to let the LLM(model) fix the bug
    '''
    prompt = "Proofreader can rewrite a paragraph and correct its grammar errors. Once the paragraph and the errors are given.\
        the Proofreader can correct all the listed error phrases in the paragraph into the right form and write out the correct sentence\
               For example, 'Paragraph: This are the expire sand wiches, I don't want them.\n Errors: 'This','expire','sand wiches'\
                  \nProofreader: These are the expired sandwiches, I don't want them.'\
                    Proofreader accuratedly rewrite the following paragraph. Paragraph:\
                       :{}\nErrors:{}\nProofreader:"
    input_raw = prompt.format(original_answer,','.join(["\'"+i+"\'" for i in target_words]))
    input_ids = tokenizer(input_raw, return_tensors="pt").input_ids
    input_ids = input_ids.cuda()
    outputs = model.generate(input_ids, max_new_tokens=200, do_sample = True, top_k = 50, top_p = 0.85, temperature = 0.8, repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id, pad_token_id=tokenizer.pad_token_id)
    rets = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    rets = rets[0].strip().replace(input_raw, "").strip('.').strip().lower()
    return rets


def find_the_mistake(model_input:str,target_file:str):
    '''
    extract answer words from LLMs's OUTPUT since they don't always follow the instruction 
    '''
    model_input = '/online1/ycsc_chenkh/chenkh_a100/xumufan/LLMs/' + model_input
    target_file = './datasets/' + target_file
    if model_input.find('Alpaca') > -1 or model_input.find('LLaMA') >-1 or model_input.find('Bloom') > -1:
        model =  AutoModelForCausalLM.from_pretrained(model_input, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_input)
    elif model_input.find('ChatGLM') > -1:
        model =  AutoModel.from_pretrained(model_input, device_map='auto', trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_input, trust_remote_code=True)
    else:
        model =  AutoModelForSeq2SeqLM.from_pretrained(model_input,torch_dtype="auto",trust_remote_code=True, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_input,trust_remote_code=True)
    Prompt = "Proofreader can check the grammar of a given paragraph and point out all the errors. Proofreader points out the errors by saying 'The error words are (list the error words here) ....' For example, 'Paragraph: This are the expire sand wiches, I don't want them. Proofreader: The error words are 'This,expire,sand,wiches.' Proofreader accuratedly find all the errors in the following paragraph. Paragraph: '{}' \nProofreader:"
    model.eval()
    data_file = open(target_file,'r',encoding='utf-8')
    dataset = data_file.readlines()

    for line in tqdm(dataset):
        dict_content = json.loads(line.strip())
        inputs = Prompt.format(dict_content.get('text',''))
        input_ids = tokenizer(inputs, return_tensors="pt").input_ids
        input_ids = input_ids.cuda()
        outputs = model.generate(input_ids, max_new_tokens=200, do_sample = True, top_k = 50, top_p = 0.85, temperature = 0.8, repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id, pad_token_id=tokenizer.pad_token_id)

        rets = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        rets = rets[0].strip().replace(inputs, "").strip('.').strip().lower()
        predict_word = regex_catch_predict_word(rets)
        if predict_word == []:
            continue
        print('original: ',dict_content.get('text',''))
        correction = correct_the_mistake(predict_word,dict_content.get('text',''),model,tokenizer)
        print('\ncorrection: ',correction)

if __name__ == '__main__':
    find_the_mistake(model_input='Alpaca_13B',target_file='all.json')