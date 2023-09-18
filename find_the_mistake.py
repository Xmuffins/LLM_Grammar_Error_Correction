import sys
import re
import os
import torch
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM
import json
from fuzzywuzzy import fuzz
import numpy
import argparse

'''
    Usage: python find_the_mistake.py --model_input Alpaca_13B & 
'''
def judgement_format():
    po_rate_record = []

    print(rets+'\n')
    #if rets.find('\"') > -1 or rets.find('none') > -1 or rets.find('\'') > -1:
    right_format += 1

    if start_index == -1:
        if rets.find('none') > -1:        
            ng_correct_count += 1
        else:
            ng_wrong_count += 1
    else:
        if rets.find('none') == -1:
            true_word = ' '.join(dict_content['output'].split(' ')[start_index:dict_content['replace_end']])
            # if rets.find('\"') > -1:
            #     predict_word = re.split("\"",rets)[1].strip()
            # elif rets.find('\'')>-1:
            #     predict_word = re.split("\'",rets)[1].strip()
            # else:
            #     try:
            #         predict_word = re.split(",|is|\.",rets)[1].strip()
            #     except:
            #         po_wrong_count += 1
            #         continue
            predict_list = regex_catch_predict_word(rets)
            if predict_list != []:
                predict_word = ' '.join(predict_list)
		# predict_word = predict_list[0].strip()
		# if predict
            else:
                right_format -= 1
                po_wrong_count += 1
                #continue
            rate_now = fuzz.partial_ratio(true_word,predict_word)
            po_rate_record.append(rate_now)

            if  rate_now > 90:
                po_correct_count += 1
            else:
                po_wrong_count += 1
        else:
            po_wrong_count += 1

def get_LLM_response(input:str,model,tokenizer):
    wrong_Prompt = "Proofreader can point out the wrong phrases in a paragraph by listing them out. Proofreader points out the wrong phrases by saying 'The wrong phrases are (repeat the phrase)'. Enclose the listed words in quotation marks. Given the following paragraph: \"{}\" \nProofreader: The wrong phrases are"
    missing_Prompt = "Proofreader can find the missing words or phrases in a paragraph. Proofreader points out the missing phrases by saying 'The missing phrases are (list the phrase)'. Enclose the listed words in quotation marks. Given the following paragraph: \"{}\" \nProofreader: The wrong phrases are"
    
    inputs = wrong_Prompt.format(dict_content.get('text',''))
    input_ids = tokenizer(inputs, return_tensors="pt").input_ids
    input_ids = input_ids.cuda()
    outputs = model.generate(input_ids, max_new_tokens=100, do_sample = True, top_k = 50, top_p = 0.85, temperature = 0.8, repetition_penalty=1.2, eos_token_id=2, bos_token_id=1, pad_token_id=0)

    rets = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    rets = rets[0].strip().replace(inputs, "").strip('.').strip().lower()
    return rets

def judge_correction(predict_str:str,data:dict):
    predict_list = ' '.join(regex_catch_predict_word(predict_str))
    total_error = 0
    target_list = []
    for editor in data['edits']:
        target_content = editor[1]
        for info in target_content:
            if info[0] == info[1]:
                continue
            total_error += 1
            target_list.append(data['text'][info[0]:info[1]].lower())
    detection_count = 0
    for target in target_list:
        if fuzz.partial_ratio(predict_list,target) > 90:
            detection_count += 1
    return total_error,detection_count

def regex_catch_predict_word(answer:str):
    '''
    extract answer words from LLMs's OUTPUT since they don't always follow the instruction 
    '''
    answer = re.sub("repeat the phrase",'',answer).replace('(','').replace(')','')
    if answer.find('\"') > -1:
        predict_word = re.findall(r"\"(.+?)\"",answer)
    elif answer.find('\'') > -1:
        predict_word = re.findall(r"\'(.+?)\'",answer)
    else:
        if len(answer) > 100:
            return []
        else:
            predict_word = [answer]
    result = []
    for item in predict_word:
        if item.find(','):
            result = result + item.strip().split(',')
        else:
            result.append(item.strip().strip('.'))
    return [i for i in result if i != '' and i != ',']

parser = argparse.ArgumentParser()
parser.add_argument('--model_input',type=str,default="Alpaca_13B")
parser.add_argument('--data_input',type=str,default='all.json')
args = parser.parse_args()

DATA_INPUT = 'all' if args.data_input == 'all.json' else ''
model_path = '../../LLMs/'+ args.model_input + '/' # You can modify the path for storing the local model

print("loading model, path:", model_path)

if args.model_input.find('Alpaca') > -1 or args.model_input.find('LLaMA') > -1 or args.model_input.find('Bloom') > -1:
    model =  AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
elif args.model_input.find('ChatGLM') > -1:
    model =  AutoModel.from_pretrained(model_path, device_map='auto', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
else:
    model =  AutoModelForSeq2SeqLM.from_pretrained(model_path,torch_dtype="auto",trust_remote_code=True, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
#Prompt = "Proofreader can evaluate the fluency of a given paragraph and point out the incoherent phrases. Proofreader points out the incoherent phrases by saying 'The incoherent phrase is (repeat the phrase)'. Proofreader would say 'No incoherent phrase found' if the full paragraph is fluent. Proofreader successfully evaluate the following paragraph.  '{}' \nProofreader:"
model.eval()
data_file = open('./datasets/'+args.data_input,'r',encoding='utf-8')
dataset = data_file.readlines()

ng_wrong_count = 0
ng_correct_count = 0
po_correct_count = 0
po_wrong_count = 0
right_format = 0

correction_count = 0
total_count = 0

position_count = [[],[]]
other_count = [[],[]]

for line in tqdm(dataset[:10000]):
    dict_content = json.loads(line.strip())
    start_index = dict_content.get('replace_start','')

    # if start_index == '':
    #     continue
    
    

    inputs = Prompt.format(dict_content.get('text',''))
    input_ids = tokenizer(inputs, return_tensors="pt").input_ids
    input_ids = input_ids.cuda()
    outputs = model.generate(input_ids, max_new_tokens=100, do_sample = True, top_k = 50, top_p = 0.85, temperature = 0.8, repetition_penalty=1.2, eos_token_id=2, bos_token_id=1, pad_token_id=0)

    rets = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    rets = rets[0].strip().replace(inputs, "").strip('.').strip().lower()
    print(rets)
    id1,id2 = judge_correction(rets,dict_content)
    total_count += id1
    correction_count += id2


if DATA_INPUT == 'all':
    experiment_name = 'cambridge_2019'
else:
    experiment_name = args.data_input.rsplit('_',1)[0]

try:
    os.mkdir('./experiment_result/'+experiment_name+'/')
except:
    print('already exist')

#numpy.save('./experiment_result/'+experiment_name+'/'+experiment_name.split('_')[0]+'_'+args.model_input+'.npy',po_rate_record)
if DATA_INPUT == 'all':
    with open('./experiment_result/'+experiment_name+'/'+experiment_name+'_'+args.model_input+'.txt','w',encoding='utf-8') as f:
        f.writelines("correct_count: "+str(correction_count)+'\n')
        f.writelines("total_count: "+str(total_count)+'\n')
    print("correct_count: "+str(correction_count)+'\n')
    print("total_count: "+str(total_count)+'\n')
else:
    with open('./experiment_result/'+experiment_name+'/'+experiment_name+'_'+args.model_input+'.txt','w',encoding='utf-8') as f:
        f.writelines("ng_wrong_count: "+str(ng_wrong_count)+'\n')
        f.writelines("ng_correct_count: "+str(ng_correct_count)+'\n')
        f.writelines("po_correct_count: "+str(po_correct_count)+'\n')
        f.writelines("po_wrong_count: "+str(po_wrong_count)+'\n')
        f.writelines("right_format: "+str(right_format))

    print("ng_wrong_count: ",ng_wrong_count)
    print("ng_correct_count: ",ng_correct_count)
    print("po_correct_count: ",po_correct_count)
    print("po_wrong_count: ",po_wrong_count)
    print("right_format: ",right_format)
