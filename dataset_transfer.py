import json
from random import sample
import random
import re
import argparse
from transformers import BertTokenizer, BertForMaskedLM
import torch
from tqdm import tqdm
from fuzzywuzzy import fuzz
#from english_words import get_english_words_set

IS_ERROR_CORRECTION_DATASET = True

def retrieve_correct_sentence(json_content):
    text = json_content['text']
    position_offset = 0
    #print('before: ',text)
    for _,edit_list in json_content['edits']:
        for edit_start,edit_end,original_text in edit_list:
            
            if not original_text:
                original_text = ''
            text = text[:edit_start+position_offset] + original_text + text[edit_end+position_offset:]
            position_offset += len(original_text) - (edit_end-edit_start)
    #print('after: ',text)
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='指定数据转换方法')
    parser.add_argument('--mode',type=str,default='random')
    parser.add_argument('--replace_length',type=int,default=3)
    args = parser.parse_args()

    mode = args.mode
    mode_dict = ['random','random_character','bert']

    assert mode in mode_dict
    assert args.replace_length > 0


    mode = 'random' if mode == 'random' else 'bert'   
    output_file = open("./datasets/{}_fill_{}_word.json".format(mode,str(args.replace_length)),'w',encoding='utf-8')
    if IS_ERROR_CORRECTION_DATASET:
        input_file = open('./datasets/all.json','r')
        content = []
        for line in input_file:
            content.append(json.loads(line.strip()))
        random.shuffle(content)
    else:
        input_file = open('./alpaca_data.json','r')    
        content = json.load(input_file)
        random.shuffle(content)
    dictionary = []
    with open('./vocab_common_english_word.txt','r') as f:
        for line in f:
            dictionary.append(line.strip())
    model_path = '../../LLMs/Bert_multilingual/'
    model = BertForMaskedLM.from_pretrained(model_path)
    tokenizer = tokenizer = BertTokenizer.from_pretrained(model_path)


    for data_module in tqdm(content):
        #output_file.write(json.dumps(data_module,ensure_ascii=False)+'\n')
        output_json = dict()
        if random.random()<0.9:
            if IS_ERROR_CORRECTION_DATASET:
                input_text = retrieve_correct_sentence(data_module)
            else:
                input_text = data_module['output']
                if data_module['output'].find(' ') < 0:
                    continue
            response_word_list = re.split(" ",input_text)
            if len(response_word_list) < args.replace_length:
                continue
            random_replace_length = args.replace_length
            random_replace_start_index = round(random.random()*(len(response_word_list)-random_replace_length))
            random_replace_end_index = random_replace_start_index+random_replace_length
            original_word_list = response_word_list[random_replace_start_index:random_replace_end_index]

            if mode == 'bert':
                first_half = response_word_list[:random_replace_start_index].copy()
                second_half = response_word_list[random_replace_end_index:].copy()
                mask_list = ['[MASK]']*random_replace_length
                inputs_new = tokenizer(' '.join(first_half+mask_list+second_half),max_length=512,return_tensors="pt")
                
                with torch.no_grad():
                    logits = model(**inputs_new).logits

                mask_token_index = (inputs_new.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
                #predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
                predicted_token_id = logits[0, mask_token_index].topk(5,dim=-1).indices[:,-1]
                replaced_ones = tokenizer.decode(predicted_token_id)
                if fuzz.partial_ratio(' '.join(response_word_list[random_replace_start_index:random_replace_end_index]),replaced_ones) > 80:
                    # data_module['replace_start'] = -1
                    continue
                output_json['output'] = ' '.join(first_half+[replaced_ones]+second_half)

            else:
                replacement_word_list = random.sample(dictionary,random_replace_length)            
                for i in range(random_replace_start_index,random_replace_end_index):
                    try:
                        print('{} --> {}'.format(response_word_list[i],replacement_word_list[i-random_replace_start_index]))
                        response_word_list[i] = replacement_word_list[i-random_replace_start_index]
                    except:
                        print(len(response_word_list))
                        print(i)
                        print(random_replace_start_index,random_replace_end_index)
                        exit()
                output_json['output'] = ' '.join(response_word_list)
            
            output_json['replace_start'] = random_replace_start_index
            output_json['replace_end'] = random_replace_end_index
            output_json['original_phrase'] = ' '.join(original_word_list)
        else:
            output_json['replace_start'] = -1

        new_positive_sample = output_json.copy()
        output_file.write(json.dumps(new_positive_sample,ensure_ascii=False)+'\n')
    
