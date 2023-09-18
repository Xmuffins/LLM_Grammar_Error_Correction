import sys
import re
import os
import torch
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM
import json
import numpy
import argparse
import time
import json
import torch.nn.functional as F

# os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

parser = argparse.ArgumentParser()
parser.add_argument('--model_input',type=str,default="llama2-13b")
parser.add_argument('--data_input',type=str,default='all.json')
args = parser.parse_args()
model_path = '/share/home/hubaotian/hbt_user02/data/pretrained-models/'+args.model_input+ '/'

model =  AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True,device_map='auto').eval()
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)

#bidirection_language_model = 
#bidirection_language_model
prompt = "Please replace all the '<blank>' marks in the following paragraph with proper phrases to make the whole paragraph fluent:\"{}\". Below is the paragraph after all the replacements:"

def outlier_detection(input_content=None,attention_mask=None):
    inputs = input_content
    input_ids = tokenizer(inputs, return_tensors="pt").input_ids
    input_ids_new = torch.transpose(input_ids,0,1).cuda()
    if attention_mask:
        mask = torch.tensor([attention_mask]).cuda()
    else:
        mask = tokenizer(inputs, return_tensors="pt").attention_mask
    input_ids = input_ids.cuda()
    outputs = model.forward(input_ids,attention_mask=mask).logits[0,:-1]
    result = outputs.detach().cpu().tolist()
    for index in range(len(result)):
        index_list = [j for j in range(len(result[0]))]
        result[index] = sorted(index_list,key = lambda x:result[index][x],reverse=True)
        result[index] = sorted(index_list,key = lambda x:result[index][x])
    result_indices = torch.tensor(result).cuda()
    target_indice = torch.gather(result_indices,dim=1,index = input_ids_new[1:,:])
    target_indice_new = torch.transpose(target_indice,0,1)[0,:].detach().cpu().tolist()
    return target_indice_new,mask[0].cpu().tolist()

def calculate_precision_recall(ranking_threshold,content,record):
    indice_new,last_mask = outlier_detection(input_content=content['text'])
    remain_list = [num for num in indice_new if num >= ranking_threshold]
    while len(remain_list) > 0:
        new_mask = [int(indice_new[i] < ranking_threshold) for i in range(len(indice_new))]
        new_mask.append(1)
        mask = []
        for i,j in zip(last_mask,new_mask):
            mask.append(i*j)
        if last_mask==mask:
            break
        indice_new,last_mask = outlier_detection(input_content=content['text'],attention_mask=mask)
        remain_list = [num for num in indice_new if num >= ranking_threshold]
    tokenize_input = tokenizer.tokenize(content['text'])

    index = 0
    index_list = []
    token_index_list = []
    for token,masks in zip(tokenize_input,last_mask[:-1]):
        token_index_list.append([index,index+len(token)-1])
        if masks == 0:
            if len(index_list) > 0 and index_list[-1][1] == index:
                index_list[-1][1] += len(token)
            else:
                index_list.append([index,index+len(token)])
        index += len(token)
    #print(index_list)
    true_positive = record[0]
    false_positive = record[1]
    true_negative = record[2]
    false_negative = record[3]
    error_token_list = [1 for _ in range(len(tokenize_input))]
    ## 得到token对应在string中的位置区间列表
    for index,edit in enumerate(content['edits'][0][1]):
        for token_index in token_index_list:
            if edit[1] >= token_index[0] and edit[0] <= token_index[1]:
                error_token_list[index] = 0
                break
    
    # for index,item in enumerate(content['edits'][0][1]):
    #     for index_span in index_list:
    #         if item[1] >= index_span[0] and item[0] <= index_span[1]:
    #             true_positive += 1
    #             break

    for index in range(len(last_mask)-1):
        if last_mask[index+1] == 0:
            if error_token_list[index] == 0:
                true_positive += 1
            else:
                false_positive += 1
        else:
            if error_token_list[index] == 0:
                false_negative += 1
            else:
                true_negative += 1
    
    return [true_positive,false_positive,true_negative,false_negative]

def outlier_detection_new(input_content=None,threshold=10):
    inputs = input_content
    input_ids = tokenizer(inputs, return_tensors="pt").input_ids
    input_ids = input_ids.cuda()
    mask = [1]
    extra_window_mask=[1]
    rank_list = []
    with torch.no_grad():
        for cur_generation_index in range(input_ids.size()[1]-1):
            attention_mask_new = [i*j for i,j in zip(mask,extra_window_mask)]
            if cur_generation_index == 0:
                logits = model(input_ids[:,cur_generation_index].view(1, -1),attention_mask=torch.tensor([attention_mask_new]).cuda(),use_cache=True)
                past_key_values = logits.past_key_values
            else:
                logits = model(input_ids[:,cur_generation_index].view(1, -1),attention_mask=torch.tensor([attention_mask_new]).cuda(),past_key_values=past_key_values, use_cache=True)
                past_key_values = logits.past_key_values
            generation_next_token_logits = logits.logits[0,-1].detach().cpu().tolist()
            next_token_id = input_ids[0,cur_generation_index+1].item()
            
            ###排序两次方便取出对应token id的logit大小排序位置信息###
            index_list = [j for j in range(len(generation_next_token_logits))]
            result = sorted(index_list,key = lambda x:generation_next_token_logits[x],reverse=True)
            result = sorted(index_list,key = lambda x:result[x])
            
            rank = result[next_token_id]
            rank_list.append(rank)

            # top300_result = logits.logits[0,-1].topk(200,dim=0)[0]
            # top300_result =F.log_softmax(top300_result,dim=0)
            # plt.figure()
            # plt.plot([i for i in range(len(top300_result))],top300_result.cpu().tolist())
            # plt.title(tokenizer.decode([next_token_id])+'  rank:'+str(rank))
            # plt.savefig('./figures/'+str(len(rank_list))+'.jpg')
            # plt.close()
            ##### 此处用某种方式判断是否被mask #####
            if second_time_judgement(rank,threshold) and len(mask) > 1:
                mask.append(0)
            else:
                mask.append(1)
            if len(mask) > 100:
                extra_window_mask.insert(0,0)
            else:
                extra_window_mask.insert(0,1)
        # exit()
    torch.cuda.empty_cache()
    # print(mask)
    
    return rank_list,mask

def calculate_precision_recall_new(ranking_threshold,content,record):
    indice_new,last_mask = outlier_detection_new(input_content=content['text'],threshold=ranking_threshold)

    # exit()
    tokenize_input = tokenizer.tokenize(content['text'])

    index = 0
    index_list = []
    token_index_list = []
    for token,masks in zip(tokenize_input,last_mask[:-1]):
        token_index_list.append([index,index+len(token)-1])
        if masks == 0:
            if len(index_list) > 0 and index_list[-1][1] == index:
                index_list[-1][1] += len(token)
            else:
                index_list.append([index,index+len(token)])
        index += len(token)
    #print(index_list)
    true_positive = record[0]
    false_positive = record[1]
    true_negative = record[2]
    false_negative = record[3]
    error_token_list = [1 for _ in range(len(tokenize_input))]
    
    ## 得到token对应在string中的位置区间列表
    for index,edit in enumerate(content['edits'][0][1]):
        for new_index,token_index in enumerate(token_index_list):
            if edit[1] >= token_index[0] and edit[0] <= token_index[1]:
                error_token_list[new_index] = 0
                #break
    print(len(error_token_list))
    for index in range(len(last_mask)-1):
        if last_mask[index+1] == 0:
            if 0 in error_token_list[index:index+2]:
                true_positive += 1
            else:
                false_positive += 1
        else:
            if error_token_list[index] == 0:
                false_negative += 1
            else:
                true_negative += 1
    # if false_negative - record[3] > 20:
    #     for index,item in enumerate(content['edits'][0][1]):
    #         try:
    #             print("错误{}：".format(str(index))+content['text'][item[0]:item[1]]+' --> '+item[2]+'\n')
    #         except:
    #             print(item)
        # print_string = ''
        # for i in range(len(tokenize_input)):
        #     if error_token_list[i] == 0:
        #         print_string = print_string + "<" + tokenize_input[i].replace('_',' ') + ">"
        #     else:
        #         print_string = print_string + tokenize_input[i].replace('_',' ')
        # print(error_token_list)


    return [true_positive,false_positive,true_negative,false_negative]

def fillin_strategy(past_key_values=None,eos_token=None,Now_length=None):
    limit_length = 5

    return 0

def second_time_judgement(rank,threshold):
    #,logit,target_word,original_sentence
    if rank > threshold:
        return True
    else:
        return False
    
# def bert_refill():

if __name__ == '__main__':
    file = open('./datasets/all.json','r')
    ranking_threshold=10
    record = [0,0,0,0]
    step = 0
    file_out = open('./result.txt','w')
    for line in tqdm(file):
        # if step < 70:
        #     step += 1
        #     continue
        content = json.loads(line.strip('\n').strip())
        # for index,item in enumerate(content['edits'][0][1]):
        #     try:
        #         print("错误{}：".format(str(index))+content['text'][item[0]:item[1]]+' --> '+item[2]+'\n')
        #     except:
        #         print(item)
        record = calculate_precision_recall_new(ranking_threshold,content,record)
        
        step += 1
        if step % 100 == 0:
            true_positive = record[0]
            false_positive = record[1]
            true_negative = record[2]
            false_negative = record[3]
            Precision = true_positive/(true_positive+false_positive)
            Recall = true_positive/(true_positive+false_negative)
            print("Precision: ",Precision)
            print("Recall: ",Recall)
            print("F0.5: ",(1.25*Precision*Recall)/(0.25*Precision+Recall))
            print('step {} TP: {}, FP: {},FN: {}'.format(str(step),str(record[0]),str(record[1]),str(record[3])))
            file_out.write('step {} TP: {}, FP: {},FN: {}\n'.format(str(step),str(record[0]),str(record[1]),str(record[3])))
            file_out.write("Precision:{} Recall:{} F0.5:{} \n".format(str(Precision),str(Recall),str((1.25*Precision*Recall)/(0.25*Precision+Recall))))
    # print(str(true_positive)+'/'+str(len(content['edits'][0][1])))
    #print("错误{}：".format(str(index))+content['text'][item[0]:item[1]]+' --> '+item[2]+'\n')
    file_out.close()