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
import time
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model_input',type=str,default="llama2-13b")
parser.add_argument('--data_input',type=str,default='all.json')
args = parser.parse_args()
model_path = '/share/home/hubaotian/hbt_user02/data/pretrained-models/'+args.model_input+ '/'

model =  AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True,device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)

file = open('./datasets/all.json','r')
line_count = 0
while line_count > 0:
    line = file.readline()
    line_count-=1
line = file.readline().strip()
content = json.loads(line)

inputs = "After eating a hamburger yesterday, my stomach started feeling uncomfortable. Later, I went to the hospital, and the doctor prescribed medication for me. I feel much better today. I'll need to pay more attention to food hygiene in the future."
inputs = "We replaced all the '<blank>' marks in the following paragraph with proper phrases to make the whole paragraph fluent:\"{}\". Below is the paragraph after all the replacements:"

content_text = content['text']
difference = 0
for errors in content['edits'][0][1]:
    content_text = content_text[:errors[0]+difference] + '<blank>' + content_text[errors[1]+difference:]
    difference -= (errors[1]-errors[0]) - 7
inputs = inputs.format(content_text)

input_ids = tokenizer(inputs, return_tensors="pt").input_ids
output = model.generate(input_ids,max_length=400)
print(output)
exit()
# print(inputs)
# inputs = inputs.replace('plays','playing',1)
# inputs = inputs.replace('sport play','sport that is played',1)
# inputs = inputs.replace(' place','where',1)
# inputs = inputs.replace('when I was','When I was',1)

##输出切分后的中文句子
# tokenize_input = tokenizer.tokenize(inputs)
# print(len(tokenize_input))
# print("tokenize_input: ",tokenize_input)

##输出tokenize后的input，包括input token ids以及mask ids
input_ids = tokenizer(inputs, return_tensors="pt").input_ids
attention_mask = tokenizer(inputs, return_tensors="pt").attention_mask[0].detach().cpu().tolist()

##转置input ids以方便后续做torch.gather
input_ids_new = torch.transpose(input_ids,0,1).cuda()
print(input_ids_new[:,:])
# print(tokenizer.batch_decode(input))

##可以在这里对attention mask做一个修正，来帮助识别更多错误
# attention_mask[12] = 0
# attention_mask[35] = 0
mask = torch.tensor([attention_mask]).cuda()
input_ids = input_ids.cuda()
print("here:",input_ids.size())

##从model前馈得到outputs
outputs = model.forward(input_ids,attention_mask=mask).logits[0,:-1]
print('output: ',outputs.size())
result = outputs.detach().cpu().tolist()
print(len(result))
print(len(result[0]))

##进行两次排序，这样可以获得和vocab顺序一致的排名信息
for index in range(len(result)):
    index_list = [j for j in range(len(result[0]))]
    result[index] = sorted(index_list,key = lambda x:result[index][x],reverse=True)
    result[index] = sorted(index_list,key = lambda x:result[index][x])
result_indices = torch.tensor(result).cuda()
print(result_indices)

target_indice = torch.gather(result_indices,dim=1,index = input_ids_new[1:,:])
print("target_indice:",target_indice)
target_indice_new = torch.transpose(target_indice,0,1)[0,:].detach().cpu().numpy()
plt.figure(figsize=(14, 6))
plt.plot([i for i in range(len(target_indice_new))],target_indice_new,color='orange')
plt.xticks([i for i in range(len(target_indice_new))],tokenize_input,rotation=70)
plt.savefig('./figure.jpg')
# for index,i in enumerate(target_indice_new):
#     print(tokenize_input[index],'   ',i[0])

result =F.log_softmax(outputs,dim=1)
result_max = torch.max(result,dim=1,keepdim=True).values
result_min = torch.min(result,dim=1,keepdim=True).values
print(result_max-result_min)
result = torch.div(result - result_min,result_max-result_min)
top50_result = result.topk(50,dim=1)[0]
top50_mean = torch.mean(top50_result,dim=1,keepdim=False)
print(top50_mean)
# top50_result_index = result.topk(10,dim=1)[1]
# result_indices = torch.argsort(outputs,dim = 1,descending=True)
# print(result_indices.size())

# for i in range(1,20):
#     print(tokenize_input[i-1],input_ids[0][i])
#     print(top50_result_index[i,:],tokenizer.decode(top50_result_index[i,:]))
# print("top50_mean: ",top50_result.size())
# top50_max = torch.max(outputs,dim=1,keepdim=False).values
# top50_min = torch.min(outputs,dim=1,keepdim=False).values
top5_result = result.topk(5,dim=1)[0]
top5_mean = torch.mean(top5_result,dim=1,keepdim=False)
print(top5_mean)
# target_logit = torch.gather(outputs,dim=1,index = input_ids[:,1:])



filtered_logit = torch.div(top50_mean,top5_mean).cpu().tolist()
print(filtered_logit)
plt.figure()
plt.hist(filtered_logit,bins=50)
plt.savefig('./figure.jpg')

# # filtered_top50_mean = torch.div(top50_mean - top50_min,top50_max-top50_min)

# print("top50_mean: ",top50_mean.size())
# print("top50_mean: ",top50_mean)

# outputs_mean = torch.mean(result,dim=0,keepdim=False)
# print('outputs_mean: ',outputs_mean.size())
# average = torch.gather(outputs_mean,dim=0,index = input_ids[0,1:])
# print("average.size(): ",average.size())
# print("average: ",average)
# line = torch.gather(result,dim=1,index = input_ids[:,1:])
# print(line)
# final_result = target_indice.detach().cpu()[0].numpy()
# #.detach().cpu()[0].numpy()
# plt.plot([i for i in range(len(final_result))],final_result)
# plt.savefig('./figure.jpg')
# numpy.save('save.npy',final_result)
# problem_index = (numpy.argwhere(final_result < 1))
# for i in problem_index:
#    print(tokenize_input[i[0]] + '\n')
for index,item in enumerate(content['edits'][0][1]):
    try:
        print("错误{}：".format(str(index))+content['text'][item[0]:item[1]]+' --> '+item[2]+'\n')
    except:
        print(item)
# file.close()
#rets = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
#rets = rets[0].strip().replace(inputs, "")
# print(rets)
# print(len(outputs[0])/(time.time()-start_time))
