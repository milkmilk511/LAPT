# Your API key here
import openai
import json
from decimal import Decimal
from api import TransNASBenchAPI as API
import random
import numpy as np
import copy
import pandas as pd
import re
import ast




#REA-based NAS method
def random_spec():
    return random.choice(achi_list)

def random_combination(iterable, sample_size):
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), sample_size))
    return tuple(pool[i] for i in indices)

def random_spec():
    spec = []
    for i in range(6):
        spec.append(random.choice(ops_range[i]))
    return spec

def mutate_spec(old_spec):
    count = 0
    flag = True
    while (flag):
        idx_to_change = random.randrange(len(old_spec))
        if len(ops_range[idx_to_change]) == 1:
            continue
        entry_to_change = old_spec[idx_to_change]
        possible_entries = [x for x in ops_range[idx_to_change] if x != entry_to_change]
        new_entry = random.choice(possible_entries)
        new_spec = copy.copy(old_spec)
        new_spec[idx_to_change] = new_entry
        if new_spec not in hist:
            flag = False

        count = count+1
        if count>1000:
            break
        # if new_spec not in pop:
        #     flag = False
    return new_spec

def initilize(task,pop_size = 10):
    best_valids = [0.0]
    pop = []  # (validation, spec) tuples
    num_trained_models = 0

    for i in range(pop_size):
        spec = random_spec()
        arch = encode(spec)
        acc = api.get_single_metric(arch, task, metrics[task], mode='best')
        if 'room' in task:
            acc = 1+acc
        num_trained_models += 1
        pop.append((acc, spec))
        hist.append(spec)
        performance_list.append(acc)
        if acc > best_valids[-1]:
            best_valids.append(acc)
        else:
            best_valids.append(best_valids[-1])
    best_valids.pop(0)
    return best_valids,pop

def run_evolution_search(valid,pop,tournament_size=5,task = None):
    sample = random_combination(pop, tournament_size)
    best_spec = sorted(sample, key=lambda i: i[0])[-1][1]
    new_spec = mutate_spec(best_spec)
    arch = encode(new_spec)
    acc = api.get_single_metric(arch, task, metrics[task], mode='best')
    if 'room' in task:
        acc = 1 + acc

    # kill the oldest individual in the population.
    pop.append((acc, new_spec))
    pop.pop(0)
    hist.append(new_spec)
    performance_list.append(acc)
    # if info['valid-accuracy']>91.6:
    #     print(new_spec,syn[spec_to_idx[str(new_spec)]])

    if acc > valid[-1]:
        valid.append(acc)
    else:
        valid.append(valid[-1])



#Set you OpenAI
api_key =""
client = openai.OpenAI(api_key=api_key, base_url='https://api.gptapi.us/v1')
print(client.base_url)


#Extract architectual information from the Trans-bench-101
def encode(arch):
    head = '64-41414-'
    code = head+str(arch[0])+'_'+str(arch[1])+str(arch[2])+'_'+str(arch[3])+str(arch[4])+str(arch[5])
    return code
path2nas_bench_file = "transnas-bench_v10141024.pth"
api = API(path2nas_bench_file)
member_list = [9,11,12,14,15,16]
achi_list = []
for i in range(3256,7352):
    xarch = api.index2arch(i)
    arch = []
    for j in member_list:
        arch.append(int(xarch[j]))
    achi_list.append(arch)


best_val = {'class_scene': 0, 'class_object': 0, 'room_layout': 100, 'jigsaw': 0, 'segmentsemantic': 0, 'normal': 0,
            'autoencoder': 0}

metrics = {'class_scene': 'valid_top1', 'class_object': 'valid_top1',
           'room_layout': 'valid_neg_loss', 'jigsaw': 'valid_top1',
           'segmentsemantic': 'valid_mIoU', 'normal': 'valid_ssim',
           'autoencoder': 'valid_ssim'}

tasks = ['class_scene', 'class_object', 'room_layout', 'jigsaw']




#Prompt of the architectural search space
system_content = "You are an expert in the field of neural network design."

user_input = '''Your task is to assist me to learn the design principles from a set of given neural architectures.

Each architecture can be implementation via the followed Pytorch code:

#This architecture can be represented by a layer list, i.e., layer_list = [op0, op1, op2, op3, op4, op5], where each layer is with a specific operation such as pooling and convolution.

#There are 4 candidate operations for each layer:
operators = {
0: Zeroize()     # This operation simply outputs a tensor of zeros regardless of the input, which breaks the gradient flow between two nodes.
1: nn.Identity() # Skip Connection.
2: ReLUConvBN(channels, channels, kernal_size=1, stride=1, padding=0) # The input channels and output channels are the same.
3: ReLUConvBN(channels, channels, kernal_size=3, stride=1, padding=1) # The input channels and output channels are the same.
}

#A class is used to represent the architecture:
class architecture(nn.Module):
    def __init__(self, layer_list):
        super(architecture, self).__init__()
        self.layer0 = operators[layer_list[0]]
        self.layer1 = operators[layer_list[1]]
        self.layer2 = operators[layer_list[2]]
        self.layer3 = operators[layer_list[3]]
        self.layer4 = operators[layer_list[4]]
        self.layer5 = operators[layer_list[5]]

    def forward(self, input):
        s1 = self.layer0(input)
        s2 = self.layer1(input)
        s3 = self.layer3(input)
        s4 = self.layer2(s1)
        s5 = self.layer4(s1)
        s6 = self.layer5(s2+s4)
        output = s3+s5+s6
        return output

#we can obtain a specific model architecture via the followed function:
model_architecture = architecture(layer_list)
'''

#Obtain the architectural information on the computer version task Jigsaw
acc_list = []
for ind in achi_list:
    arch = encode(ind)
    acc = api.get_single_metric(arch, 'jigsaw', metrics['jigsaw'], mode='best')
    acc_list.append(-acc)
rank_index = np.argsort(np.array(acc_list))


#Input the architectural information of the established ones into the LLM
Steps = '''
Let's break this down step by step:

First, please analyse information flow in the architecture.

Then, the layer_list of 50 top-performing architectures are as follows:
{}
please analyse the common patterns among these architectures and further provide explanations.

Finally, please infer design principles that are generalized across all these patterns.

Moreover, according to these design principles, please give the ID of available operations for each layer to build top-performing architectures (the number of these operators is no more than 3).
'''.format(''.join(['architecture {}\n'.format(achi_list[rank_index[i]]) for i in range(50)]))

messages = [
    {"role": "system", "content": system_content},
    {"role": "user", "content": user_input+Steps},
]

#Obtain the intial design principles
res = client.chat.completions.create(model='gemini-1.5-pro', messages=messages, temperature=1,max_tokens=1000)
information = res.choices[0].message
messages.append(information)




#Set the target_task
targe_task = 'room_layout'  # cifar10-valid, cifar100, ImageNet16-120

#Setting of the REA NAS method
num_rounds = 5
length = 50
valid = np.zeros((num_rounds, length))


#Please input the search space according to the initial design principles, such as [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]] denoting the candidate available operators of each layer and here are total 6 layers in each architecture.
refine_ops_range =json.loads(input('\nPlease enter the LLM suggested operations:'))

# Conduct the architecture search in the refined search space

for index in range(num_rounds):
    random.seed(index)
    ops_range = copy.deepcopy(refine_ops_range)
    hist = []
    performance_list = []
    best_valids, pop = initilize(task=targe_task)
    current_best = best_valids[-1]
    for l in range(10):
        run_evolution_search(best_valids, pop, task=targe_task)
    total = 30
    NAS_messages = copy.deepcopy(messages)
    original = copy.deepcopy(messages)

    # Principle Adaptation
    while (total > 0):
        if total % 10 == 0:
            if best_valids[-1]>current_best:
                current_best = best_valids[-1]
                target_rank_index = np.argsort(np.array(performance_list))
                refinement = '''
                According to these given available operations, multiple architectures are built, whose layer_list are as follows:
                {}
                please update these design principles based on these obtained architectures.
    
                Finally, according to updated design principles, please suggest available operations for each layer to build better architectures.
    
                Please do not include anything other than the available operation ID list for each layer and the ID of the corresponding layer in your response. Then, all the lists are as a dictionary as the output.
                '''.format(''.join(['architecture {}\n'.format(hist[target_rank_index[-(i+1)]]) for i in range(15)]))
                NAS_messages.append({"role": "user", "content": refinement})
                res = client.chat.completions.create(model='gemini-1.5-pro', messages=NAS_messages, temperature=1, max_tokens=800)
                information = res.choices[0].message
                NAS_messages.append(information)

                pattern = re.compile(r'{.*?}', re.S)

                results = re.findall(pattern, information.content)[0]

                dic = ast.literal_eval(results)
                index_layer = 0
                for k,v in dic.items():
                    ops_range[index_layer] = v
                    index_layer = index_layer+1

            else:
                refinement = '''
                               According to these given available operations, it is hard to find the architectures with better performance.

                               Please suggest other available operations for each layer based on these original design principles.

                               Please do not include anything other than the available operation ID list for each layer and the ID of the corresponding layer in your response. Then, all the lists are as a dictionary as the output.
                               '''
                original.append({"role": "user", "content": refinement})
                res = client.chat.completions.create(model='gemini-1.5-pro', messages=original, temperature=0, n=1)
                information = res.choices[0].message
                NAS_messages.append(information)

                pattern = re.compile(r'{.*?}', re.S)

                results = re.findall(pattern, information.content)[0]

                dic = ast.literal_eval(results)
                index_layer = 0
                for k,v in dic.items():
                    ops_range[index_layer] = v
                    index_layer = index_layer+1
        run_evolution_search(best_valids, pop, task=targe_task)
        total = total - 1
    valid[index] = valid[index] + np.array(best_valids)[0:length].T
mean = np.mean(valid, axis=0)
std = np.std(valid, axis=0)

