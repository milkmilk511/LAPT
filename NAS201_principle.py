# Your API key here
import random

import openai
import json
from decimal import Decimal
from nas_201_api import NASBench201API as API
import numpy as np
import re
import copy
import ast
import pandas as pd


evaluation_trac=[]

_opname_to_index = {
    'none': 0,
    'skip_connect': 1,
    'nor_conv_1x1': 2,
    'nor_conv_3x3': 3,
    'avg_pool_3x3': 4
}

#Extract architectural information from the NAS-bench-201
def get_spec_from_arch_str(arch_str):
    nodes = arch_str.split('+')
    nodes = [node[1:-1].split('|') for node in nodes]
    nodes = [[op_and_input.split('~')[0] for op_and_input in node] for node in nodes]
    spec = [_opname_to_index[op] for node in nodes for op in node]
    return spec

#REA-based NAS method
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

def initilize(task,pop_size = 5):
    best_valids = [0.0]
    best_test = [0.0]
    pop = []  # (validation, spec) tuples
    num_trained_models = 0

    for i in range(pop_size):
        spec = random_spec()
        acc = database[str(spec)]
        num_trained_models += 1
        pop.append((acc, spec))
        hist.append(spec)
        performance_list.append(acc)
        if acc > best_valids[-1]:
            best_valids.append(acc)
            best_test.append(testbase[str(spec)])
        else:
            best_valids.append(best_valids[-1])
            best_test.append(best_test[-1])
    best_valids.pop(0)
    best_test.pop(0)
    return best_test,best_valids,pop

def run_evolution_search(valid,pop,tournament_size=2,task = None):
    sample = random_combination(pop, tournament_size)
    best_spec = sorted(sample, key=lambda i: i[0])[-1][1]
    new_spec = mutate_spec(best_spec)
    acc = database[str(new_spec)]

    # kill the oldest individual in the population.
    pop.append((acc, new_spec))
    pop.pop(0)
    hist.append(new_spec)
    performance_list.append(acc)
    # if info['valid-accuracy']>91.6:chiy
    #     print(new_spec,syn[spec_to_idx[str(new_spec)]])

    if acc > valid[-1]:
        valid.append(acc)
        best_test.append(testbase[str(new_spec)])
    else:
        valid.append(valid[-1])
        best_test.append(best_test[-1])

#Set you OPENAI
api_key =""
client = openai.OpenAI(api_key=api_key, base_url='https://api.gptapi.us/v1')
print(client.base_url)


#Prompt of the architectural search space
system_content = "You are an expert in the field of neural network design."

user_input = '''Your task is to assist me to learn the design principles from a set of given neural architectures.

Each architecture can be implementation via the followed Pytorch code:

#This architecture can be represented by a layer list, i.e., layer_list = [op0, op1, op2, op3, op4, op5], where each layer is with a specific operation such as pooling and convolution.

#There are 5 candidate operations for each layer:
operators = {
0: Zeroize()     # This operation simply outputs a tensor of zeros regardless of the input, which breaks the gradient flow between two nodes.
1: nn.Identity() # Skip Connection.
2: ReLUConvBN(channels, channels, kernal_size=1, stride=1, padding=0) # The input channels and output channels are the same.
3: ReLUConvBN(channels, channels, kernal_size=3, stride=1, padding=1) # The input channels and output channels are the same.
4: nn.AvgPool2d(kernel_size=3, stride=1, padding=1)                   # This operation does not change the spatial resolution.
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
#Source task and the target task. Due the reduce of the search space may lead to the elimination of the global optimal architecture, we target to the top-0.1% architecture in the search space and record the time cost.
with open('./nasbench201_imagenet.json', 'r') as infile:
    data = json.loads(infile.read())

#Build the database of the target task
target_task = 'cifar10-valid'  # 'cifar10-valid', 'cifar100', 'ImageNet16-120'
with open('./nasbench201_cifar10.json', 'r') as infile:
    data = json.loads(infile.read())
search_space = []
acc_list = []
database = {}
testbase = {}
models = ()
for d in data:
    spec = get_spec_from_arch_str(d)
    search_space.append(spec)
    acc_list.append(-data[d]['val_acc_200'] / 100)
    database[str(spec)] = data[d]['val_acc_200'] / 100
    testbase[str(spec)] = data[d]['test_acc_200'] / 100
global_optimal = 0.9160 #cifar10: 0.9160 CIFAR-100: 0.7349


#validation accuracy of the top-0.1% architecture
top_01 = 0.9142 #CIFAR100:0.9142  CIFAR10:0.7272


#Record the architectural information of the estalished architectures
search_space = []
acc_list = []
database = {}
for d in data:
    spec = get_spec_from_arch_str(d)
    search_space.append(spec)
    acc_list.append(-data[d]['val_acc_200'] / 100)
    database[str(spec)] = data[d]['val_acc_200'] / 100
index_list = np.argsort(np.array(acc_list))
arch_code_len = len(search_space[0])
print("the length of architecture codes:", arch_code_len)
print("total architectures:", len(search_space))


pop_size = 50
pop = []
for ind in range(pop_size):
    pop.append(search_space[index_list[ind]])

#Input the architectural information of the established ones into the LLM
Steps = '''
Let's break this down step by step:

First, please analyse information flow in the architecture.

Then, the layer_list of 50 top-performing architectures are as follows:
{}
please analyse the common patterns among these architectures and further provide explanations.

Finally, please infer design principles of top architectures from these comment patterns.

Moreover, according to these design principle, please suggest available operations for each layer to build top-performing architectures (the number of availalbe operations for each layer is no more than 2).
'''.format(''.join(['{}\n'.format(arch) for arch in pop]))

messages = [
    {"role": "system", "content": system_content},
    {"role": "user", "content": user_input+Steps},
]


#Obtain the intial design principles
res = client.chat.completions.create(model='gpt-4-32k', messages=messages, temperature=0, n=1)
information = res.choices[0].message
messages.append(information)



#Setting of the REA method
num_rounds = 5
length = 20
total_num = np.zeros(num_rounds)
valid = np.zeros((num_rounds, length))
test = np.zeros((num_rounds, length))

#Please input the search space according to the initial design principles, such as [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]] denoting the candidate available operators of each layer and here are total 6 layers in each architecture.
refine_ops_range = json.loads(input('\nPlease enter the LLM suggested operations:'))
evaluations = 0
ave_test_acc = 0



# Conduct the architecture search in the refined search space
for index in range(0,num_rounds):
    random.seed(index)
    ops_range = copy.deepcopy(refine_ops_range)

    # initialization
    hist = []
    performance_list = []
    best_test,best_valids, pop = initilize(task=target_task)
    current_best = best_valids[-1]
    for l in range(5):
        run_evolution_search(best_valids, pop, task=target_task)
    total = 10
    NAS_messages = copy.deepcopy(messages)
    original = copy.deepcopy(messages)

    #Principle Adaptation
    while (total > 0):
        if total % 5 == 0:
            if best_valids[-1] > current_best:
                current_best = best_valids[-1]
                target_rank_index = np.argsort(np.array(performance_list))
                refinement = '''
                According to these given available operations, multiple architectures are built, whose layer_list are as follows:
                {}
                please update these design principles based on these obtained architectures.

                Finally, according to updated design principles, please suggest available operations for each layer to build better architectures.

                Please do not include anything other than the available operation ID list for each layer and the ID of the corresponding layer in your response. Then, all the lists are as a dictionary as the output.
                '''.format(''.join(['architecture {}\n'.format(hist[target_rank_index[-(i + 1)]]) for i in range(5)]))

                NAS_messages.append({"role": "user", "content": refinement})
                res = client.chat.completions.create(model='gpt-4-32k', messages=NAS_messages, temperature=0, n=1)
                information = res.choices[0].message
                NAS_messages.append(information)

                pattern = re.compile(r'{.*?}', re.S)

                results = re.findall(pattern, information.content)[0]

                dic = ast.literal_eval(results)
                index_layer = 0
                for k, v in dic.items():
                    ops_range[index_layer] = v
                    index_layer = index_layer + 1

            else:
                refinement = '''
                                   According to these given available operations, it is hard to find the architectures with better performance.

                                   Please suggest other available operations for each layer based on these original design principles.

                                   Please do not include anything other than the available operation ID list for each layer and the ID of the corresponding layer in your response. Then, all the lists are as a dictionary as the output.
                                   '''
                original.append({"role": "user", "content": refinement})
                res = client.chat.completions.create(model='gpt-4-32k', messages=original, temperature=0, n=1)
                information = res.choices[0].message
                NAS_messages.append(information)

                pattern = re.compile(r'{.*?}', re.S)

                results = re.findall(pattern, information.content)[0]

                dic = ast.literal_eval(results)
                index_layer = 0
                for k, v in dic.items():
                    ops_range[index_layer] = v
                    index_layer = index_layer + 1
        run_evolution_search(best_valids, pop, task=target_task)
        total = total - 1

    test[index] = test[index] + np.array(best_test)[0:length].T
    valid[index] = valid[index] + np.array(best_valids)[0:length].T

    #Record the time cost to reach the global optimal architecture or the top-0.1% ones
    num_global = None
    num_top = None
    for index_num in range(20):
        if valid[index][index_num]>global_optimal and num_global is None:
            num_global = index_num+1
        if valid[index][index_num]>top_01 and num_top is None:
            num_top = index_num+1
    if num_global is not None:
        evaluations = evaluations+num_global
        ave_test_acc = ave_test_acc+test[index][num_global-1]
    elif num_top is not None:
        evaluations = evaluations+num_top
        ave_test_acc = ave_test_acc+test[index][num_top - 1]
    else:
        evaluations = evaluations+20
        ave_test_acc = ave_test_acc + test[index][-1]

acc_mean = ave_test_acc/num_rounds
num_mean = evaluations/num_rounds
print(acc_mean,num_mean)




