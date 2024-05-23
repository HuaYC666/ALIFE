import pandas as pd
import numpy as np
import math
import torch
import random
from torch import nn
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch.cuda as cuda
import copy

device = torch.device("cuda:0")
device_cpu = torch.device("cpu")
expr_train_addr = "Address of training gene expression data"   #for example: "/home/usr/expr_train.txt"
clin_train_addr = "Address of training clinic data"  #for example: "/home/usr/clin_train.txt"
expr_test_addr = "Address of testing gene expression data"  #for example: "/home/usr/expr_test.txt"
clin_test_addr = "Address of testing clinic data"  #for example: "/home/usr/clin_test.txt"
mask_addr = "Address of pathway information"  #for example: "/home/usr/pathway_matrix_HALLMARK.csv"
save_addr = "Address of saving model output"  #for example: "/home/usr"
n_pathway_embeding = 220
num_epochs_ae = 4000
batch_size_ae = 100
lr_ae = 0.01
num_epochs_sup = 2000
batch_size_sup = 100
lr_sup = 0.0001





def get_gpu_memory_usage():
    allocated = cuda.memory_allocated()
    reserved = cuda.memory_reserved()
    return allocated, reserved


'''
Read the pathway CSV file and return the mask matrix list along with all gene names (required for standardized expression matrix)
'''


def pathway_read(path, n_mask_embeding=24):

    pathway_data = pd.read_csv(path)
    pathway_data = pathway_data.drop(labels=["Unnamed: 0"], axis=1)
    pathway_name = pathway_data.columns
    pathway_data = np.array(pathway_data)
    pathway_data = pathway_data.transpose(1, 0)

    all_gene_names = []
    all_pathway = []
    for pathway_id in range(pathway_data.shape[0]):
        all_gene_names += list(pathway_data[pathway_id, :])
        pathway_genes = [gene for gene in list(pathway_data[pathway_id, :]) if gene != "Havenogene"]
        all_pathway.append(pathway_genes)
    all_gene_names = set(all_gene_names)  # 去除重复基因
    all_gene_names = [gene for gene in all_gene_names if gene != "Havenogene"]

    pathway_embeding_dimension = n_mask_embeding
    mask_mat_list = []
    for pathway_id in range(len(all_pathway)):

        genes_in_pathway = all_pathway[pathway_id]
        gene_id_nomask = [all_gene_names.index(gene) for gene in genes_in_pathway]


        mask_mat = torch.zeros([len(all_gene_names), pathway_embeding_dimension], dtype=torch.float32)
        for gene_id in gene_id_nomask:
            mask_mat[gene_id, :] = 1.
        mask_mat_list.append(mask_mat)
    mask_multimat = torch.stack(mask_mat_list)
    return mask_multimat, all_gene_names, pathway_name


'''
Standardized expression matrix format (expanding pathway genes that do not exist in expression data into the expression matrix) 
expr_mat: expression matrix,of which row represents sample, column represents gene
gene_name: a list of genes in expr_mat (arranged in their column order) 
all_gene names: all genes contained in all pathways 
The returned matrix is in tensor format
'''
def standard_expr_mat(expr_mat, gene_name, all_gene_names):
    expr_mat = torch.tensor(expr_mat, dtype=torch.float32)
    n_sample, _ = expr_mat.shape
    expr_tensor = torch.zeros([n_sample, len(all_gene_names)])
    gene_match = 0
    for id in range(len(gene_name)):
        if gene_name[id] in all_gene_names:
            gene_match += 1
            gene_position = all_gene_names.index(gene_name[id])
            expr_tensor[:, gene_position] = expr_mat[:, id]
    gene_match_rate = gene_match / len(all_gene_names)
    print("GeneMatchRate:", gene_match_rate)
    return expr_tensor


'''
Set seed
'''
def set_seed(seed=666):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed(1)
'''
Define functions for batch construction 
The inputted myexpr is the expression data (tensor), OStime is the survival time list, OS is the outcome list, batch_size is the batch size, and the_event_id is the sample location where the outcome event occurred 
The output consists of three lists, representing the slices of data, as well as their corresponding survival time and outcomes, as well as the positions of the samples corresponding to the occurrence of outcome events
'''
def batch_construct(myexpr, OStime, OS, batch_size, the_event_id, seed):
    random.seed(seed)
    expr = myexpr
    n_gene = myexpr.shape[1]

    my_event_id = the_event_id
    id_all = list(range(len(OStime)))
    n_batch = round(len(OStime) / batch_size)
    OStime_tensor = torch.tensor(OStime)
    OS_tensor = torch.tensor(OS)
    expr_list = []
    OStime_list = []
    OS_list = []
    event_id_list = []
    no_event_id = [i for i in id_all if i not in my_event_id]
    event_batchsize = int(batch_size * len(my_event_id) / len(id_all))  # 让最后一个batch前的每一次的event_batchsize偏少
    noevent_batchsize = batch_size - event_batchsize

    for batch in range(n_batch):
        if batch != (n_batch - 1):
            event_id_for_batch = random.sample(my_event_id, event_batchsize)
            noevent_id_for_batch = random.sample(no_event_id, noevent_batchsize)

            id_for_batch = event_id_for_batch + noevent_id_for_batch
            id_for_batch.sort()
            no_event_id = [i for i in no_event_id if i not in noevent_id_for_batch]
            my_event_id = [i for i in my_event_id if i not in event_id_for_batch]

            expr_list.append(expr[id_for_batch, :n_gene])
            OStime_list.append(OStime_tensor[id_for_batch].tolist())
            OS_list.append(OS_tensor[id_for_batch].tolist())

            event_id = [i for i in range(len(id_for_batch)) if OS_tensor[id_for_batch][i] == 1]
            event_id_list.append(event_id)

        if batch == (n_batch - 1):
            event_id_for_batch = my_event_id
            noevent_id_for_batch = no_event_id

            id_for_batch = event_id_for_batch + noevent_id_for_batch
            id_for_batch.sort()

            expr_list.append(expr[id_for_batch, :n_gene])
            OStime_list.append(OStime_tensor[id_for_batch].tolist())
            OS_list.append(OS_tensor[id_for_batch].tolist())
            event_id = [i for i in range(len(id_for_batch)) if OS_tensor[id_for_batch][i] == 1]
            event_id_list.append(event_id)

    return expr_list, OStime_list, OS_list, event_id_list


'''
Define a function, tensorlist_connect 
Used to connect the tensor matrices in the list together
'''
def tensorlist_connect(atten_list, dimension):
    atten_list_samedimen = atten_list[0:(len(atten_list) - 1)]
    atten_0 = torch.stack(atten_list_samedimen, dimension)
    atten_0 = torch.reshape(atten_0,
                            (atten_0.shape[0] * atten_0.shape[1], atten_0.shape[2], atten_0.shape[3], atten_0.shape[4]))
    atten_1 = atten_list[len(atten_list) - 1]
    atten = torch.cat([atten_0, atten_1], 0)
    return atten


'''
Building the autoencoder 
'''
class AE_Embedding(nn.Module):
    def __init__(self, n_allgene, n_embeding, device):
        super(AE_Embedding, self).__init__()
        self.encoder = nn.parameter.Parameter(
            torch.randn([n_allgene, n_embeding], dtype=torch.float32, device=device))
        self.decoder = nn.parameter.Parameter(
            torch.randn([n_embeding, n_allgene], dtype=torch.float32, device=device))

    def forward(self, x):
        embeding = torch.matmul(x, self.encoder)
        x_recon = torch.matmul(embeding, self.decoder)
        return embeding, x_recon


'''
Calculation of embedded features in the path 
Here, x inputs a matrix of rows representing samples and columns representing genes 
The output is [n_sample, n_embedding, n_pathway+1], where CLS is concatenated
'''
class Pathway_Embedding(nn.Module):
    def __init__(self, mask_mat_mrna,  linear_mrna, device):
        super(Pathway_Embedding, self).__init__()
        self.mask_mat_mrna = mask_mat_mrna
        self.n_pathway, self.n_mrna, self.n_embeding_mrna = mask_mat_mrna.shape
        # self.activation = nn.Tanh()
        self.activation = nn.Sigmoid()
        self.linear_mrna = nn.parameter.Parameter(
            linear_mrna.unsqueeze(0).to(device), requires_grad=False)

        self.activation = nn.Sigmoid()

        self.ln_mrna = nn.LayerNorm([self.n_embeding_mrna, self.n_pathway])


    def forward(self, x_mrna):
        n_sample = x_mrna.shape[0]
        #myCLS = self.CLS.repeat(n_sample, 1, 1)


        mylinear = self.linear_mrna.repeat(self.n_pathway, 1, 1)
        linear_mask = torch.mul(self.mask_mat_mrna, mylinear)
        embeding = torch.matmul(x_mrna, linear_mask)
        embeding_0 = embeding.permute(1, 2, 0)
        embeding = self.ln_mrna(embeding_0)


        return embeding, embeding_0




'''
MLP module construction 
Here, the input x is the feature of the previous layer 
Input as [n_sample, n_pathway+1, n_embedding]
Output as [n_sample, n_pathway+1, n_embedding]
'''
class MLP(nn.Module):
    def __init__(self, n_embeding):
        super(MLP, self).__init__()
        self.linear0 = nn.Linear(n_embeding, n_embeding)
        self.linear1 = nn.Linear(n_embeding, n_embeding)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x_input = self.linear0(x)
        x_input = self.gelu(x_input)
        x_input = self.dropout(x_input)
        x_input = self.linear1(x_input)
        x_input = self.gelu(x_input)
        return (x_input)


'''
MHA module construction 
The input is [n_sample, n_embedding, n_pathway+1] 
Output as [n_sample, n_embedded, n_pathway+1]
'''
class MHA(nn.Module):
    def __init__(self, n_embeding, n_pathway, n_head, device):
        super(MHA, self).__init__()
        self.n_embeding = n_embeding
        self.n_pathway = n_pathway
        self.n_head = n_head
        # self.dimension_perhead = int(self.n_embeding/self.n_head)
        self.Q = nn.parameter.Parameter(torch.randn([n_head, n_embeding, n_embeding], device=device))
        self.K = nn.parameter.Parameter(torch.randn([n_head, n_embeding, n_embeding], device=device))
        self.V = nn.parameter.Parameter(torch.randn([n_head, n_embeding, n_embeding], device=device))
        self.softmax = nn.Softmax(3)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(n_head * n_embeding, n_embeding)

    def forward(self, x):
        x_input = x.unsqueeze(1)
        x_input = x_input.repeat(1, self.n_head, 1, 1)
        x_input = x_input.permute(0, 1, 3, 2)
        q = torch.matmul(x_input, self.Q)
        k = torch.matmul(x_input, self.K)
        v = torch.matmul(x_input, self.V)
        cor = torch.matmul(q, k.permute(0, 1, 3, 2)) / (self.n_embeding ** 0.5)
        atten = self.softmax(cor)


        atten_dropout = self.dropout(atten)
        output = torch.matmul(atten_dropout, v)
        output = output.permute(0, 2, 1, 3)
        nsample = output.shape[0]
        output = torch.reshape(output, (nsample, self.n_pathway + 1, self.n_head * self.n_embeding))
        output = self.linear(output)
        output = output.permute(0, 2, 1)
        return output, atten




'''
ENCODER BLOCK 
Here, the input x is [n_sample, n_embedding, n_pathway+1] 
Output as [n_sample, n_embedding, n_pathway+1]
'''

class EncoderBlock(nn.Module):
    def __init__(self, n_embeding, n_head, n_pathway, MLP, MHA, device):
        super(EncoderBlock, self).__init__()
        self.n_embeding = n_embeding
        self.n_head = n_head
        self.n_pathway = n_pathway
        self.MLP = MLP(n_embeding)
        self.MHA = MHA(n_embeding, n_pathway, n_head, device)
        self.ln0 = nn.LayerNorm([n_embeding, n_pathway + 1])
        self.ln1 = nn.LayerNorm([n_embeding, n_pathway + 1])
        self.dropout = nn.Dropout(0)

    def forward(self, x):
        x_input = self.ln0(x)
        x_input = x_input.permute(0, 2, 1)
        x_input0 = self.MLP(x_input)
        x_input0 = self.dropout(x_input0)
        x_input0 = x_input0 + x_input
        x_input0 = x_input0.permute(0, 2, 1)
        x_input1 = self.ln1(x_input0)
        x_input1, atten = self.MHA(x_input1)
        output = x_input1 + x_input0
        return output, atten





'''
Overall network construction of ALIFE 
Here, input x as [n_sample, gene] 
Output as [n_sample, 1]
'''
class ALIFE(nn.Module):
    def __init__(self, n_embeding, n_head, n_pathway, mask_mat_mrna, Pathway_Embeding, EncoderBlock,
                 linear_mrna,  MLP, device,n_MLP):
        super(ALIFE, self).__init__()
        self.n_embeding = n_embeding
        self.n_head = n_head
        self.n_pathway = n_pathway
        self.Pathway_Embeding = Pathway_Embeding(mask_mat_mrna,linear_mrna,  device)
        self.EncoderBlock0 = EncoderBlock(n_embeding, n_head, n_pathway, MLP, MHA, device)
        self.EncoderBlock1 = EncoderBlock(n_embeding, n_head, n_pathway, MLP, MHA, device)
        #self.EncoderBlock2 = EncoderBlock(n_embeding, n_head, n_pathway, MLP, MHA, device)
        self.ln = nn.LayerNorm([n_embeding, n_pathway + 1])
        self.linear2 = nn.Linear(n_embeding, 1)
        self.linear1 = nn.Linear(n_embeding, n_embeding)
        self.linear0 = nn.Linear(n_embeding, n_embeding)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout0 = nn.Dropout(0.2)
        self.dropout1 = nn.Dropout(0)
        self.dropout2 = nn.Dropout(0)
        self.MLP1 = MLP(n_embeding)
        self.MLP2 = MLP(n_embeding)
        self.MLP3 = MLP(n_embeding)
        self.n_MLP = n_MLP
        self.CLS = nn.parameter.Parameter(
             torch.randn([1, self.n_embeding, 1], device=device))

    def forward(self, x_mrna):
        n_sample = x_mrna.shape[0]
        my_cls = self.CLS.repeat(n_sample ,1,1)

        embeding, embeding_0 = self.Pathway_Embeding(x_mrna)
        embeding = torch.cat([my_cls,embeding],2)

        embeding = embeding.permute(0, 2, 1)

        if self.n_MLP == 1:
            embeding0 = self.MLP1(embeding)
            embeding = embeding0 + embeding
        if self.n_MLP == 2:
            embeding0 = self.MLP1(embeding)
            embeding = embeding0 + embeding
            embeding0 = self.MLP2(embeding)
            embeding = embeding0 + embeding
        if self.n_MLP == 3:
            embeding0 = self.MLP1(embeding)
            embeding = embeding0 + embeding
            embeding0 = self.MLP2(embeding)
            embeding = embeding0 + embeding
            embeding0 = self.MLP3(embeding)
            embeding = embeding0 + embeding

        embeding = embeding.permute(0, 2, 1)
        output1, atten1 = self.EncoderBlock0(embeding)
        output1, atten2 = self.EncoderBlock1(output1)
        #output1, atten3 = self.EncoderBlock2(output1)
        output = self.ln(output1)
        CLS = output[:, :, 0]
        CLS = self.dropout0(CLS)
        # risk = self.linear0(CLS)
        # risk = self.relu(risk)
        # risk = self.dropout1(risk)

        # risk = self.linear1(risk)
        # risk  = self.relu(risk)
        # risk = self.dropout2(risk)
        risk = self.linear2(CLS)
        risk = 10 * self.sigmoid(risk)
        return risk, atten2




'''
Define Cox loss function
'''
def coxloss_mask(OStime, event_id):
    n_sample = len(OStime)
    mask_mat = torch.ones([n_sample, n_sample])

    event_OStime = []
    for i in event_id:
        event_OStime.append(OStime[i])

    for id in event_id:
        OStime_for_id = OStime[id]
        if OStime[id:].count(OStime_for_id) > 1:
            sametime_id = [index for index, value in enumerate(OStime) if
                           value == OStime_for_id and index > id and index in event_id]
            mask_mat[id, sametime_id] = 0
    return mask_mat



class coxloss(nn.Module):
    def __init__(self, OStime, event_id, device):
        super(coxloss, self).__init__()
        self.event_id = event_id
        self.mask = coxloss_mask(OStime, event_id).to(device)

    def forward(self, x):
        x_adjust = x
        x_exp = torch.exp(x_adjust)
        n_sample = x.shape[0]
        x_exp = x_exp.unsqueeze(0)
        x_exp_mat = x_exp.repeat(n_sample, 1)


        x_risk = x_adjust[self.event_id]


        x_tril_mat = torch.triu(x_exp_mat, 1)
        x_tril_mat = torch.mul(x_tril_mat, self.mask)
        x_tril_mat_rowsum = torch.sum(x_tril_mat, 1)
        if self.event_id[-1] == (len(x_tril_mat_rowsum) - 1):
            x_tril_mat_logrowsum = torch.log(x_tril_mat_rowsum[self.event_id][:-1])
        if self.event_id[-1] != (len(x_tril_mat_rowsum) - 1):
            x_tril_mat_logrowsum = torch.log(x_tril_mat_rowsum[self.event_id])

        loss = - x_risk.sum() + x_tril_mat_logrowsum.sum()

        return loss, x_exp, x_tril_mat_rowsum, x_tril_mat_logrowsum, x_risk, x_tril_mat



# extract mask matrix and all gene names
mask_multimat_mrna, all_mrna_names, pathway_name = pathway_read(
    mask_addr, n_pathway_embeding)


#read train dataset
expr_train = pd.read_table(
    expr_train_addr, sep="\t", engine='python')
expr_train_np = np.array(expr_train)
gene_name = list(expr_train.columns)
#Standardized format
expr_train_tensor = standard_expr_mat(expr_train_np, gene_name, all_mrna_names)
expr_train_tensor_ori_order = copy.deepcopy(expr_train_tensor)

clin_train = pd.read_table(
    clin_train_addr, sep="\t", engine='python')
clin_train_np = np.array(clin_train)
train_rfs = list(clin_train_np[:, 2])
train_outcome = list(clin_train_np[:, 1])
for i in range(len(train_outcome)):
    train_outcome[i] = int(train_outcome[i])

# sort gene expression arrays and clinical arrays in chronological order
train_rfs_order = list(np.argsort(train_rfs))
nsample, ngene = expr_train_tensor.shape
expr_train_ordered = torch.zeros([nsample, ngene], dtype=torch.float32)
train_outcome_ordered = []
for i in range(len(train_rfs_order)):
    expr_train_ordered[i, :] = expr_train_tensor[train_rfs_order[i], :]
    train_outcome_ordered += [train_outcome[train_rfs_order[i]]]
train_outcome = train_outcome_ordered
expr_train_tensor = expr_train_ordered
train_rfs.sort()
event_id = []
for i in range(len(train_outcome)):
    if train_outcome[i] == 1:
        event_id.append(i)


#read test dataset
expr_test = pd.read_table(
    expr_test_addr, sep="\t", engine='python')
expr_test_np = np.array(expr_test)
gene_name = list(expr_test.columns)
#Standardized format
expr_test_tensor = standard_expr_mat(expr_test_np, gene_name, all_mrna_names)


clin_test = pd.read_table(
    clin_test_addr, sep="\t", engine='python')
clin_test_np = np.array(clin_test)
test_rfs = list(clin_test_np[:, 2])
test_outcome = list(clin_test_np[:, 1])
for i in range(len(test_outcome)):
    test_outcome[i] = int(test_outcome[i])


'''
start train
'''
n_head = [3, 6, 9]
n_MLP = [1, 2, 3]
n_model = [0,1,2]
model_size = ["light","base","large"]

for model_id in n_model:
    # start training of autoencoder
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = num_epochs_ae
    step_size = num_epochs / 2
    criterion = nn.L1Loss()
    model_ae_mrna = AE_Embedding(len(all_mrna_names), n_pathway_embeding, device)

    batch_size = batch_size_ae
    model_ae_mrna.to(device)

    optimizer_mrna = torch.optim.Adam(model_ae_mrna.parameters(), lr=lr_ae)

    lr_scheduler_sl_mrna = torch.optim.lr_scheduler.StepLR(optimizer_mrna, step_size=step_size, gamma=0.1)

    myloss_train = []
    for epoch in range(num_epochs):
        mrna_list, _, _, _ = batch_construct(expr_train_tensor,
                                             train_rfs, train_outcome, batch_size, event_id, epoch)
        myloss = []
        for iter in range(len(mrna_list)):
            optimizer_mrna.zero_grad()

            mrna_batch = mrna_list[iter]
            embeding_mrna, mrna_recon = model_ae_mrna(mrna_batch.to(device))

            loss_mrna = criterion(mrna_recon, mrna_batch.to(device))  # + loss_reg*n_sample

            myloss.append(loss_mrna.item())

            loss_mrna.backward(retain_graph=True)

            optimizer_mrna.step()

        loss_average = np.average(myloss)
        if epoch % 10 == 0:
            print('epoch {}, loss {}, learningRate {}'.format(epoch, loss_average,
                                                              lr_scheduler_sl_mrna.get_last_lr()))

        myloss_train.append(loss_average)
        # 更新学习率
        lr_scheduler_sl_mrna.step()

    for n, p in model_ae_mrna.named_parameters():
        p.requires_grad = False

    '''
    训练模型
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")
    linear_mrna = model_ae_mrna.encoder.clone().detach()

    model = ALIFE(n_embeding=n_pathway_embeding, n_head=n_head[model_id], n_pathway=len(pathway_name),  # 嵌入数必须要可以整除头数
                  mask_mat_mrna=mask_multimat_mrna.to(device),
                  Pathway_Embeding=Pathway_Embedding, EncoderBlock=EncoderBlock,
                  linear_mrna=linear_mrna,
                  MLP=MLP, device=device,n_MLP=n_MLP[model_id])
    model.to(device)
    batch_size = batch_size_sup
    num_epochs = num_epochs_sup
    step_size = num_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_sup)
    lr_scheduler_sl = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                                                      gamma=0.1)  # 学习率每5个epoch衰减成原来的1/10


    train_outcome_tensor = torch.tensor(train_outcome)
    train_outcome_tensor = torch.nn.functional.one_hot(train_outcome_tensor)
    train_outcome_tensor = torch.tensor(train_outcome_tensor, dtype=torch.float32)
    train_outcome_tensor = torch.tensor(train_outcome_tensor, dtype=torch.float32)

    myloss_train = []
    risk_train_all = []
    atten_train_all = []
    os_train_all = []
    ostime_train_all = []
    atten_train_all_ori = []
    model.train()
    for epoch in range(num_epochs):
        mrna_list, OStime_list, OS_list, event_id_list = batch_construct(expr_train_tensor,
                                                                         train_rfs, train_outcome,
                                                                         batch_size, event_id, epoch)
        have_event = True
        # Detecting the occurrence of events
        for event_id in event_id_list:
            if len(event_id) == 0:
                have_event = False
        if not have_event:
            print("No_event!Resample!")

        myloss = []
        for iter in range(len(mrna_list)):
            criterion = coxloss(OStime=OStime_list[iter],
                                event_id=event_id_list[iter],
                                device=device)
            optimizer.zero_grad()

            risk, atten= model(mrna_list[iter].to(device))
            risk = risk.squeeze(1)
            risk = risk - torch.median(risk)
            risk_train = risk.tolist()
            outcome_train = train_outcome
            os_train = train_rfs

            loss, x_exp, x_tril_mat_rowsum, x_tril_mat_logrowsum, x_risk, x_tril_mat = criterion(
                risk)
            myloss.append(loss.item())

            loss.backward(retain_graph=True)
            # gradient cropping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)

            optimizer.step()

        if epoch % 10 == 0:
            print('epoch {}, loss {}, learningRate {}'.format(epoch, np.average(myloss),
                                                              lr_scheduler_sl.get_last_lr()))

        myloss_train.append(np.average(myloss).item())
        # 更新学习率
        lr_scheduler_sl.step()

    plt.plot(list(range(len(myloss_train))), myloss_train)
    plt.show()

    model.eval()
    # obtain output of train dataset
    n_sample_predict = 10  # predict 10 samples each time
    n_batch = int(expr_train_tensor_ori_order.shape[0] / n_sample_predict)
    # generative sample id to be predicted
    predict_id = []
    for i in range(n_batch):
        if i != (n_batch - 1):
            id_start = i * n_sample_predict
            id_end = (i + 1) * n_sample_predict
            predict_id.append(list(range(id_start, id_end)))
        if i == (n_batch - 1):
            id_start = i * n_sample_predict
            id_end = expr_train_tensor_ori_order.shape[0]
            predict_id.append(list(range(id_start, id_end)))

    output_list = []
    atten_train_list = []
    with torch.no_grad():
        model.eval()
        for ids in predict_id:
            output, atten_train = model(expr_train_tensor_ori_order[ids, :].to(device))
            output = output.squeeze(1)
            output = output.tolist()
            output_list += output
            atten_train_list.append(atten_train.to(device_cpu))

    atten_train = tensorlist_connect(atten_train_list, 0)

    risk_train = output_list
    outcome_train = train_outcome
    os_train = train_rfs

    # make data frame of predicted risk
    output_train = [risk_train, outcome_train, os_train]

    output_train = np.array(output_train)
    output_train = output_train.transpose(1, 0)

    data_train = pd.DataFrame(output_train)

    data_train.columns = ["risk", "outcome", "os"]

    data_train_addr = save_addr + "/output_risk_train_model"+ str(model_size[model_id]) + ".csv"
    data_train.to_csv(
        data_train_addr, index=True)

    # save attention matrix
    attention_pathway = atten_train[:, :, 0, 1:]
    attention_pathway_mean = torch.mean(attention_pathway, 1)
    attention_pathway_mean = attention_pathway_mean.detach().to(device_cpu).numpy()
    data_atten_addr = save_addr + "/output_atten_train_model"+ str(model_size[model_id]) + ".csv"
    data_atten = pd.DataFrame(attention_pathway_mean)
    data_atten.columns = pathway_name
    data_atten.to_csv(
        data_atten_addr,
        index=True)

    # obtain output of test dataset
    n_sample_predict = 10  # predict 10 samples every time
    n_batch = int(expr_test_tensor.shape[0] / n_sample_predict)
    # generative sample id to be predicted
    predict_id = []
    for i in range(n_batch):
        if i != (n_batch - 1):
            id_start = i * n_sample_predict
            id_end = (i + 1) * n_sample_predict
            predict_id.append(list(range(id_start, id_end)))
        if i == (n_batch - 1):
            id_start = i * n_sample_predict
            id_end = expr_test_tensor.shape[0]
            predict_id.append(list(range(id_start, id_end)))

    output_list = []
    atten_test_list = []
    with torch.no_grad():
        model.eval()
        for ids in predict_id:
            output, atten_test= model(expr_test_tensor[ids, :].to(device))
            output = output.squeeze(1)
            output = output.tolist()
            output_list += output
            atten_test_list.append(atten_test.to(device_cpu))

    atten_test = tensorlist_connect(atten_test_list, 0)

    risk_test = output_list
    outcome_test = test_outcome
    os_test = test_rfs


    # 制作训练集和测试集相关指标表格
    output_test = [risk_test, outcome_test, os_test]

    output_test = np.array(output_test)
    output_test = output_test.transpose(1, 0)

    data_test = pd.DataFrame(output_test)

    data_test.columns = ["risk", "outcome", "os"]

    data_test_addr = save_addr + "/output_risk_test_model"+ str(model_size[model_id]) + ".csv"
    data_test.to_csv(
        data_test_addr, index=True)

    # 保存注意力矩阵
    attention_pathway = atten_test[:, :, 0, 1:]
    attention_pathway_mean = torch.mean(attention_pathway, 1)
    attention_pathway_mean = attention_pathway_mean.detach().to(device_cpu).numpy()
    data_atten_addr = save_addr + "/output_atten_test_model"+ str(model_size[model_id]) + ".csv"
    data_atten = pd.DataFrame(attention_pathway_mean)
    data_atten.columns = pathway_name
    data_atten.to_csv(
        data_atten_addr,
        index=True)

    #save model
    model_addr = save_addr + "/model" + str(model_size[model_id]) + ".pkl"
    torch.save(model,model_addr)



    del model
    del model_ae_mrna
    del linear_mrna
    del mrna_recon


    allocated, reserved = get_gpu_memory_usage()
    print("Allocated:", allocated, "bytes")
    print("Reserved:", reserved, "bytes")
    print("---------------------")
    print(" ")
    torch.cuda.empty_cache()
