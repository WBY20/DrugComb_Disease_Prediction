#!/usr/bin/env python
# coding: utf-8

# In[22]:


import importlib
import pandas as pd
import pubchempy
import json
import FCN_drug_gene
importlib.reload(FCN_drug_gene)
# 这个函数已不使用
import FCN_disease_gene
importlib.reload(FCN_disease_gene)
import random_walk
importlib.reload(random_walk)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from IPython.display import clear_output, display, update_display, HTML
import gseapy
importlib.reload(gseapy)
import random
import warnings
import pickle
import time
import scipy.stats as stats
import networkx as nx
import copy
import matplotlib.pyplot as plt


# ### 读取并处理数据

# In[1]:


# 后续数据有更新，将部分文件替换为带_20240318后缀的


# In[6]:


d_g=pd.read_csv('CTD_chemicals_diseases.csv',skiprows=range(0, 29),delimiter=',',header=None)
d_g.columns=['ChemicalName','ChemicalID','CasRN','DiseaseName','DiseaseID','DirectEvidence','InferenceGeneSymbol','InferenceScore','OmimIDs','PubMedIDs']
d_g=d_g[['ChemicalName','DiseaseID','DiseaseName','DirectEvidence']]
d_g=d_g[[True if x.split(':')[0]=='MESH' else False for x in d_g['DiseaseID']]]
d_g.loc[:,'DiseaseID']=[x.split(':')[1] for x in d_g['DiseaseID']]
d_g.dropna(subset=['DirectEvidence'],inplace=True)


# In[17]:


mesh=pd.read_csv('mesh_info.csv')
mesh.rename(columns={'mesh':'DiseaseName'},inplace=True)
mesh['mesh_embedding']=mesh['mesh_embedding'].apply(json.loads)


# In[18]:


# 限定d_g的疾病为mesh和ctd中都存在的
d_g=pd.merge(d_g,mesh,on='DiseaseName',how='inner')


# In[20]:


drugs_superstore=pd.read_csv('drugs_superstore.csv')[['ChemicalName','cid','smiles','drug_embedding']]
drugs_superstore['drug_embedding']=drugs_superstore['drug_embedding'].apply(json.loads)


# In[24]:


# 限定d_g的药物为药物库里有的
d_g=pd.merge(d_g,drugs_superstore,on='ChemicalName',how='inner')
d_g=d_g.drop_duplicates(subset=['ChemicalName','DiseaseName']).reset_index(drop=True)


# In[25]:


# 获取现在情况下涉及的药物和疾病列表
drugs_df_all=d_g[['ChemicalName','drug_embedding']].drop_duplicates(subset='ChemicalName').reset_index(drop=True)
diseases_df_all=d_g[['DiseaseName','mesh_embedding']].drop_duplicates(subset='DiseaseName').reset_index(drop=True)
print(f'drugs:{len(drugs_df_all)}    diseases:{len(diseases_df_all)}    d_g pairs:{len(d_g)}')


# In[9]:


# 读取疾病名称和TCGA疾病名称的转化
diseases_TCGA=pd.read_csv('Xena/diseases.csv').dropna(subset=['GTEx_category']).reset_index(drop=True)


# In[11]:


# 把疾病限制为TCGA中有的疾病
d_g_TCGA=d_g[[x in diseases_TCGA['DiseaseName'].tolist() for x in d_g['DiseaseName']]].reset_index(drop=True)
d_g_TCGA=pd.merge(d_g_TCGA,diseases_TCGA,on='DiseaseName',how='inner')


# In[12]:


# 获取现在情况下涉及的药物和疾病列表
drugs_df_TCGA=d_g_TCGA[['ChemicalName','drug_embedding']].drop_duplicates(subset='ChemicalName').reset_index(drop=True)
diseases_df_TCGA=d_g_TCGA[['DiseaseName','mesh_embedding']].drop_duplicates(subset='DiseaseName').reset_index(drop=True)
print(f'TCGA drugs:{len(drugs_df_TCGA)}    TCGA diseases:{len(diseases_df_TCGA)}    d_g_TCGA pairs:{len(d_g_TCGA)}')


# In[2]:


# 读取疾病特异性网络合集
network_symbol_all=pd.read_csv('Xena/network_symbol_all.csv')


# In[14]:


# 读取基因embedding信息
gene_info=pd.read_csv('gene_info_process.csv')
gene_info.rename(columns={'embedding':'gene_embedding'},inplace=True)
gene_info['gene_embedding']=gene_info['gene_embedding'].apply(json.loads)
gene_info=gene_info.drop_duplicates(subset=['Gene']).reset_index(drop=True)[['gene_embedding','entrez_gene_id','Gene']]


# ### 计算所有药物在所有基因上的预测结果，保存以加快后续处理速度

# In[62]:


use_gene_list_all=list(gene_info['Gene'])
gene_embedding_list = list(gene_info['gene_embedding'])
# 拆分计算节省内存
count=0
for zzz in range(0,len(drugs_df_all),200):
    drug_gene_embeddings=[]
    this_drug=[]
    for x in range(zzz,min(zzz+200,len(drugs_df_all))):
        count+=1
        clear_output(wait=True)
        print(f"Count: {count}/{len(drugs_df_all)}")
        drug_embedding=drugs_df_all.loc[x,'drug_embedding']
        drug_gene_embedding=[]
        for y in gene_embedding_list:
            drug_gene_embedding.append(drug_embedding+y)
        drug_gene_embeddings.append(drug_gene_embedding)
        this_drug.append(drugs_df_all.loc[x,'ChemicalName'])
    drug_gene_preds=FCN_drug_gene.FCN_drug_gene(drug_gene_embeddings)
    
    with open('save_drug_pred_all_gene_results/drug_gene_preds_'+str(zzz)+'.pkl', 'wb') as f:
        pickle.dump(drug_gene_preds, f)
    with open('save_drug_pred_all_gene_results/drugs_'+str(zzz)+'.pkl', 'wb') as f:
        pickle.dump(this_drug, f)


# In[63]:


with open('save_drug_pred_all_gene_results/use_gene_list_all.pkl', 'wb') as f:
    pickle.dump(use_gene_list_all, f)


# In[64]:


drug_gene_preds_all=[]
drug_list_all=[]
for zzz in range(0,len(drugs_df_all),200):
    with open('save_drug_pred_all_gene_results/drug_gene_preds_'+str(zzz)+'.pkl', 'rb') as f:
        drug_gene_preds_all.extend(pickle.load(f))
    with open('save_drug_pred_all_gene_results/drugs_'+str(zzz)+'.pkl', 'rb') as f:
        drug_list_all.extend(pickle.load(f))
with open('save_drug_pred_all_gene_results/use_gene_list_all.pkl', 'rb') as f:
    use_gene_list_all=pickle.load(f)


# In[65]:


with open('drug_gene_preds_all.pkl', 'wb') as f:
    pickle.dump(np.array(drug_gene_preds_all), f)
with open('drug_list_all.pkl', 'wb') as f:
    pickle.dump(drug_list_all, f)
with open('use_gene_list_all.pkl', 'wb') as f:
    pickle.dump(use_gene_list_all, f)
pd.DataFrame({'Gene':use_gene_list_all}).to_csv('use_gene_list_all.csv',index=False)


# In[23]:


# 加载已保存的所有药物在所有基因上的预测结果
with open('drug_gene_preds_all.pkl', 'rb') as f:
    # numpy array类型
    drug_gene_preds_all=pickle.load(f)
with open('drug_list_all.pkl', 'rb') as f:
    drug_list_all=pickle.load(f)
with open('use_gene_list_all.pkl', 'rb') as f:
    use_gene_list_all=pickle.load(f)
print(f'load saved all drug-gene-preds data, {len(drug_list_all)} drugs on {len(use_gene_list_all)} genes')


# ### 一次游走完所有的

# In[ ]:


# 多进程进行药物游走
def process_sublist_drug(sublist):
    # 调用 random_walk 函数，并传递子列表 sublist
    walk_results_drug = random_walk.random_walk_specific_network(copy.deepcopy(G),sublist, use_gene_list,use_gene_list_all, epoch=15, alpha=0.3, factor=0.95, length=15, p=1, q=1)
    # 下面为使用非特异性网络，平时注释掉
    # walk_results_drug = random_walk.random_walk_new(sublist, use_gene_list,use_gene_list_all, epoch=15, alpha=0.3, factor=0.95, length=15, p=1, q=1)
    return walk_results_drug

def split_list(lst, n):
    # 将列表 lst 拆分成 n 个较小的列表
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
for ttttt in range(len(diseases_TCGA)):
    # 选取特定TCGA疾病
    TCGA_disease=diseases_TCGA.loc[ttttt,'TCGA']
    TCGA_drugs=d_g_TCGA.loc[d_g_TCGA['TCGA']==TCGA_disease,'ChemicalName']
    print(f'disease:{TCGA_disease}    target_drugs:{len(TCGA_drugs)}')
    network=network_symbol_all.loc[network_symbol_all['category']==TCGA_disease,:]
    network.loc[network['cor']>0,'cor']=1
    network.loc[network['cor']<0,'cor']=-1
    # 删去边为0的边
    network=network.loc[network['cor']!=0,:].reset_index(drop=True)
    # 创建TCGA特异性有向图
    G = nx.DiGraph()
    # 添加边和节点及其属性
    for index, row in network.iterrows():
        G.add_edge(row['node1'], row['node2'], direction=row['cor'])
    print(f'create DiGraph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges')
    # 获取网络中基因与基因数据库的基因的交集
    use_gene_list=list((set(network['node1']).union(set(network['node2']))).intersection(set(gene_info['Gene'])))
    # # 这是原先使用的非特异性网络，平时都注释掉
    # ppi=pd.read_csv('HuamnSignalingNet_v7.csv')[['Unnamed: 1','Unnamed: 3','Unnamed: 4']]
    # ppi.rename(columns={'Unnamed: 1': 'gene1', 'Unnamed: 3': 'gene2', 'Unnamed: 4': 'direction'}, inplace=True)
    # ppi=ppi[ppi['direction']!='Phy']
    # ppi=ppi.reset_index(drop=True)
    # use_gene_list=list(set(list(ppi['gene1'])+list(ppi['gene2'])).intersection(set(gene_info['Gene'])))
    gene_df=gene_info.loc[[x in use_gene_list for x in list(gene_info['Gene'])],:].drop_duplicates(subset=['Gene']).reset_index(drop=True)
    use_gene_list=list(gene_df['Gene'])
    print(f'{len(use_gene_list)} genes exist in gene_info')
    disease_embedding=d_g_TCGA['mesh_embedding'][d_g_TCGA['TCGA'].tolist().index(TCGA_disease)]
    disease_gene_embedding=[]
    for x in list(gene_df['gene_embedding']):
        disease_gene_embedding.append(disease_embedding+x)
    disease_gene_preds=FCN_disease_gene.FCN_disease_gene([disease_gene_embedding],target_number=100,use_gene_list=use_gene_list)
    # 依据use_gene_list在use_gene_list_all中的顺序，从drug_gene_preds_all中提取对应的基因，药物顺序为drug_list_all
    idx=[use_gene_list_all.index(x) for x in use_gene_list]
    drug_gene_preds=list(drug_gene_preds_all[:,idx])
    print(f'predict on {len(drug_gene_preds[0])} genes for each of {len(drug_gene_preds)} drugs')
    # 拆分 drug_gene_preds 列表
    process_number=48
    sublists = split_list(drug_gene_preds, process_number)
    
    # 创建一个进程池
    with multiprocessing.Pool() as pool:
        # 使用 map 函数将 process_sublist 函数应用到每个子列表
        results = pool.map(process_sublist_drug, sublists)
    
    # 合并所有进程的结果
    walk_results_drug = [item for sublist in results for item in sublist]
    
    print(f'cpu processes number:{process_number}')
    print(f'finish random walk on {G.number_of_nodes()} genes for each of {len(drug_gene_preds)} drugs\n\
    each drug return a dataframe with {len(walk_results_drug[0])} rows corresponding to {len(use_gene_list_all)} genes in use_gene_list_all')
    
    # 保存列表到文件
    with open('walk_results_drug_TCGA/walk_results_drug_'+TCGA_disease+'.pkl', 'wb') as f:
        pickle.dump(walk_results_drug, f)
    # 从文件加载列表
    # with open('walk_results_drug_TCGA/walk_results_drug_'+TCGA_disease+'.pkl', 'rb') as f:
    #     walk_results_drug = pickle.load(f)


# ### 构建疾病特异性网络

# In[173]:


# 选取特定TCGA疾病
TCGA_disease=diseases_TCGA.loc[11,'TCGA']
TCGA_drugs=d_g_TCGA.loc[d_g_TCGA['TCGA']==TCGA_disease,'ChemicalName']
print(f'disease:{TCGA_disease}    target_drugs:{len(TCGA_drugs)}')


# In[161]:


network=network_symbol_all.loc[network_symbol_all['category']==TCGA_disease,:]
network.loc[network['cor']>0,'cor']=1
network.loc[network['cor']<0,'cor']=-1
# 删去边为0的边
network=network.loc[network['cor']!=0,:].reset_index(drop=True)


# In[162]:


# 创建TCGA特异性有向图
G = nx.DiGraph()
# 添加边和节点及其属性
for index, row in network.iterrows():
    G.add_edge(row['node1'], row['node2'], direction=row['cor'])
print(f'create DiGraph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges')


# In[163]:


# 获取网络中基因与基因数据库的基因的交集
use_gene_list=list((set(network['node1']).union(set(network['node2']))).intersection(set(gene_info['Gene'])))


# In[164]:


# # 这是原先使用的非特异性网络，平时都注释掉
# ppi=pd.read_csv('HuamnSignalingNet_v7.csv')[['Unnamed: 1','Unnamed: 3','Unnamed: 4']]
# ppi.rename(columns={'Unnamed: 1': 'gene1', 'Unnamed: 3': 'gene2', 'Unnamed: 4': 'direction'}, inplace=True)
# ppi=ppi[ppi['direction']!='Phy']
# ppi=ppi.reset_index(drop=True)
# use_gene_list=list(set(list(ppi['gene1'])+list(ppi['gene2'])).intersection(set(gene_info['Gene'])))


# In[165]:


gene_df=gene_info.loc[[x in use_gene_list for x in list(gene_info['Gene'])],:].drop_duplicates(subset=['Gene']).reset_index(drop=True)
use_gene_list=list(gene_df['Gene'])
print(f'{len(use_gene_list)} genes exist in gene_info')


# ### 计算药物谱

# In[168]:


# 依据use_gene_list在use_gene_list_all中的顺序，从drug_gene_preds_all中提取对应的基因，药物顺序为drug_list_all
idx=[use_gene_list_all.index(x) for x in use_gene_list]
drug_gene_preds=list(drug_gene_preds_all[:,idx])
print(f'predict on {len(drug_gene_preds[0])} genes for each of {len(drug_gene_preds)} drugs')


# In[169]:


# 多进程进行药物游走
def process_sublist_drug(sublist):
    # 调用 random_walk 函数，并传递子列表 sublist
    walk_results_drug = random_walk.random_walk_specific_network(copy.deepcopy(G),sublist, use_gene_list,use_gene_list_all, epoch=15, alpha=0.3, factor=0.95, length=15, p=1, q=1)
    # 下面为使用非特异性网络，平时注释掉
    # walk_results_drug = random_walk.random_walk_new(sublist, use_gene_list,use_gene_list_all, epoch=15, alpha=0.3, factor=0.95, length=15, p=1, q=1)
    return walk_results_drug

def split_list(lst, n):
    # 将列表 lst 拆分成 n 个较小的列表
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


# In[170]:


# 拆分 drug_gene_preds 列表
process_number=48
sublists = split_list(drug_gene_preds, process_number)

# 创建一个进程池
with multiprocessing.Pool() as pool:
    # 使用 map 函数将 process_sublist 函数应用到每个子列表
    results = pool.map(process_sublist_drug, sublists)

# 合并所有进程的结果
walk_results_drug = [item for sublist in results for item in sublist]

print(f'cpu processes number:{process_number}')
print(f'finish random walk on {G.number_of_nodes()} genes for each of {len(drug_gene_preds)} drugs\n\
each drug return a dataframe with {len(walk_results_drug[0])} rows corresponding to {len(use_gene_list_all)} genes in use_gene_list_all')


# In[171]:


# 保存列表到文件
with open('walk_results_drug_TCGA/walk_results_drug_'+TCGA_disease+'.pkl', 'wb') as f:
    pickle.dump(walk_results_drug, f)
# 从文件加载列表
# with open('walk_results_drug_TCGA/walk_results_drug_'+TCGA_disease+'.pkl', 'rb') as f:
#     walk_results_drug = pickle.load(f)

