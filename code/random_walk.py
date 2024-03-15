import networkx as nx
import random
import pandas as pd
import numpy as np
from IPython.display import clear_output

def node2vec_walk(G, start, sign=1, alpha=0.5,factor=0.9, length=10, p=1, q=1):
    """
    使用Node2Vec的随机游走策略在图G中进行随机游走。

    参数:
        G: 图
        sign: 起始节点的符号，促进/抑制
        alpha: 重启系数
        factor: 衰减系数
        start: 起始节点
        length: 游走的最大步数
        p: 返回参数
        q: 探索参数
        
    返回:
        一个节点列表，表示游走的路径
    """
    path = [start]
    prev_node = None
    sign_start=sign
    G.nodes[start]['value']+=sign
    # print(f'{start}+={sign}')
    sign=sign*factor
    while len(path) < length:
        current = path[-1]
        # 设置重启
        if (random.random()>alpha) or (prev_node is None):
            neighbors = list(G.neighbors(current))
            
            if not neighbors:
                break
        
            if prev_node is None:  # 当前节点是游走的起始节点
                probs = [1/len(neighbors) for _ in neighbors]
            else:
                probs = []
                for node in neighbors:
                    if node == prev_node:
                        probs.append(1/p)
                    elif G.has_edge(node, prev_node):
                        probs.append(1)
                    else:
                        probs.append(1/q)
                
                # 归一化概率
                sum_probs = sum(probs)
                probs = [prob/sum_probs for prob in probs]
        
            next_node = random.choices(neighbors, probs)[0]
            path.append(next_node)
            
            if G[current][next_node]['direction']=='Pos':
                G.nodes[next_node]['value']+=sign
                # print(f'{next_node}+={sign}')
            else:
                sign=-sign
                G.nodes[next_node]['value']+=sign
                # print(f'{next_node}+={sign}')
            sign=sign*factor
            prev_node = current
        else:
            # 重启
            prev_node = None
            path.append(start)
            sign=np.sign(sign_start)*abs(sign)
            G.nodes[start]['value']+=sign
            # print(f'{start}+={sign}')
            sign=sign*factor
    return path


def random_walk_specific_network(G,gene_preds,use_gene_list,use_gene_list_all,epoch,alpha,factor,length,p,q):
    results=[]
    for count in range(len(gene_preds)):
        clear_output(wait=True)
        print(f'random walk {count+1}/{len(gene_preds)}')
        for node in G.nodes:
            G.nodes[node]['value'] = 0
        input_df=pd.DataFrame({'gene':use_gene_list,'sign':gene_preds[count]})
        input_df=input_df[input_df['sign']!=0]
        input_df.reset_index(inplace=True)
        for x in range(len(input_df)):
            gene=input_df.loc[x,'gene']
            sign=input_df.loc[x,'sign']
            for y in range(epoch):
                node2vec_walk(G, start=gene, sign=sign, alpha=alpha, factor=factor,length=length,p=p, q=q)
        result_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
        result_df['Gene']=result_df.index.tolist()
        
        Genedf=pd.DataFrame({'Gene':use_gene_list_all})
        result_df=pd.merge(Genedf,result_df,on='Gene',how='left')
        result_df.loc[result_df['value'].isna(),'value']=0
        
        results.append(result_df)
    return results

def random_walk_new(gene_preds,use_gene_list,use_gene_list_all,epoch,alpha,factor,length,p,q):
    print('reading csv')
    ppi=pd.read_csv('HuamnSignalingNet_v7.csv')[['Unnamed: 1','Unnamed: 3','Unnamed: 4']]
    ppi.rename(columns={'Unnamed: 1': 'gene1', 'Unnamed: 3': 'gene2', 'Unnamed: 4': 'direction'}, inplace=True)
    print('creating graph')
    # 创建一个有向图
    G = nx.DiGraph()
    # 添加边和节点及其属性
    for index, row in ppi.iterrows():
        G.add_edge(row['gene1'], row['gene2'], direction=row['direction'])
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())
    results=[]
    for count in range(len(gene_preds)):
        clear_output(wait=True)
        print(f'random walk {count+1}/{len(gene_preds)}')
        for node in G.nodes:
            G.nodes[node]['value'] = 0
        input_df=pd.DataFrame({'gene':use_gene_list,'sign':gene_preds[count]})
        input_df=input_df[input_df['sign']!=0]
        input_df.reset_index(inplace=True)
        for x in range(len(input_df)):
            gene=input_df.loc[x,'gene']
            sign=input_df.loc[x,'sign']
            for y in range(epoch):
                node2vec_walk(G, start=gene, sign=sign, alpha=alpha, factor=factor,length=length,p=p, q=q)
        result_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
        result_df['Gene']=result_df.index.tolist()
        
        Genedf=pd.DataFrame({'Gene':use_gene_list_all})
        result_df=pd.merge(Genedf,result_df,on='Gene',how='left')
        result_df.loc[result_df['value'].isna(),'value']=0
        
        results.append(result_df)
    return results


def random_walk_application(G,gene_pred,use_gene_list,use_gene_list_all,epoch,alpha,factor,length,p,q):
    input_df=pd.DataFrame({'gene':use_gene_list,'sign':gene_pred})
    input_df=input_df[input_df['sign']!=0]
    input_df.reset_index(inplace=True)
    for x in range(len(input_df)):
        gene=input_df.loc[x,'gene']
        sign=input_df.loc[x,'sign']
        for y in range(epoch):
            node2vec_walk(G, start=gene, sign=sign, alpha=alpha, factor=factor,length=length,p=p, q=q)
    result_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    result_df['Gene']=result_df.index.tolist()
    Genedf=pd.DataFrame({'Gene':use_gene_list_all})
    result_df=pd.merge(Genedf,result_df,on='Gene',how='left')
    result_df.loc[result_df['value'].isna(),'value']=0
    return result_df