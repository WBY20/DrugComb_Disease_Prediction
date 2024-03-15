import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from IPython.display import clear_output
import torch.nn.functional as F

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, x_tensor):
        self.x = x_tensor

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)

# 定义神经网络模型
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size1, num_classes):
        super(Classifier, self).__init__()
        
        # Layers
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size1)
        self.layer4 = nn.Linear(hidden_size1, num_classes)
        
        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.bn2 = nn.BatchNorm1d(hidden_size1)
        
        # Activation Function and Dropout
        self.relu = nn.ELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer4(x)
        
        return x

def FCN_drug_gene(drug_gene_embeddings):
    print(f'predicting {len(drug_gene_embeddings)} drugs')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}')
    
    model = Classifier(600,1024,3).to(device)
    model.load_state_dict(torch.load('drug_gene_FCN_weights_1019_F.pth',map_location=device))
    
    preds=[]
    count=0
    for drug_gene_embedding in drug_gene_embeddings:
        count+=1
        clear_output(wait=True)
        print(f'predicting {count}/{len(drug_gene_embeddings)}')
        x_data=torch.tensor(drug_gene_embedding)
        dataset = CustomDataset(x_data)
        test_loader = DataLoader(dataset, batch_size=1000, shuffle=False)
        with torch.no_grad():
            model.eval()
            all_predictions = []
            for data in test_loader:
                data = data.to(device)
                outputs = model(data)
                outputs_p=F.softmax(outputs,dim=-1)
                predicted = torch.zeros(outputs_p.size(0), device=device,dtype=int)
                predicted[outputs_p[:, 0] > 0.88] = -1
                predicted[outputs_p[:, 2] > 0.45] = 1
                predicted=predicted.int()
                # _, predicted = torch.max(outputs, 1)
                # predicted-=1
                all_predictions.extend(predicted.cpu().tolist())
            preds.append(all_predictions)
    clear_output(wait=True)
    print(f'finish predict targets of {len(drug_gene_embeddings)} drugs')
    return preds



class Classifier_new(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
        super(Classifier_new, self).__init__()
        
        # Layers
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, hidden_size3)
        self.layer4 = nn.Linear(hidden_size3, num_classes)
        
        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        
        # Activation Function and Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer4(x)
        return x

# 预测结果更多
def FCN_drug_gene_new(drug_gene_embeddings):
    print(f'predicting {len(drug_gene_embeddings)} drugs')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}')
    
    model = Classifier_new(600,200,60,20,3).to(device)
    model.load_state_dict(torch.load('drug_gene_FCN_weights.pth',map_location=device))
    preds=[]
    count=0
    for drug_gene_embedding in drug_gene_embeddings:
        count+=1
        clear_output(wait=True)
        print(f'predicting {count}/{len(drug_gene_embeddings)}')
        x_data=torch.tensor(drug_gene_embedding)
        dataset = CustomDataset(x_data)
        test_loader = DataLoader(dataset, batch_size=1000, shuffle=False)
        with torch.no_grad():
            model.eval()
            all_predictions = []
            for data in test_loader:
                data = data.to(device)
                outputs = model(data)
                # outputs_p=F.softmax(outputs,dim=-1)
                # predicted = torch.zeros(outputs_p.size(0), device=device,dtype=int)
                # predicted[outputs_p[:, 0] > 0.8] = -1
                # predicted[outputs_p[:, 2] > 0.5] = 1
                # predicted=predicted.int()
                _, predicted = torch.max(outputs, 1)
                predicted-=1
                all_predictions.extend(predicted.cpu().tolist())
            preds.append(all_predictions)
    clear_output(wait=True)
    print(f'finish predict targets of {len(drug_gene_embeddings)} drugs')
    return preds