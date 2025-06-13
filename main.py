import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from dataset import *
import numpy as np
from early_stop import  EarlyStopping
from tqdm import tqdm 
from model import * 
from utils import * 
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, accuracy_score, precision_recall_curve, precision_score, recall_score, matthews_corrcoef
from prettytable import PrettyTable

seed = 42
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

def ptable_to_csv(table, filename, headers=True):
    """Save PrettyTable results to a CSV file.

    Adapted from @AdamSmith https://stackoverflow.com/questions/32128226

    :param PrettyTable table: Table object to get data from.
    :param str filename: Filepath for the output CSV.
    :param bool headers: Whether to include the header row in the CSV.
    :return: None
    """
    raw = table.get_string()
    data = [tuple(filter(None, map(str.strip, splitline)))
            for line in raw.splitlines()
            for splitline in [line.split('|')] if len(splitline) > 1]
    if table.title is not None:
        data = data[1:]
    if not headers:
        data = data[1:]
    with open(filename, 'w') as f:
        for d in data:
            f.write('{}\n'.format(','.join(d)))

def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    for data in dataloader:
        pro_seq_id, smiles_id, mol_graph, label = data
        pro_seq_id, smiles_id, mol_graph, label = pro_seq_id.to(device),   smiles_id.to(device), mol_graph.to(device), label.to(device)
                 
        with torch.no_grad():
            pred   = model(pro_seq_id, smiles_id, mol_graph )
            pred = torch.sigmoid(pred)
            pred_list.append(pred.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            
    pred = np.concatenate(pred_list, axis=0) 
    y_label = np.concatenate(label_list, axis=0)
    y_pred = pred.reshape( pred.shape[0], 1 )
    
    auroc = roc_auc_score(y_label, y_pred)
    auprc = average_precision_score(y_label, y_pred)
    fpr, tpr, thresholds = roc_curve(y_label, y_pred)
    prec, recall, _ = precision_recall_curve(y_label, y_pred)
    try:
        precision = tpr / (tpr + fpr)
    except RuntimeError:
        pass

    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
    thred_optim = thresholds[5:][np.argmax(f1[5:])]
    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
    accuracy = accuracy_score(y_label, y_pred_s)
    recall = recall_score(y_label, y_pred_s)
    precision = precision_score(y_label, y_pred_s)
    mcc = matthews_corrcoef(y_label, y_pred_s)
    model.train()

    return auroc, auprc, np.max(f1[5:]), precision, recall, accuracy, mcc, thred_optim

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

if __name__ == '__main__':
    
    split_mode = 'random'
    repeats = 10
    t_tables = PrettyTable(['repreat', 'AUROC', 'AUPRC', 'F1', 'ACC', 'MCC'   ])
    t_tables.float_format = '.4'   
    dataset_name = 'BioSNAP'
    for repeat in range(repeats):

        data_root = f'/data/DTA/MambaDTA/{dataset_name}/'
    
        dataFolder = data_root + split_mode + '/' + str(repeat) + '/'
        """load data"""
        train_df = pd.read_csv(dataFolder + "train.csv")
        valid_df = pd.read_csv(dataFolder + "validation.csv")
        test_df = pd.read_csv(dataFolder + "test.csv")

        train_set = ProtDrugSeqDatasetCLS(train_df  )
        valid_set = ProtDrugSeqDatasetCLS(valid_df  )
        test_set = ProtDrugSeqDatasetCLS(test_df  )
         
        batch_size = 16
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=8, drop_last=True, collate_fn=train_set.collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=valid_set.collate_fn)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,   num_workers=8, collate_fn=test_set.collate_fn)

        print(f"train data: {len(train_set)}")
        print(f"valid data: {len(valid_set)}")
        print(f"test  data: {len(test_set)}")

        device = torch.device('cuda:0')
        model = MambaCPAModelWoPretrained().to(device)
        model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr=5e-5 )
        criterion = nn.BCEWithLogitsLoss()
        running_loss = AverageMeter()
        running_acc = AverageMeter()
        running_best_mse = BestMeter("min")
        best_model_list = []
        save_dir = 'model'
        early_stop_epoch = 10
        epochs = 100
        stopper = EarlyStopping(patience=early_stop_epoch,
                                filename=save_dir + '/model.pth', mode='higher')
        for epoch in range(epochs):
            model.train()
            pbar = tqdm(train_loader)
            for data in pbar:
                pbar.set_description(f"Epoch {epoch}")
                pro_seq_id, smiles_id, mol_graph, label = data
                pro_seq_id, smiles_id, mol_graph, label = pro_seq_id.to(device),   smiles_id.to(device), mol_graph.to(device), label.to(device)
                 
                pred = model(pro_seq_id, smiles_id, mol_graph)
                mainloss = criterion(pred.view(-1), label.view(-1)) 
                loss = mainloss    
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()
                
                pbar.set_postfix(loss=mainloss.item()  )
                running_loss.update(loss.item(), label.size(0)) 

            epoch_loss = running_loss.get_average()
            epoch_rmse = np.sqrt(epoch_loss)
            running_loss.reset()

            # start validating
            auroc, auprc, f1, precision, recall, accuracy, mcc, thred_optim  = val(model, valid_loader, device)
            early_stop = stopper.step(auroc, model)

            e_tables = PrettyTable(['epoch', 'AUROC', 'AUPRC', 'F1', 'ACC', 'MCC'   ])
            row = [epoch, auroc, auprc, f1,  accuracy, mcc]
        

            e_tables.float_format = '.3' 
            
            e_tables.add_row(row)
            print(e_tables)
           
            if early_stop:
                break

        # final testing
        stopper.load_checkpoint(model)
        auroc, auprc, f1, precision, recall, accuracy, mcc, thred_optim  = val(model, test_loader, device)       
        row = [repeat, auroc, auprc, f1,  accuracy, mcc]
        t_tables.float_format = '.3' 
        t_tables.add_row(row)
        print(t_tables)
    
    results_filename =   split_mode + '-' + dataset_name+ '.csv'
    ptable_to_csv(t_tables, results_filename)
    df = pd.read_csv(results_filename)
    print(df.mean())
            
       
        