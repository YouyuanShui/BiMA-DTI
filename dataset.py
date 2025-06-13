import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import numpy as np 
from transformers import AutoTokenizer
from mol_graph import * 
from torch_geometric.data import Batch

class ProtDrugSeqDataset(Dataset):
    def __init__(self,  data_df,  prot_tokenizer, compund_tokenizer, training=False):
        super(ProtDrugSeqDataset, self).__init__()
        self.df = data_df
        self.prot_seq_id, self.prot_seq_attention_mask = self.tokenize_sequences(
            self.df["prot_seq"].to_list(), prot_tokenizer, 1000)
        self.comp_smiles_id, self.comp_smiles_attention_mask = self.tokenize_sequences(
            self.df["smiles"].to_list(), compund_tokenizer, 100)
        self.mol_graphs = {}
        self.generate_mol_graph()
    
    def generate_mol_graph(self):
        
        for smiles in self.df["smiles"].unique():
            self.mol_graphs[smiles] = from_smiles(smiles)
         

    def tokenize_sequences(self, sequences, tokenizer, max_length):
        tokenized_sentences = tokenizer(
            sequences,
            return_tensors="pt",
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        input_ids = tokenized_sentences["input_ids"]
        attention_mask = tokenized_sentences["attention_mask"]
        return input_ids, attention_mask

    

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = torch.FloatTensor( [ self.df.loc[idx]['-logKd/Ki']  ])
        pro_seq_id = self.prot_seq_id[idx]
        pro_seq_att = self.prot_seq_attention_mask[idx]
        smiles_id = self.comp_smiles_id[idx]
        smiles_att = self.comp_smiles_attention_mask[idx]    
        mol_graph = self.mol_graphs[ self.df.loc[idx]['smiles'] ]      
        return pro_seq_id, smiles_id, mol_graph, label

    def collate_fn(self, data_list):
        # print(data_list)
        pro_seq_id = torch.stack([data[0]  for data in data_list], axis=0)
        smiles_id = torch.stack([data[1]  for data in data_list], axis=0)
        label = torch.stack([data[3] for data in data_list], axis=0)
        
        mol_graph = Batch.from_data_list([data[2] for data in data_list])
        return pro_seq_id, smiles_id, mol_graph, label



class ProtDrugSeqDatasetCLS(Dataset):
    def __init__(self,  data_df,  training=False):
        super(ProtDrugSeqDatasetCLS, self).__init__()
        self.df = data_df
        self.seqlen = 1024
        self.smilen = 128
        self.charseqset = {
            "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
            "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
            "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
            "U": 19, "T": 20, "W": 21,
            "V": 22, "Y": 23, "X": 24,
            "Z": 25
        }
        self.charseqset_size = len(self.charseqset)

        self.inv_charseqset = {v: k for k, v in self.charseqset.items()}

        self.charsmiset = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                           "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                           "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                           "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                           "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                           "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                           "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                           "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64
                           }
        self.inv_charsmiset = {v: k for k, v in self.charsmiset.items()}

        self.charsmiset_size = len(self.charsmiset)
 
        self.mol_graphs = {}
        self.onehot_smiles = {}
        self.onehot_protseq = {}
        self.generate_data()
    
    def generate_data(self):
        for smiles in self.df["SMILES"].unique():
            self.mol_graphs[smiles] = from_smiles(smiles)
            self.onehot_smiles[smiles] = label_smiles(smiles, self.smilen, self.charsmiset)
        
        for protseq in self.df["Protein"].unique():
            self.onehot_protseq[protseq] = label_sequence(protseq, self.seqlen, self.charseqset)

  

    

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = torch.FloatTensor( [ self.df.loc[idx]['Y']  ])
        pro_seq_id = torch.from_numpy(self.onehot_protseq[self.df.loc[idx]['Protein']]).long()
        smiles_id = torch.from_numpy(self.onehot_smiles[self.df.loc[idx]['SMILES']]).long()
        mol_graph = self.mol_graphs[ self.df.loc[idx]['SMILES'] ]      
        return pro_seq_id, smiles_id, mol_graph, label

    def collate_fn(self, data_list):
        # print(data_list)
        pro_seq_id = torch.stack([data[0]  for data in data_list], axis=0)
        smiles_id = torch.stack([data[1]  for data in data_list], axis=0)
        label = torch.stack([data[3] for data in data_list], axis=0)
        
        mol_graph = Batch.from_data_list([data[2] for data in data_list])
        return pro_seq_id, smiles_id, mol_graph, label



def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):  # x, smi_ch_ind, y
        X[i] = smi_ch_ind[ch]

    return X  # tolist()

def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros(MAX_SEQ_LEN)

    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]

    return X  # tolist()
