import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Reduce
from mambapy.mamba import MambaBlock, MambaConfig
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import TransformerConv, global_max_pool
from torch_geometric.nn.norm import GraphNorm
class GPSConv(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        edge_dim: int,
        heads: int = 1,
        dropout: float = 0.2,
        attn_dropout: float = 0.2,
        act=torch.relu
    ):
        super().__init__()
        self.channels = channels
        self.act = act
        self.heads = heads
        self.dropout = dropout 
        self.conv = TransformerConv(
            channels,
            channels//heads,
            heads=heads,
            edge_dim=edge_dim,
            beta=True,
            dropout=0.1 
        )
        self.linear = nn.Linear(
            channels*heads, 
            channels
        ) 
        self.attn = BidirMambaBlock(channels) 
        self.mlp = SwiGLU(channels, channels * 4)
        self.bn = GraphNorm(channels)
    
    def forward(
        self,
        x,
        edge_index ,
        edge_attr ,
        batch 
    )  :
        r"""Runs the forward pass of the module."""
        hs = []       
        h = self.conv(x, edge_index, edge_attr)
        h = self.linear(h)
        
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x
        h = self.bn(h)
        hs.append(h)

        # Global attention transformer-style model.
        h, mask = to_dense_batch(x, batch)
        h = self.attn(h)
        h = h[mask]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Residual connection.
        h = self.bn(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        out = self.bn(out)
        return out

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0., 
                norm_layer=nn.LayerNorm, subln=False
            ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x

class BidirMambaBlock(nn.Module):
    def __init__(self, n_embed, weight_tie = True ) -> None:
        super().__init__()
        config = MambaConfig(d_model=n_embed, n_layers=1)
        self.mixer = MambaBlock( config ) 
        self.mixer_back = MambaBlock( config ) 
        self.ln = nn.LayerNorm(n_embed)

        if weight_tie:  # Tie in and out projections (where most of param count lies) (from Caduceus)
            self.mixer_back.in_proj.weight = self.mixer.in_proj.weight
            self.mixer_back.in_proj.bias = self.mixer.in_proj.bias
            self.mixer_back.out_proj.weight = self.mixer.out_proj.weight
            self.mixer_back.out_proj.bias = self.mixer.out_proj.bias
        self.mlp = SwiGLU(n_embed, n_embed*4)
        self.norm3 = nn.LayerNorm(n_embed)

    def forward(self, x): # x must be batch x time x channels

        mix_flip = self.mixer_back(x.flip(1))
        x = self.ln(x + self.mixer(x) +  mix_flip.flip(1))

        return x

class AM_Layer(nn.Module):
    def __init__(self, d_model):
        super(AM_Layer, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model, d_model//32, batch_first=True)
        self.mamba =  BidirMambaBlock(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        
    def forward(self, x ):
        x = x +   self.self_attention( x, x, x )[0]
        x = self.norm1(x)
        x = self.mamba(x)
        return x


class MultimodalGatingNetwork(nn.Module):
    def __init__(self, dim):
        super(MultimodalGatingNetwork, self).__init__()
        self.gated_g = SwiGLU(dim, dim*4)
        self.gated_s = SwiGLU(dim, dim*4)
        self.gated_p = SwiGLU(dim, dim*4)
         
        self.ff_net1 = Feature_Fusion(dim)
        self.ff_net2 = Feature_Fusion(dim)
         
    def forward(self, mg, ms, mp):
        mg = self.gated_g(mg)
        ms = self.gated_s(ms)
        mp = self.gated_p(mp)
        f1 = self.ff_net1( mg, mp )
        f2 = self.ff_net2( ms, mp )
        return  torch.cat( (f1, f2), 1 ) 

class MambaCPAModelWoPretrained(nn.Module):
    def __init__(self):
        super().__init__()
        
        hidden_size = 256
        self.comp_embedding = nn.Embedding( 64,  hidden_size)
        self.comp_position_embeddings = nn.Embedding( 128, hidden_size)
        self.prot_embedding = nn.Embedding( 25,  hidden_size)
        self.prot_position_embeddings = nn.Embedding( 1024, hidden_size)
        
        self.x2h =  nn.Linear(26, hidden_size)  #
        self.mol_graph_net1 = GPSConv(hidden_size, 14)
        self.mol_graph_net2 = GPSConv(hidden_size, 14)
        self.mol_graph_net3 = GPSConv(hidden_size, 14)

        config = MambaConfig(d_model=hidden_size, n_layers=1)
        self.pre_mamba_comp = MambaBlock(config)  
        self.pre_mamba_prot = MambaBlock(config)    
        self.prot_mamba = nn.Sequential(
            AM_Layer( hidden_size ),
            # AM_Layer( hidden_size ),
        )
        self.comp_mamba = nn.Sequential(
            AM_Layer( hidden_size ),
            # AM_Layer( hidden_size ),
        )
        
        dropout = 0.2
        self.pred_net =  nn.Sequential(
            nn.BatchNorm1d(   4*   hidden_size),
            nn.Linear(  4* hidden_size, 1024), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),   
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512 ),   
            nn.GELU(),
            nn.Linear(512, 1 ),  
        ) 
        self.multi_modal_fusion = MultimodalGatingNetwork(hidden_size)

    def forward(self, pro_seq_id, smiles_id, mol_graph ):
        mol_graph.x = self.x2h( mol_graph.x )
        mol_graph.x = self.mol_graph_net1( mol_graph.x, mol_graph.edge_index, mol_graph.edge_attr, mol_graph.batch )
        mol_graph_x = global_max_pool(mol_graph.x, mol_graph.batch)

        prot_seq_emb =  self.prot_embedding(pro_seq_id) 
        comp_seq_emb =  self.comp_embedding(smiles_id) 

        prot_seq_emb = self.pre_mamba_prot(prot_seq_emb)
        comp_seq_emb = self.pre_mamba_comp(comp_seq_emb)

        prot_seq_emb = self.prot_mamba(prot_seq_emb)
        comp_seq_emb = self.comp_mamba(comp_seq_emb)

        comp_seq_emb =  Reduce('b n d -> b d', 'max')(comp_seq_emb)
        prot_seq_emb = Reduce('b n d -> b d', 'max')(prot_seq_emb)
        cp_embedding = self.multi_modal_fusion( mol_graph_x,  comp_seq_emb, prot_seq_emb)
        pred = self.pred_net( cp_embedding )

        return pred

class Feature_Fusion(nn.Module):
    def __init__(self, channels, r=3):
        super(Feature_Fusion, self).__init__()
        inter_channels = int(channels * r)
        layers = [nn.Linear(channels*2, inter_channels), nn.GELU(),  nn.Linear(inter_channels, channels)]
        self.att1 = nn.Sequential(*layers)
        self.att2 = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fd, fp):
        concat = torch.cat((fd, fp), 1)
        w1 = self.sigmoid(self.att1( concat ))
        fout1 = torch.cat( (fd * w1 , fp * (1 - w1)), 1 ) 
        w2 = self.sigmoid(self.att2(fout1))
        fout2 = torch.cat( (fd * w2 , fp * (1 - w2)), 1 ) 
        return fout2
