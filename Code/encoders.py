import torch
from torch import nn
import torch.nn.functional as F
import dgl

from layers import NodeConv, EdgeConv
from layers import MLP
from layers import NodeMPNN, EdgeMPNN, NodeMPNNV2
from layers import GENConv



class UVNetGraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        input_edge_dim,
        output_dim,
        hidden_dim=64,
        learn_eps=True,
        num_layers=3,
        num_mlp_layers=2,
    ):
        """
        This is the graph neural network used for message-passing features in the
        face-adjacency graph.  (see Section 3.2, Message passing in paper)

        Args:
            input_dim ([type]): [description]
            input_edge_dim ([type]): [description]
            output_dim ([type]): [description]
            hidden_dim (int, optional): [description]. Defaults to 64.
            learn_eps (bool, optional): [description]. Defaults to True.
            num_layers (int, optional): [description]. Defaults to 3.
            num_mlp_layers (int, optional): [description]. Defaults to 2.
        """
        super(UVNetGraphEncoder, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of layers for node and edge feature message passing
        self.node_conv_layers = torch.nn.ModuleList()
        self.edge_conv_layers = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            node_feats = input_dim if layer == 0 else hidden_dim
            edge_feats = input_edge_dim if layer == 0 else hidden_dim
            self.node_conv_layers.append(
                NodeConv(
                    node_feats=node_feats,
                    out_feats=hidden_dim,
                    edge_feats=edge_feats,
                    num_mlp_layers=num_mlp_layers,
                    hidden_mlp_dim=hidden_dim,
                ),
            )
            self.edge_conv_layers.append(
                EdgeConv(
                    edge_feats=edge_feats,
                    out_feats=hidden_dim,
                    node_feats=node_feats,
                    num_mlp_layers=num_mlp_layers,
                    hidden_mlp_dim=hidden_dim,
                )
            )

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

        self.drop1 = nn.Dropout(0.3)
        self.drop = nn.Dropout(0.5)
        self.pool = dgl.nn.MaxPooling()

    def forward(self, g, h, efeat):
        hidden_rep = [h]
        he = efeat

        for i in range(self.num_layers - 1):
            # Update node features
            h = self.node_conv_layers[i](g, h, he)
            # Update edge features
            he = self.edge_conv_layers[i](g, h, he)
            hidden_rep.append(h)

        out = hidden_rep[-1]
        out = self.drop1(out)
        score_over_layer = 0

        # Perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        return out, score_over_layer

class TGNet(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        num_layers,
        delta,
        mlp_ratio=4,
        drop=0.,
        drop_path=0.,
        conv_on_edge=True,
        transformer_heads=4,
        transformer_layers=1,
    ):
        super().__init__()
        self.num_layers=num_layers
        self.conv_on_edge = conv_on_edge
        self.node_convs = nn.ModuleList()
        self.edge_convs = nn.ModuleList()
        for _ in range(2):
            if self.conv_on_edge:
                from .layers import EdgeMPNN
                self.edge_convs.append(
                    EdgeMPNN(node_dim, edge_dim, mlp_ratio, drop, drop_path))
            from .layers import NodeMPNN
            self.node_convs.append(
                NodeMPNN(node_dim, edge_dim, delta, mlp_ratio, drop, drop_path))
        self.post_norm = nn.LayerNorm(node_dim)

        # global transformer encoder layer
        encoder_layer=nn.TransformerEncoderLayer(
            d_model=node_dim,
            nhaed=transformer_heads,
            batch_first=True,
            dim_feedforward=node_dim*4,
            dropout=drop
        )
        self.transformer_encoder=nn.TransformerEncoder(
            encoder_layer,num_layers=transformer_layers
        )

        self.pool=dgl.nn.Avgpooling()
        self.linear=MLP(1,node_dim,0,node_dim,nn.LayerNorm,True)

    def forward(self,g,h,he):
        #GNN part
        if self.conv_on_edge:
            he =self.edge_convs[0](g,h,he)
        h=self.node_convs[0](g,h,he)
        for i in range(self.num_layers-1):
            if self.conv_on_edge:
                he=self.edge_convs[1](g,h,he)
            h=self.node_convs[1](g,h,he)
        h=self.post_norm(h)

        #Trans part
        node_batch_ids=g.batch_num_nodes()
        start=0
        node_feats_batch=[]
        max_len=node_batch_ids.max().item()
        for n in node_batch_ids:
            node_feats = h[start:start+n, :]
            # [n, c] -> [max_len, c], pad为0
            if n < max_len:
                pad = torch.zeros(max_len - n, h.size(1), device=h.device, dtype=h.dtype)
                node_feats = torch.cat([node_feats, pad], dim=0)
            node_feats_batch.append(node_feats.unsqueeze(0))
            start += n
        node_feats_batch = torch.cat(node_feats_batch, dim=0)  # [B, max_len, C]
        # mask padding
        mask = torch.arange(max_len, device=h.device)[None, :].expand(len(node_batch_ids), -1)
        mask = mask >= node_batch_ids[:, None]
        h_global = self.transformer_encoder(node_feats_batch, src_key_padding_mask=mask)
        
        h_list = []
        for i, n in enumerate(node_batch_ids):
            h_list.append(h_global[i, :n, :])
        h = torch.cat(h_list, dim=0)  # [N, C]

        # graph pooling
        local_feat = h
        global_feat = self.linear(self.pool(g, local_feat))
        return local_feat, global_feat





class GCN(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        num_layers,
        delta,
        mlp_ratio=4,
        drop=0.,
        drop_path=0.,
        conv_on_edge=True,
    ):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        gcn_type = 'GraphConv'
        assert gcn_type in ['GraphConv', 'EdgeConv', 'TAGConv']
        GCNLayer = getattr(dgl.nn, gcn_type)
        # List of layers for node feature message passing
        self.node_conv_layers = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            self.node_conv_layers.append(
                GCNLayer(node_dim, node_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        # linear functions for graph average poolings of output
        self.pool = dgl.nn.AvgPooling()
        self.linear = MLP(1, node_dim, 0, node_dim)

    def forward(self, g, h, he):
        # g = dgl.add_self_loop(g)

        for i in range(self.num_layers):
            # Update node features
            h = self.node_conv_layers[i](g, h)
            h = F.relu(h)
        
        local_feat = h
        # perform graph sum pooling over all nodes
        global_feat = self.linear(self.pool(g, local_feat))
        return local_feat, global_feat


class SAGE(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        num_layers,
        delta,
        mlp_ratio=4,
        drop=0.,
        drop_path=0.,
        conv_on_edge=True,
    ):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        aggregator_type='pool'
        # List of layers for node feature message passing
        self.node_conv_layers = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            self.node_conv_layers.append(
                dgl.nn.SAGEConv(node_dim, node_dim, aggregator_type)
            )

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        # linear functions for graph average poolings of output
        self.pool = dgl.nn.AvgPooling()
        self.linear = MLP(1, node_dim, 0, node_dim)

    def forward(self, g, h, he):
        # g = dgl.add_self_loop(g)

        for i in range(self.num_layers):
            # Update node features
            h = self.node_conv_layers[i](g, h)
            h = F.relu(h)
        
        local_feat = h
        # perform graph sum pooling over all nodes
        global_feat = self.linear(self.pool(g, local_feat))
        return local_feat, global_feat


class GIN(nn.Module):
    def __init__(self, 
                 node_dim,
                 edge_dim,
                 num_layers,
                 delta,
                 mlp_ratio=4,
                 drop=0.,
                 drop_path=0.,
                 conv_on_edge=True):
        super().__init__()
        input_dim = node_dim
        output_dim = node_dim 
        hidden_dim = node_dim
        num_mlp_layers=2
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        # five-layer GCN with l-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1): # excluding the input layer
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(dgl.nn.GINConv(mlp, learn_eps=False, aggregator_type='max')) # set to True if learning epsilon
            self.batch_norms.append(nn.LayerNorm(hidden_dim))
        
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        
        self.drop1 = nn.Dropout(0.3)
        self.drop = nn.Dropout(0.5)
        self.pool = dgl.nn.AvgPooling() # change to mean readout (AvgPooling) on social network datasets

    def forward(self, g, h, he):
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        
        out = hidden_rep[-1]
        out = self.drop1(out)
        score_over_layer = 0

        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))

        return out, score_over_layer

    
class GAT(nn.Module):
    def __init__(self, 
                 node_dim,
                 edge_dim,
                 num_layers,
                 delta,
                 mlp_ratio=4,
                 drop=0.,
                 drop_path=0.,
                 conv_on_edge=True):
        super().__init__()
        in_size = node_dim
        hid_size = node_dim
        out_size = node_dim
        heads=[4, 4, 6]
        self.gat_layers = nn.ModuleList()
        # three-layer GAT
        self.gat_layers.append(
            dgl.nn.GATConv(in_size, hid_size, heads[0], activation=F.elu)
        )
        self.gat_layers.append(
            dgl.nn.GATConv(
                hid_size * heads[0],
                hid_size,
                heads[1],
                residual=True,
                activation=F.elu,
            )
        )
        self.gat_layers.append(
            dgl.nn.GATConv(
                hid_size * heads[1],
                out_size,
                heads[2],
                residual=True,
                activation=None,
            )
        )
        
        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        # linear functions for graph average poolings of output
        self.pool = dgl.nn.AvgPooling()
        self.linear = MLP(1, node_dim, 0, node_dim)

    def forward(self, g, h, he):
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 2:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        local_feat = h
        # perform graph sum pooling over all nodes
        global_feat = self.linear(self.pool(g, local_feat))
        return local_feat, global_feat



        
class AAGNetGraphEncoder(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        num_layers,
        delta,
        mlp_ratio=4,
        drop=0.,
        drop_path=0.,
        conv_on_edge=True
    ):
        """

        Args:
            input_dim (int): [description]
            input_edge_dim (int): [description]
            output_dim (int): [description]
            num_layers (int, optional): [description].
        """
        super(AAGNetGraphEncoder, self).__init__()
        self.num_layers = num_layers
        self.conv_on_edge = conv_on_edge
        self.node_convs = nn.ModuleList()
        self.edge_convs = nn.ModuleList()        
        # since 2nd layer, the subsequent layers are share-weight
        for _ in range(2):
            if self.conv_on_edge:
                self.edge_convs.append(
                    EdgeMPNN(node_dim, edge_dim, mlp_ratio, drop, drop_path))
            self.node_convs.append(
                NodeMPNN(node_dim, edge_dim, delta, mlp_ratio, drop, drop_path))

        self.post_norm = nn.LayerNorm(node_dim)
        # linear functions for graph average poolings of output
        self.pool = dgl.nn.AvgPooling()
        self.linear = MLP(1, node_dim, 0, node_dim, nn.LayerNorm, True)
    
    def forward(self, g, h, he):
        # first layer
        if self.conv_on_edge:
            he = self.edge_convs[0](g, h, he)
        h = self.node_convs[0](g, h, he)
        
        # subsequent share-weight layer
        for i in range(self.num_layers-1):
            if self.conv_on_edge:
                he = self.edge_convs[1](g, h, he)
            h = self.node_convs[1](g, h, he)
        
        local_feat = self.post_norm(h)
        # perform graph sum pooling over all nodes
        global_feat = self.linear(self.pool(g, local_feat))
        return local_feat, global_feat

    
