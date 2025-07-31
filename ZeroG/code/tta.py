
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dropout_edge
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_undirected

from tqdm.autonotebook import trange
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers import AutoTokenizer


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class LightPrompt(torch.nn.Module):
    def __init__(self, token_dim, token_num_per_group):
        super(LightPrompt, self).__init__()
        self.token_num = token_num_per_group
        self.token_list = torch.nn.Parameter(torch.empty(token_num_per_group, token_dim))
        # self.shared_edge_attr = torch.nn.Parameter(torch.empty(1, token_dim))
        # self.shared_edge_attr = nn.Parameter(torch.empty(token_dim))
        self.shared_edge_weight = nn.Parameter(torch.tensor(1., dtype=torch.float))

        self.token_init(init_method="kaiming_uniform")

        adj = torch.ones(token_num_per_group, token_num_per_group)
        edge_index, _ = dense_to_sparse(adj)
        edge_index, _ = remove_self_loops(edge_index)
        self.register_buffer('edge_index', edge_index)


    def token_init(self, init_method="kaiming_uniform"):
        if init_method == "zeros":
            nn.init.zeros_(self.token_list)
            # nn.init.zeros_(self.token_pe)
        elif init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.token_list, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
            # nn.init.kaiming_uniform_(self.shared_edge_attr.unsqueeze(0), nonlinearity='leaky_relu', mode='fan_in', a=0.01)
            # nn.init.kaiming_uniform_(self.token_pe, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
            # nn.init.zeros_(self.token_pe)
        else:
            raise ValueError("Only support 'zeros' and 'kaiming_uniform' initialization methods.")
            

    def get_prompt(self):
        edge_weight = self.shared_edge_weight.expand(self.edge_index.size(1))
        # edge_attr = self.shared_edge_attr.expand(self.edge_index.size(1), -1)
        data = Data(x=self.token_list, edge_index=self.edge_index, edge_weight=edge_weight, y=torch.tensor([-1]).long())
        return data


class HeavyPrompt(LightPrompt):
    def __init__(self, token_dim, token_num, len):
        super(HeavyPrompt, self).__init__(token_dim, token_num)  # only has one prompt graph.
        self.edge_weight = nn.Parameter(torch.empty(len+1))
        # self.edge_attr = nn.Parameter(torch.empty(token_dim))

        # nn.init.kaiming_uniform_(self.edge_attr.unsqueeze(0), nonlinearity='leaky_relu', mode='fan_in', a=0.01)        
        nn.init.ones_(self.edge_weight)


    def forward(self, graph_list, de, drop_e):

        pg = self.get_prompt()
        inner_edge_index = pg.edge_index
        inner_edge_weight = pg.edge_weight
        # inner_edge_attr = pg.edge_attr
        token_num = pg.x.shape[0]
        device = graph_list.x.device

        re_graph_list = []
        for g in Batch.to_data_list(graph_list):
            if de:
                g.edge_index, edge_mask = dropout_edge(g.edge_index, p=drop_e)

            num_nodes = g.x.shape[0]
            x = torch.cat([pg.x, g.x], dim=0)
            root_n_index = g.root_n_index + token_num
            g_edge_index = g.edge_index + token_num

            pi, gi = torch.meshgrid(
                torch.arange(token_num, device=g.x.device),
                torch.arange(num_nodes, device=g.x.device),
                indexing='ij'
            )
            cross_edge_index = torch.stack([
                pi.reshape(-1), 
                gi.reshape(-1) + token_num
            ], dim=0)

            spd = g.spd
            spd_clamped = torch.clamp(spd, min=0, max=self.edge_weight.size(0) - 1)
            cross_edge_weight = self.edge_weight[spd_clamped[gi.reshape(-1)]]
            spd_clamped_repeated = spd_clamped.repeat(token_num)
            cross_edge_weight = self.edge_weight[spd_clamped_repeated[gi.reshape(-1)]]

            cross_edge_index, cross_edge_weight = to_undirected(cross_edge_index, cross_edge_weight, reduce="mean")

            g_edge_weight = torch.ones(g_edge_index.size(1), dtype=cross_edge_weight.dtype, device=device)

            edge_index = torch.cat([inner_edge_index, cross_edge_index, g_edge_index], dim=1)
            edge_weight = torch.cat([inner_edge_weight, cross_edge_weight, g_edge_weight], dim=0)

            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, root_n_index=root_n_index)
            re_graph_list.append(data)

        graphp_batch = Batch.from_data_list(re_graph_list)
        return graphp_batch
    

class TextEncoder(torch.nn.Module):
    def __init__(self, text_lora):
        super().__init__()
        self.lora_model = text_lora.lora_model

    def forward(self, prompts, attention_mask):
        outputs = self.lora_model(
            inputs_embeds=prompts,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs[0][:, 0, :]
    


class TextPrompt(torch.nn.Module):
    def __init__(self, args, classnames, text_lora, device):
        super().__init__()
        self.vars = nn.ParameterList()
        n_cls = len(classnames)
        self.device = device

        with torch.no_grad():
            tokenized_prompts = text_lora.tokenizer(classnames, max_length=256, return_tensors='pt', truncation=True, padding=True).to(args.device)
            embedding = text_lora.lora_model.model.embeddings.word_embeddings(tokenized_prompts['input_ids'])

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + args.text_prompt_len :, :])

        ctx_vectors = embedding[0, 1: 1 + args.text_prompt_len, :].unsqueeze(0)
        self.ctx_init_state = ctx_vectors.clone()
        self.ctx = torch.nn.Parameter(ctx_vectors, requires_grad=True)  # to be optimized
        self.vars.append(self.ctx)

        self.tokenized_prompts = tokenized_prompts
        self.n_cls = n_cls
        self.n_ctx = args.text_prompt_len
        self.classnames = classnames


    def forward(self):
        ctx = self.ctx
        ctx = ctx.expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts, self.tokenized_prompts['attention_mask']


    def parameters(self):
        return self.vars



class CoOp(nn.Module):
    def __init__(self, args, classnames, text_lora, in_dim, token_num, path_length, device):
        super().__init__()
        self.args = args
        self.classnames = classnames

        self.text_model = TextEncoder(text_lora)
        self.text_prompt = TextPrompt(args, classnames, text_lora, device)
        self.prompt_learner = HeavyPrompt(in_dim, token_num, path_length)


    def forward(self, desc_embed, graph_batch, de=False, drop_e=0.0):
        prompts, attention_mask = self.text_prompt()   # [nclass, seq_len, dim]
        text_output = self.text_model(prompts, attention_mask)
        graph_batch = self.prompt_learner(graph_batch, de, drop_e)
        outputs, embs = self.zero_shot_eval(graph_batch.x, desc_embed, text_output, graph_batch)

        return outputs, embs

    def zero_shot_eval(self, node_embeds, desc_embed, label_embeds, data):
        node_embeds = torch.cat([node_embeds, desc_embed], dim=0)

        if self.args.if_norm:
            node_embeds = (node_embeds - node_embeds.mean(0)) / node_embeds.std(0)
            label_embeds = (label_embeds - label_embeds.mean(0)) / label_embeds.std(0)

        num_existing_nodes = data.x.shape[0] + 1
        virtual_node_index = data.x.shape[0]

        # 更新 edge_index 和 edge_weight
        new_edges_to_virtual = []
        for node_idx in range(num_existing_nodes - 1):
            new_edges_to_virtual.append([node_idx, virtual_node_index])
            new_edges_to_virtual.append([virtual_node_index, node_idx])

        new_edge_index = torch.cat([
            data.edge_index.t(),
            torch.tensor(new_edges_to_virtual, dtype=torch.long, device=self.args.device)
        ], dim=0).t()

        original_edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else torch.ones(data.edge_index.shape[1], device=self.args.device)
        new_edge_weight = torch.ones(len(new_edges_to_virtual), device=self.args.device)
        edge_weight = torch.cat([original_edge_weight, new_edge_weight], dim=0)

        adj_normed = self.normalize_adjacency_matrix(new_edge_index, edge_weight, num_existing_nodes)

        for _ in range(self.args.R):
            node_embeds = torch.sparse.mm(adj_normed, node_embeds)

        node_embeds = node_embeds[data.root_n_index, :]

        node_embeds_ = node_embeds / node_embeds.norm(dim=-1, keepdim=True)
        label_embeds = label_embeds / label_embeds.norm(dim=-1, keepdim=True)
        dists = torch.einsum('bn,cn->bc', node_embeds_, label_embeds)

        return dists, node_embeds


    def normalize_adjacency_matrix(self, edge_index, edge_weight, num_nodes):
        # 1. 增加 self-loop
        self_loop = torch.arange(num_nodes, device=self.args.device)
        self_loop = torch.stack([self_loop, self_loop], dim=0)
        edge_index = torch.cat([edge_index, self_loop], dim=1)
        edge_weight = torch.cat([edge_weight, torch.ones(num_nodes, device=self.args.device)], dim=0)

        # 2. 计算 D^{-1/2}
        deg = torch.zeros(num_nodes, device=self.args.device).scatter_add_(0, edge_index[0], edge_weight)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # 3. 归一化边权重 w_{ij} = w / sqrt(D_i * D_j)
        norm_weight = deg_inv_sqrt[edge_index[0]] * edge_weight * deg_inv_sqrt[edge_index[1]]

        adj = torch.sparse_coo_tensor(edge_index, norm_weight, (num_nodes, num_nodes))
        return adj.coalesce()
