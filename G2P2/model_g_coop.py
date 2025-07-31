import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import model
from simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch_geometric.nn.inits import zeros
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dropout_edge
from torch_geometric.utils import to_undirected
from torch_geometric.utils import remove_self_loops, coalesce
from torch_geometric.utils import dense_to_sparse


_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class Text_Prompt(nn.Module):
    def __init__(self, args, classnames, clip_model, device):
        super().__init__()
        self.vars = nn.ParameterList()
        n_cls = len(classnames)
        dtype = clip_model.dtype

        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [name for name in classnames]

        tokenized_prompts = torch.cat(
            [model.tokenize(p, context_length=args.context_length) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + args.text_prompt_len:, :])  # CLS, EOS

        ctx_vectors = embedding[0, 1: 1 + args.text_prompt_len, :].unsqueeze(0)
        self.ctx_init_state = ctx_vectors.detach().clone()

        self.ctx = nn.Parameter(ctx_vectors, requires_grad=True)  # to be optimized
        self.vars.append(self.ctx)


        self.n_cls = n_cls
        self.n_ctx = args.text_prompt_len
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

        # self.reset()


    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors)
        

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
        return prompts

    def parameters(self):
        return self.vars


class SimplePrompt(nn.Module):
    def __init__(self, in_channels):
        super(SimplePrompt, self).__init__()
        self.vars = nn.ParameterList()
        self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))
        self.vars.append(self.global_emb)
        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.global_emb)

    def forward(self, x):
        return x + self.global_emb

    def parameters(self):
        return self.vars


class LightPrompt(torch.nn.Module):
    def __init__(self, token_dim, token_num_per_group):
        super(LightPrompt, self).__init__()
        self.token_num = token_num_per_group
        self.token_list = torch.nn.Parameter(torch.empty(token_num_per_group, token_dim))
        self.shared_edge_weight = nn.Parameter(torch.tensor(1., dtype=torch.float))

        self.token_init(init_method="kaiming_uniform")
        adj = torch.ones(token_num_per_group, token_num_per_group)
        edge_index, _ = dense_to_sparse(adj)
        edge_index, _ = remove_self_loops(edge_index)
        self.register_buffer('edge_index', edge_index)


    def token_init(self, init_method="kaiming_uniform"):
        if init_method == "zeros":
            nn.init.zeros_(self.token_list)
        elif init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.token_list, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        else:
            raise ValueError("Only support 'zeros' and 'kaiming_uniform' initialization methods.")
    

    def get_prompt(self):
        edge_weight = self.shared_edge_weight.expand(self.edge_index.size(1))
        data = Data(x=self.token_list, edge_index=self.edge_index, edge_attr=edge_weight, y=torch.tensor([-1]).long())
        return data


class HeavyPrompt(LightPrompt):
    def __init__(self, token_dim, token_num, len):
        super(HeavyPrompt, self).__init__(token_dim, token_num)  # only has one prompt graph.
        self.edge_weight = nn.Parameter(torch.empty(len+1))
        nn.init.ones_(self.edge_weight)


    def forward(self, graph_list, de, drop_e):
        pg = self.get_prompt()
        inner_edge_index = pg.edge_index
        inner_edge_weight = pg.edge_attr
        token_num = pg.x.shape[0]

        re_graph_list = []
        for g in Batch.to_data_list(graph_list):
            if de:
                g.edge_index, edge_mask = dropout_edge(g.edge_index, p=drop_e)
                # g.edge_attr = g.edge_attr[edge_mask]
            
            num_nodes = g.x.shape[0]
            g_edge_index = g.edge_index + token_num
            root_n_index = g.root_n_index + token_num

            pi, gi = torch.meshgrid(
                torch.arange(token_num, device=g.x.device),
                torch.arange(num_nodes, device=g.x.device),
                indexing='ij'
            )
            cross_edge_index = torch.stack([
                pi.reshape(-1), 
                gi.reshape(-1) + token_num
            ], dim=0)

            spd = g.spd  # [num_nodes]
            spd_clamped = torch.clamp(spd, min=0, max=self.edge_weight.size(0) - 1)
            spd_clamped_repeated = spd_clamped.repeat(token_num)
            cross_edge_weight = self.edge_weight[spd_clamped_repeated[gi.reshape(-1)]]

            cross_edge_index, cross_edge_weight = to_undirected(cross_edge_index, cross_edge_weight, reduce="mean")

            edge_dtype = cross_edge_weight.dtype
            device = cross_edge_weight.device
            g_edge_weight = torch.ones(g_edge_index.size(1), dtype=edge_dtype, device=device)
            
            x = torch.cat([pg.x, g.x], dim=0)

            edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
            edge_weight = torch.cat([inner_edge_weight, g_edge_weight, cross_edge_weight], dim=0)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, root_n_index=root_n_index)
            # data.edge_index, data.edge_attr = to_undirected(data.edge_index, data.edge_attr, reduce='mean')

            re_graph_list.append(data)

        graphp_batch = Batch.from_data_list(re_graph_list)
        return graphp_batch
    

class CoOp(nn.Module):
    def __init__(self, args, classnames, clip_model, in_dim, token_num, path_length, device):
        super().__init__()
        self.args = args
        self.classnames = classnames

        self.prompt_learner = HeavyPrompt(in_dim, token_num, path_length)

        self.text_prompt = Text_Prompt(args, classnames, clip_model, device)
        self.tokenized_prompts = self.text_prompt.tokenized_prompts
        self.image_encoder = clip_model.gnn
        self.text_model = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype


    def forward(self, graph_batch, de=False, drop_e=0.0):
        graph_batch = self.prompt_learner(graph_batch, de, drop_e)
        x, adj, s_n = graph_batch.x, graph_batch.edge_index, graph_batch.root_n_index
        image_features = self.image_encoder(x, adj, graph_batch.edge_attr)
        image_features_ = image_features[s_n]

        prompts = self.text_prompt()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_model(prompts, tokenized_prompts)


        image_features = image_features_ / image_features_.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, image_features_


    def inference(self, graph_batch, de=False, drop_e=0.0):
        with torch.no_grad():
            graph_batch = self.prompt_learner(graph_batch, de, drop_e)
            x, adj, s_n = graph_batch.x, graph_batch.edge_index, graph_batch.root_n_index
            image_features = self.image_encoder(x, adj, graph_batch.edge_attr)
            image_features = image_features[s_n]

            prompts = self.text_prompt()
            prompts = prompts.to(x.device)
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_model(prompts, tokenized_prompts)

            image_features_ = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features_ @ text_features.t()

        return logits, image_features
