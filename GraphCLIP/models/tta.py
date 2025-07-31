
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dropout_edge
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_undirected


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
        self.token_pe = torch.nn.Parameter(torch.empty(token_num_per_group, 32))
        self.shared_edge_weight = nn.Parameter(torch.tensor(1., dtype=torch.float))

        self.token_init(init_method="kaiming_uniform")
        adj = torch.ones(token_num_per_group, token_num_per_group)
        edge_index, _ = dense_to_sparse(adj)
        edge_index, _ = remove_self_loops(edge_index)
        self.register_buffer('edge_index', edge_index)


    def token_init(self, init_method="kaiming_uniform"):
        if init_method == "zeros":
            nn.init.zeros_(self.token_list)
            nn.init.zeros_(self.token_pe)
        elif init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.token_list, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
            # nn.init.kaiming_uniform_(self.token_pe, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
            nn.init.zeros_(self.token_pe)
        else:
            raise ValueError("Only support 'zeros' and 'kaiming_uniform' initialization methods.")
            

    def get_prompt(self):
        edge_weight = self.shared_edge_weight.expand(self.edge_index.size(1))
        data = Data(x=self.token_list, pe=self.token_pe, edge_index=self.edge_index, edge_weight=edge_weight, y=torch.tensor([-1]).long())
        return data


class HeavyPrompt(LightPrompt):
    def __init__(self, token_dim, token_num, len):
        super(HeavyPrompt, self).__init__(token_dim, token_num)  # only has one prompt graph.
        self.edge_weight = nn.Parameter(torch.empty(len+1))
        nn.init.ones_(self.edge_weight)


    def forward(self, graph_list, de, drop_e):
        pg = self.get_prompt()
        inner_edge_index = pg.edge_index
        inner_edge_weight = pg.edge_weight
        token_num = pg.x.shape[0]

        # transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
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
            pe = torch.cat([pg.pe, g.pe], dim=0)

            edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
            edge_weight = torch.cat([inner_edge_weight, g_edge_weight, cross_edge_weight], dim=0)
            
            data = Data(x=x, pe=pe, edge_index=edge_index, edge_weight=edge_weight, root_n_index=root_n_index)
            
            # data.edge_index, data.edge_weight = to_undirected(data.edge_index, data.edge_weight, reduce='mean')
            # data=transform(data) # add PE

            re_graph_list.append(data)

        graphp_batch = Batch.from_data_list(re_graph_list)
        return graphp_batch
    

class TextEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.position_ids = clip_model.text_model.embeddings.position_ids
        self.config = clip_model.text_model.config

        self.position_embeddings = clip_model.text_model.embeddings.position_embeddings
        self.LayerNorm = clip_model.text_model.embeddings.LayerNorm
        self.dropout = clip_model.text_model.embeddings.dropout
        self.encoder = clip_model.text_model.encoder
        self.pooler = clip_model.text_model.pooler
        self.get_head_mask = clip_model.text_model.get_head_mask

    def forward(self, prompts, attention_mask):
        input_shape = prompts.size()  # [nclass, seq, dim]
        seq_length = input_shape[1]
        position_ids = self.position_ids[:, 0 : seq_length]
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = prompts + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
            attention_mask, embeddings.dtype, tgt_len=seq_length
        )
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embeddings,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
                    last_hidden_state=sequence_output,
                    pooler_output=pooled_output,
                    past_key_values=encoder_outputs.past_key_values,
                    hidden_states=encoder_outputs.hidden_states,
                    attentions=encoder_outputs.attentions,
                    cross_attentions=encoder_outputs.cross_attentions,
        )


class TextPrompt(torch.nn.Module):
    def __init__(self, args, classnames, clip_model, device):
        super().__init__()
        self.vars = nn.ParameterList()
        n_cls = len(classnames)
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        # self.tokenizer = AutoTokenizer.from_pretrained('../all-MiniLM-L6-v2', local_files_only=True)

        tokenized_prompts = self.tokenizer(classnames, truncation=True, padding=True, return_tensors="pt", max_length=512).to(self.device)

        with torch.no_grad():
            inputs_embeds = clip_model.text_model.embeddings.word_embeddings(tokenized_prompts['input_ids'])
            seq_length = (tokenized_prompts['input_ids']).shape[1]
            batch_size = (tokenized_prompts['input_ids']).shape[0]
            buffered_token_type_ids = clip_model.text_model.embeddings.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
            token_type_embeddings = clip_model.text_model.embeddings.token_type_embeddings(buffered_token_type_ids_expanded)
            embedding = (inputs_embeds + token_type_embeddings)

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
    def __init__(self, args, classnames, clip_model, in_dim, token_num, path_length, device):
        super().__init__()
        self.args = args
        self.classnames = classnames

        self.graph_model = clip_model.graph_model
        self.text_model = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale

        self.prompt_learner = HeavyPrompt(in_dim, token_num, path_length)
        self.text_prompt = TextPrompt(args, classnames, clip_model, device)


    def forward(self, graph_batch, de=False, drop_e=0.0):
        graph_batch = self.prompt_learner(graph_batch, de, drop_e)
        graph_embs_, center_embs = self.graph_model(graph_batch.x, graph_batch.pe, graph_batch.edge_index, graph_batch.edge_weight, graph_batch.batch, graph_batch.root_n_index)
        graph_embs = graph_embs_ / graph_embs_.norm(dim=-1, keepdim=True)

        prompts, attention_mask = self.text_prompt()   # [nclass, seq_len, dim]
        text_output = self.text_model(prompts, attention_mask)
        text_embs = mean_pooling(text_output.last_hidden_state, attention_mask)

        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * graph_embs @ text_embs.t()

        return logits, graph_embs_


    def inference(self, graph_batch, de=False, drop_e=0.0):
        with torch.no_grad():
            graph_batch = self.prompt_learner(graph_batch, de, drop_e)
            graph_embs, center_embs = self.graph_model(graph_batch.x, graph_batch.pe, graph_batch.edge_index, graph_batch.edge_weight, graph_batch.batch, graph_batch.root_n_index)
            graph_embs_ = graph_embs / graph_embs.norm(dim=-1, keepdim=True)

            prompts, attention_mask = self.text_prompt()   # [nclass, seq_len, dim]
            text_output = self.text_model(prompts, attention_mask)
            text_embs = mean_pooling(text_output.last_hidden_state, attention_mask)

            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * graph_embs_ @ text_embs.t()
        return logits, graph_embs
