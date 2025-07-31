import os
import pandas as pd

import argparse
import torch
from torch_geometric import seed_everything
from transformers import AutoTokenizer

from data.load import load_data
from models import GraphCLIP, CoOp
from utils.args import Arguments
from utils.process import parse_target_data, split_dataloader
import torch.nn.functional as F
import yaml
import numpy as np
from tqdm import tqdm

acc_log = {}

def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

eval_template={
    'cora': "this paper has a topic on {c}", 
    'citeseer': "good paper of {c}",
    'wikics': "it belongs to {c} research area",
    'photo':  "this product belongs to {c}",
    'computer':  "is {c} category", 
    'instagram': "{c}",
}

def entropy(input_):
    return -(input_ * torch.log(input_ + 1e-9)).sum(1)

def div(softmax_output):
    mean_softmax_output = softmax_output.mean(dim=0)
    diversity_loss = torch.sum(-mean_softmax_output * torch.log(mean_softmax_output + 1e-9))
    return diversity_loss


def im(output):
    erm_prob = F.softmax(output, dim=-1)
    en = entropy(erm_prob).mean(0)
    di = div(erm_prob)
    return en - di

def log_acc(epoch, phase, acc):
    key = f"epoch {epoch}, {phase}"
    if key not in acc_log:
        acc_log[key] = []
    acc_log[key].append(acc)


def test_time_tuning(model, data, loader, optimizer, optimizer1, args):
    for j in range(args.tta_step):

        for name, param in model.named_parameters():
            if "prompt_learner" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

        for i_batch, sample_batched in enumerate(loader):
            graph = sample_batched.to(device)
            pred, image_features = model.forward(graph)

            pred_aug, image_features_ = model.forward(graph, True, args.drop_e)
            sim_matrix = torch.matmul(F.normalize(image_features, dim=1), F.normalize(image_features_, dim=1).T)
            sim_matrix = sim_matrix / args.temp  # [B, B]

            ## negative sample
            prob = F.softmax(pred, dim=-1)
            label_pred = torch.argmax(prob, -1)

            label_matrix = label_pred.unsqueeze(1) != label_pred.unsqueeze(0)
            exp_all_sims = torch.exp(sim_matrix) * label_matrix

            ## positive sample
            pos_sims = torch.exp(sim_matrix) * (~label_matrix)

            denominator = exp_all_sims.sum(dim=1) + pos_sims.sum(dim=1)
            denominator = denominator + 1e-8
            loss_per_sample = -torch.log(pos_sims.sum(dim=1) / denominator)  # [B]

            loss = loss_per_sample.mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        for name, param in model.named_parameters():
            if "text_prompt" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

        prob_list = []
        for i_batch, sample_batched in enumerate(loader):
            graph = sample_batched.to(device)
            pred, image_features = model.forward(graph)
            loss = im(pred)
                
            loss.backward()
            optimizer1.step()
            optimizer1.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--ckpt', type=str, help="the name of checkpoint", default='graphclip_arxiv_29')
    parser.add_argument('--lm_type', type=str, help="the type of lm", default='tiny', 
                                choices=['tiny', 'sbert', 'deberta', 'bert', 'e5', 'llama2', 'llama3', 'llama2-14', 'qwen2', 'qwen2.5-0.5b', 'tiny', 'sbert2'])

    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--source_data', type=str, help="dataset name", default='arxiv')
    parser.add_argument('--target_data', type=str, help="dataset name", default='instagram')
    
    parser.add_argument('--batch_size', type=int, help="the batch size", default=64)

    parser.add_argument('--hard_prompt', type=str2bool, default="true")
    parser.add_argument('--des', type=bool, default=True)

    parser.add_argument('--tta', type=bool, default=True)
    parser.add_argument('--tta_step', type=int, default=20)

    parser.add_argument('--token_num', type=int, default=1)
    parser.add_argument('--text_prompt_len', type=int, default=1)
    parser.add_argument('--graph_prompt_lr', type=float, default=0.001)
    parser.add_argument('--text_prompt_lr', type=float, default=0.001)
    parser.add_argument('--prompt_wd', type=float, default=0.0)

    parser.add_argument('--drop_e', type=float, default=0.5)
    parser.add_argument('--temp', type=float, default=0.5)

    args = parser.parse_args()

    config = load_config(f"./config/{args.source_data}/{args.target_data}.yaml")
    for key, value in config.items():
        setattr(args, key, value)

    if args.target_data.lower() in ['photo', 'computer']:
        seed_list = list(range(1))
    else:
        seed_list = list(range(1))
    
    results = []
    for seed in tqdm(seed_list, desc=f"Running seeds for {args.target_data}"):
        seed_everything(seed) 
        attn_kwargs = {'dropout': 0.0}

        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
        
        graph_clip = GraphCLIP(384, 1024, 12, attn_kwargs, text_model=args.lm_type)
        graph_clip.load_state_dict(torch.load(f"../checkpoints/graphclip_{args.source_data}_29.pt", map_location='cpu'), strict=False)
        graph_clip.to(device)
        
        data, text, classes, c_descs = load_data(args.target_data, seed=0)
        target_graph, path_length = parse_target_data(args.target_data, data)
        
        if not args.des:
            text_inputs = ["X "*args.text_prompt_len+c for c in classes]
        else:
            text_inputs = ["X "*args.text_prompt_len+c+des for c,des in zip(classes, c_descs)]


        loader = split_dataloader(data, target_graph, args.batch_size, seed=0, name=args.target_data)


        in_dim = 384
        model = CoOp(args, text_inputs, graph_clip, in_dim, args.token_num, path_length, device)
        model = model.to(device)

        trainable_param = model.prompt_learner.parameters()
        optimizer = torch.optim.Adam(trainable_param, args.graph_prompt_lr, weight_decay=args.prompt_wd)

        trainable_param1 = model.text_prompt.parameters()
        optimizer1 = torch.optim.Adam(trainable_param1, args.text_prompt_lr, weight_decay=args.prompt_wd)

        for name, param in model.named_parameters():
            param.requires_grad_(False)

        model.eval()
        if args.tta:
            test_time_tuning(model, data, loader, optimizer, optimizer1, args)

        correct = 0
        for i, batch in enumerate(loader):
            batch = batch.to(device)

            with torch.no_grad():
                similarity, _ = model.inference(batch)
                y = batch.y
                correct += torch.sum(similarity.argmax(dim=1) == y).item()

        acc = correct / data.test_mask.sum().item()
        print(f"{round(acc * 100, 3)}")
        torch.save(model.state_dict(), f"graphclip_{args.target_data}_tta.pth")
    
    


