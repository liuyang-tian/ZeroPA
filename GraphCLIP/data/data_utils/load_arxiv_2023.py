import torch
import torch
import os.path as osp
import pandas as pd


def get_raw_text_arxiv_2023(use_text=False, seed=0):
    if osp.exists(f"../../datasets/pretrain/arxiv_2023.pt"):
        data = torch.load(f"../../datasets/pretrain/arxiv_2023.pt", map_location='cpu')
        data.num_nodes = data.y.shape[0]


        df = pd.read_csv('../../datasets/pretrain/arxiv_2023/paper_info.csv')
        raw_texts = []
        for ti, ab in zip(df['title'], df['abstract']):
            raw_texts.append(f'Title: {ti}\nAbstract: {ab}')

        # raw_texts = [] # we do not need raw texts for source data, because we already transform them into node features use miniLM
        return data, raw_texts
    else:
        raise NotImplementedError('No existing arxiv_2023 dataset!')

data, raw_texts = get_raw_text_arxiv_2023()
data.raw_texts = raw_texts