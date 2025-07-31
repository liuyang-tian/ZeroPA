
import numpy as np
import argparse
import torch
import random
import time
from model import CLIP, tokenize
from torch import nn, optim
from sklearn import preprocessing
from model_g_coop import CoOp
import json
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph, to_undirected, remove_isolated_nodes, dropout_adj, remove_self_loops, k_hop_subgraph, to_edge_index, to_dgl
from torch_sparse import SparseTensor
from copy import deepcopy
from torch_geometric.data import Data
from torch_geometric.data import Batch
import copy
from torch_geometric.utils import dropout_edge
from data_utils import SourceGraphDataset, SourceGraphDataset_Wiki
import torch.nn.functional as F
import os
import pandas as pd
from tqdm import tqdm
import yaml


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
    'photo':  "{c}",
    'computer':  "is {c} category", 
    'instagram': "{c}",
}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def entropy(input_):
    return -(input_ * torch.log(input_ + 1e-9)).sum(1)

def cross_entropy(output, scale_thre):
    erm_prob = F.softmax(output, dim=-1)

    label_pred = torch.argmax(erm_prob, -1)
    uncertainty = entropy(erm_prob)
    threshold = scale_thre * torch.log(
        torch.tensor(erm_prob.size(1), dtype=torch.float, device=erm_prob.device)
    )
    mask = uncertainty < threshold
    data_mask = label_pred != (erm_prob.size(1) - 1)

    label_one_hot = torch.nn.functional.one_hot(
        label_pred, num_classes=erm_prob.size(1)
    )

    prob_ = label_one_hot.float().detach()
    mask_ = torch.logical_and(mask, data_mask).detach()

    loss = -(prob_[mask_] * torch.log(erm_prob[mask_] + 1e-9)).sum(1).mean(0)
    return loss


def div(softmax_output):
    mean_softmax_output = softmax_output.mean(dim=0)
    diversity_loss = torch.sum(-mean_softmax_output * torch.log(mean_softmax_output + 1e-9))
    return diversity_loss


def im(output):
    erm_prob = F.softmax(output, dim=-1)
    en = entropy(erm_prob).mean(0)
    di = div(erm_prob)
    return en - di


def inner(t1, t2):
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return (1-(t1 * t2).sum(1)).mean()


def cosine_sim(t1, t2):
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return (t1 * t2).sum(1)


def pic_loss(feats, prob):
    prob = F.softmax(prob, dim=-1)
    _, c = prob.shape
    mus = (prob.T @ feats) / prob.sum(dim=0).view(c, 1)
    sq_dist = torch.square(torch.cdist(feats, mus, p=2))
    var_intra = (sq_dist * prob).sum()
    var_total = torch.sum(torch.square(feats - feats.mean(dim=0)))
    return var_intra / var_total


# def log_acc(epoch, phase, acc):
#     key = f"epoch {epoch}, {phase}"
#     acc_log[key] = acc

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

            print(image_features_.detach())
            print(loss.item())

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
            prob_list.append(pred)

            loss = im(pred)            
            loss.backward()
            optimizer1.step()
            optimizer1.zero_grad()


def main(args):
    if args.target_data.lower() in ['photo', 'computer']:
        seed_list = list(range(1))
    else:
        seed_list = list(range(1))
    
    for seed in tqdm(seed_list, desc=f"Running seeds for {args.target_data}"):
        setup_seed(seed)

        clip_model = CLIP(args)
        if args.source_data == "reddit":
            clip_model.load_state_dict(torch.load(f'../res/{args.source_data}/node_ttgt_8&12_0.1_{args.source_data}_1_epoch.pkl', map_location=device))
        else:
            clip_model.load_state_dict(torch.load(f'../res/{args.source_data}/node_ttgt_8&12_0.1_{args.source_data}_1_epoch.pkl', map_location=device))

        data = torch.load(f"../../datasets/test/{args.target_data.lower()}.pt", map_location='cpu')
        data.num_nodes = data.x.shape[0]
        if args.target_data=='ogbn-arxiv':
            data.edge_index = to_undirected(data.edge_index, num_nodes=data.x.shape[0])

        # split data
        if args.target_data!='wikics':
            node_id = np.arange(data.num_nodes)
            np.random.shuffle(node_id)

            data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
            data.val_id = np.sort(
                node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
            data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

            data.train_mask = torch.tensor(
                [x in data.train_id for x in range(data.num_nodes)])
            data.val_mask = torch.tensor(
                [x in data.val_id for x in range(data.num_nodes)])
            data.test_mask = torch.tensor(
                [x in data.test_id for x in range(data.num_nodes)])

        if args.target_data.lower() == 'cora':
            classes = ['Case Based', 'Genetic Algorithms', 'Neural Networks', 'Probabilistic Methods', 'Reinforcement Learning', 'Rule Learning', 'Theory']
            c_descs = [" which refers to research papers focusing on case-based reasoning (CBR) in the field of artificial intelligence. Case-based reasoning is a problem-solving approach that utilizes specific knowledge of previously encountered, concrete problem situations (cases). In this method, a new problem is solved by finding similar past cases and reusing them in the new situation. The approach relies on the idea of learning from past experiences to solve new problems, which makes it relevant in many applications including medical diagnosis, legal decision-making, and others. Thus, the ""Case Based"" category would include papers that primarily focus on this particular methodology and its various aspects.",
                    " which would include research papers related to genetic algorithms (GAs). Genetic algorithms are a type of optimization and search algorithms inspired by the process of natural selection and genetics. These algorithms generate solutions to optimization problems using techniques inspired by natural evolution, such as inheritance, mutation, selection, and crossover. In practice, genetic algorithms can be used to find solutions to complex problems that are difficult to solve with traditional methods, particularly in domains where the search space is large, complex, or poorly understood. This category would cover various aspects of genetic algorithms, including their design, analysis, implementation, theoretical background, and diverse applications.",
                    " which refers to research papers revolving around the concept of artificial neural networks (ANNs). Neural networks are a subset of machine learning algorithms modelled after the human brain, designed to ""learn"" from observational data. They are the foundation of deep learning technologies and can process complex data inputs, find patterns, and make decisions. The network consists of interconnected layers of nodes, or ""neurons"", and each connection is assigned a weight that shapes the data and helps produce a meaningful output. Topics covered under this category could range from the architecture and function of different neural network models, advancements in training techniques, to their application in a multitude of fields such as image and speech recognition, natural language processing, and medical diagnosis.",
                    " which pertains to research papers that focus on probabilistic methods and models in machine learning and artificial intelligence. Probabilistic methods use the mathematics of probability to make predictions and decisions. They provide a framework to handle and quantify the uncertainty and incomplete information, which is a common scenario in real-world problems. This category could include topics like Bayesian networks, Gaussian processes, Markov decision processes, and statistical techniques for prediction and inference. These methods have applications in various areas such as computer vision, natural language processing, robotics, and data analysis, among others, due to their ability to model complex, uncertain systems and make probabilistic predictions.",
                    " which refers to research papers focusing on the area of machine learning known as reinforcement learning (RL). Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to achieve a goal. The agent learns from the consequences of its actions, rather than from being explicitly taught, and adjusts its behavior based on the positive or negative feedback it receives, known as rewards or penalties. This category would include research exploring various RL algorithms, methodologies, theoretical underpinnings, performance enhancements, and practical applications. This field is particularly relevant in areas where decision making is crucial, such as game playing, robotics, resource management, and autonomous driving.",
                    " which pertains to research papers that concentrate on the domain of rule-based learning, also known as rule-based machine learning. Rule learning is a method in machine learning that involves the generation of a set of rules to predict the output in a decision-making system based on the patterns discovered from the data. These rules are often in an ""if-then"" format, making them interpretable and transparent. This category would encompass research involving various rule learning algorithms, their enhancements, theoretical foundations, and applications. Rule learning methods are particularly beneficial in domains where interpretability and understanding of the learned knowledge is important, such as in medical diagnosis, credit risk prediction, and more.",
                    " which likely refers to research papers that delve into the theoretical aspects of machine learning and artificial intelligence. This includes a broad array of topics such as theoretical foundations of various machine learning algorithms, performance analysis, studies on learning theory, statistical learning, information theory, and optimization methods. Additionally, it could encompass the development of new theoretical frameworks, investigations into the essence of intelligence, the potential for artificial general intelligence, as well as the ethical implications surrounding AI. Essentially, the ""Theory"" category encapsulates papers that primarily focus on theoretical concepts and discussions, contrasting with more application-oriented research which centers on specific techniques and their practical implementation."]
        elif args.target_data.lower() == 'citeseer':
            classes = ['Agents', 'Machine Learning', 'Information Retrieval', 'Database', 'Human Computer Interaction', 'Artificial Intelligence']
            c_descs = [". Specifically, agents are autonomous entities that perceive their environment through sensors and act upon it using actuators. They are designed to achieve specific goals or tasks.",
                    ". Specifically, ML research investigates how to create systems that can automatically improve their performance on tasks by identifying patterns and insights from vast amounts of data. Researchers in Machine Learning explore diverse techniques such as supervised learning, unsupervised learning, reinforcement learning, and deep learning to build systems that can predict outcomes, classify data, and make intelligent decisions.",
                    ". Specifically, IR research focuses on the study of information retrieval systems, which are designed to help users find relevant information in large collections of data. Researchers in Information Retrieval explore techniques such as indexing, querying, and ranking to build systems that can efficiently retrieve information based on user queries.",
                    ". Specifically, DB research investigates how to design, build, and manage databases, which are organized collections of data that can be accessed, managed, and updated. Researchers in Database Systems explore techniques such as data modeling, query languages, and transaction processing to build systems that can store, retrieve, and manipulate data.",
                    ". Specifically, HCI research focuses on the study of human-computer interaction, which explores how people interact with computers and other digital technologies. Researchers in Human-Computer Interaction investigate how to design user-friendly interfaces, improve usability, and enhance user experience to build systems that are intuitive, efficient, and effective.",
                    ". Specifically, AI research investigates how to create intelligent systems that can perform tasks that typically require human intelligence, such as perception, reasoning, learning, and decision-making. Researchers in Artificial Intelligence explore diverse techniques such as knowledge representation, planning, and natural language processing to build systems that can solve complex problems, adapt to new environments, and interact with humans."]
        elif args.target_data.lower() == 'wikics':
            classes = ['Computational linguistics', 
                    'Databases', 
                    'Operating systems', 
                    'Computer architecture',
                    'Computer security, Computer network security, Access control, Data security, Computational trust, Computer security exploits',
                    'Internet protocols', 
                    'Computer file systems', 
                    'Distributed computing architecture', 
                    'Web technology, Web software, Web services',
                    'Programming language topics, Programming language theory, Programming language concepts, Programming language classification']
            c_descs = [". Computational linguistics is an interdisciplinary field combining linguistics and computer science to analyze and model natural language. It involves developing algorithms and computational models to understand, generate, and manipulate human language. Applications include machine translation, speech recognition, sentiment analysis, and chatbot development. By leveraging statistical methods and artificial intelligence, computational linguistics aims to enhance human-computer interaction and improve the processing of linguistic data.",
                    ". Databases are organized collections of data, designed to store, manage, and retrieve information efficiently. They enable structured querying and data manipulation through languages like SQL. Databases can be categorized into relational (e.g., MySQL, PostgreSQL) and non-relational (e.g., MongoDB, Cassandra) systems, each suited for different applications and data structures. They play a vital role in various domains, including business, research, and web applications, facilitating data-driven decision-making.",
                    ". Operating systems (OS) are essential software that manage computer hardware and software resources, providing a user interface and facilitating interactions between applications and hardware. Key functions include process management, memory management, file system handling, and device control. Popular operating systems include Windows, macOS, and Linux. OSs enable multitasking, security, and resource allocation, playing a crucial role in the overall functionality and performance of computing devices.",
                    ". Computer architecture is the design and organization of computer systems, encompassing the structure and functionality of hardware components. It includes the CPU, memory hierarchy, and input/output systems, focusing on how they interact to perform tasks efficiently. Key concepts involve instruction sets, parallelism, and microarchitecture. Understanding computer architecture is crucial for optimizing performance, enhancing energy efficiency, and developing new computing technologies, impacting both hardware design and software development.",
                    ". Computer security encompasses measures to protect systems from threats, ensuring confidentiality, integrity, and availability of data. Computer network security focuses on safeguarding networks from unauthorized access and attacks. Access control regulates who can view or use resources, while data security protects sensitive information from breaches. Computational trust ensures reliability in transactions and interactions, and computer security exploits are vulnerabilities that attackers leverage to compromise systems. Together, these elements safeguard digital environments.",
                    ". Internet protocols are standardized rules that govern data communication over the internet, ensuring devices can communicate effectively. Key examples include TCP (Transmission Control Protocol), which ensures reliable data transmission, and IP (Internet Protocol), which handles addressing and routing. Other protocols, like HTTP (for web traffic) and FTP (for file transfer), facilitate specific types of data exchange. Collectively, these protocols enable the seamless functioning of the internet and support diverse applications and services.",
                    ". Computer file systems are crucial components of operating systems that manage how data is stored, organized, and accessed on storage devices. They arrange files into directories, facilitate operations like creation and deletion, and manage permissions and metadata. Various file systems exist, such as NTFS (Windows), ext4 (Linux), and HFS+ (macOS), each designed for specific performance, reliability, and compatibility needs across different platforms.",
                    ". Distributed computing architecture involves a system of interconnected computers that collaboratively process data and tasks. It enables resource sharing and parallel processing across multiple machines, enhancing performance and scalability. Key components include clients, servers, and communication protocols that facilitate coordination and data exchange. Common examples are cloud computing and grid computing. This architecture is vital for handling large-scale applications, improving efficiency, and supporting fault tolerance in various domains, from scientific research to enterprise solutions.",
                    ". Web technology encompasses tools and protocols that facilitate the creation and interaction of web applications and services. Web software refers to applications designed to run on web servers, such as content management systems and e-commerce platforms. Web services are standardized methods for enabling communication between different software systems over the internet, typically using protocols like HTTP and XML or JSON for data exchange. Together, they underpin the functionality and connectivity of the modern web.",
                    ". Programming language topics encompass the study of languages used for software development, focusing on syntax, semantics, and implementation. Programming language theory investigates foundational concepts, including type systems, compilers, and language design. Programming language concepts cover key ideas like abstraction, encapsulation, and concurrency, shaping how languages are built and used. Programming language classification categorizes languages based on paradigms (e.g., procedural, functional, object-oriented), syntax, and application domains, aiding in understanding their strengths and weaknesses.",]                   
        elif args.target_data.lower() == 'instagram':
            classes = ['Normal Users', 'Commercial Users']
            c_descs = [" who typically shares personal moments and engages with friends and family, focusing on social connections and self-expression through photos and stories. Their primary goal is to enjoy and explore content that reflects their interests and lifestyle.", 
                    " who leverages the platform to promote products or services, utilizing targeted advertising and engaging content to reach potential customers. Their focus is on brand growth and customer interaction, often employing analytics to refine strategies and enhance reach."]
        elif args.target_data.lower() == 'history':
            classes = ['World', 'Americas', 'Asia', 'Military', 'Europe', 'Russia', 'Africa', 'Ancient Civilizations', 'Middle East', 'Historical Study & Educational Resources', 'Australia & Oceania', 'Arctic & Antarctica']
            c_descs = [" which explores global events and trends throughout history.",
                    " which delves into the rich history of North, Central, and South America.",
                    " which focuses on the diverse cultures and historical developments in Asia."
                    " which examines wars, conflicts, military strategy, and their impact on history.",
                    " which covers the complex history of the European continent.",
                    " which specifically studies the history of Russia, its empires, and its role in the world.",
                    " which explores the vast and diverse histories of African nations and cultures.",
                    " which investigates the origins, rise, and fall of early civilizations.",
                    " which examines the history of the Middle East, including its cultures and pivotal events.",
                    " which provides tools, guides, and resources for the study of history.",
                    " which focuses on the history of Australia, New Zealand, and Pacific Island nations.",
                    " which explores the history and significance of the polar regions."]
        elif args.target_data.lower() == 'photo':
            classes = ['Video Surveillance', 'Accessories', 'Binoculars & Scopes', 'Video', 'Lighting & Studio', 'Bags & Cases', 'Tripods & Monopods', 'Flashes', 'Digital Cameras', 'Film Photography', 'Lenses', 'Underwater Photography']
            c_descs = [" which enhance security and monitoring with advanced video surveillance systems. From IP cameras and security camera systems to video recorders and monitoring software, these solutions offer peace of mind and visual protection.",
                    " which enhance your photography experience with a vast array of accessories. Find memory cards, batteries, camera grips, remote controls, and more to ensure you're equipped for any shooting scenario.",
                    " which bring distant subjects into focus with high-quality binoculars and scopes. These optical instruments are perfect for birdwatching, nature observation, stargazing, and outdoor adventures.",
                    " which elevate your videography skills with professional video equipment. From high-quality camcorders and action cameras to gimbals, drones, and editing software, these tools empower you to create stunning visual content.",
                    " which take your photography to the next level with professional lighting and studio equipment. Explore continuous lighting, strobes, softboxes, reflectors, and backdrops for achieving perfect illumination and creative effects.",
                    " which protect your valuable photography equipment with durable and stylish bags and cases. Choose from backpacks, messenger bags, roller cases, and protective sleeves to keep your gear safe and organized.",
                    " which achieve steady and blur-free shots with sturdy tripods and monopods. These essential tools provide stability and versatility, enabling you to capture sharp images and smooth video footage.",
                    " which illuminate your subjects with precision and control using high-performance camera flashes. From compact on-camera flashes to powerful studio strobes, these lighting solutions enhance your photography in any environment.",
                    " which upgrade your photography game with advanced digital cameras. Explore a diverse selection of DSLRs, mirrorless cameras, point-and-shoots, and camera bundles to capture precious moments with stunning clarity.",
                    " which rediscover the art of traditional film photography. Explore a wide range of film cameras, film stocks, darkroom equipment, and accessories for capturing timeless images with a vintage aesthetic.",
                    " which expand your creative possibilities with a diverse range of camera lenses. From wide-angle to telephoto, prime to zoom, these lenses offer versatility and precision for capturing breathtaking images.",
                    " which dive into the captivating world of underwater photography. Discover specialized waterproof cameras, housings, lenses, and accessories designed to withstand the aquatic environment and capture stunning marine life."]
        elif args.target_data.lower() == 'computer':
            classes = ['Computer Accessories & Peripherals', 'Tablet Accessories', 'Laptop Accessories', 'Computers & Tablets', 'Computer Components', 'Data Storage', 'Networking Products', 'Monitors', 'Servers', 'Tablet Replacement Parts']
            c_descs = [" which upgrade your computing setup with essential accessories and peripherals. Explore keyboards, mice, webcams, printers, scanners, and more to boost productivity, enhance ergonomics, and streamline your workflow.",
                    " which enhance the functionality and protection of your tablet with a variety of accessories. Cases, covers, stylus pens, keyboards, stands, and screen protectors ensure a personalized and secure tablet experience.",
                    " which enhance your laptop experience with a wide range of accessories. From protective cases and sleeves to external hard drives, wireless mice, and portable speakers, these accessories provide convenience, functionality, and personalization for your laptop.",
                    " which explore a wide range of computers and tablets to suit your personal or professional requirements. From powerful desktop PCs and portable laptops to versatile 2-in-1 devices and sleek tablets, these cutting-edge machines offer performance, mobility, and functionality.",
                    " which build or upgrade your custom computer with high-performance components. From processors and motherboards to graphics cards, RAM, and cooling solutions, these components offer power, speed, and customization for your unique computing needs.",
                    " which keeps your digital files safe and organized with reliable data storage solutions. Choose from external hard drives, solid-state drives, network-attached storage, and cloud storage options to securely store and backup your valuable data.",
                    " which establish seamless connectivity with networking products designed for home or office use. Routers, modems, switches, access points, and networking cables ensure reliable internet access, file sharing, and secure network connections.",
                    " which upgrade your visual experience with high-quality monitors suitable for various computing needs. From sleek and stylish displays for everyday use to specialized monitors for gaming, graphic design, or professional applications, these monitors deliver stunning visuals and optimal viewing angles.",
                    " which power your business or organization with reliable and high-performance servers. From entry-level servers for small businesses to enterprise-grade solutions for data centers, these servers offer robust computing power, storage capacity, and advanced features.",
                    " which extend the life of your tablet by replacing worn-out or damaged components. Find official replacement parts like batteries, screens, cameras, and other essential components to restore your tablet's functionality."]

        test_idx = data.test_mask.nonzero().squeeze()
        
        if args.target_data.lower() == 'wikics':
            walk_step = 64
            test_dataset = SourceGraphDataset_Wiki(data, test_idx, walk_step, args.aggregation_times)
        elif args.target_data.lower() == 'instagram':
            walk_step = 64
            test_dataset = SourceGraphDataset_Wiki(data, test_idx, walk_step, args.aggregation_times)
        elif args.target_data.lower() == 'history':
            walk_step = 256
            test_dataset = SourceGraphDataset_Wiki(data, test_idx, walk_step, args.aggregation_times)
        elif args.target_data.lower() == 'photo':
            walk_step = 256
            test_dataset = SourceGraphDataset_Wiki(data, test_idx, walk_step, args.aggregation_times)
        elif args.target_data.lower() == 'computer':
            walk_step = 256
            test_dataset = SourceGraphDataset_Wiki(data, test_idx, walk_step, args.aggregation_times)
        else:
            test_dataset = SourceGraphDataset(data, test_idx, args.aggregation_times)
        
        loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        if not args.des:
            text_inputs = ["X "*args.text_prompt_len+c for c in classes]
        else:
            text_inputs = ["X "*args.text_prompt_len+c+des for c,des in zip(classes, c_descs)]

        in_dim = test_dataset[0].x.shape[1]
        model = CoOp(args, text_inputs, clip_model, in_dim, args.token_num, test_dataset.path_length, device)
        model = model.to(device)

        # torch.save(model.state_dict(), f"g2p2_{args.target_data}.pth")

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
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(loader):
                graph = sample_batched.to(device)
                pred, _ = model.inference(graph)
                y = graph.y
                tune_pre = torch.sum(pred.argmax(dim=1) == y).item()
                correct += tune_pre

        acc = correct / data.test_mask.sum().item()
        print(f"{round(acc * 100, 3)}")

        torch.save(model.state_dict(), f"g2p2_{args.target_data}_tta.pth")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument("--source_data", type=str, default="arxiv")
    parser.add_argument("--target_data", type=str, default="instagram")

    parser.add_argument('--aggregation_times', type=int, default=2, help='Aggregation times')
    parser.add_argument('--edge_coef', type=float, default=10)
    parser.add_argument('--gnn_input', type=int, default=384)
    parser.add_argument('--gnn_hid', type=int, default=384)
    parser.add_argument('--gnn_output', type=int, default=384)

    parser.add_argument('--context_length', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=384)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=12)
    parser.add_argument('--transformer_width', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=49408)

    parser.add_argument('--batch_size', type=int, default=2048)

    parser.add_argument('--des', type=bool, default=False)

    parser.add_argument('--tta', type=bool, default=True)
    parser.add_argument('--tta_step', type=int, default=20)

    # parser.add_argument('--graph_prompt_lr', type=float, default=0.1)
    # parser.add_argument('--text_prompt_lr', type=float, default=0.001)
    parser.add_argument('--prompt_wd', type=float, default=0.0)

    parser.add_argument('--token_num', type=int, default=2)
    parser.add_argument('--text_prompt_len', type=int, default=2)

    parser.add_argument('--drop_e', type=float, default=0.5)
    parser.add_argument('--temp', type=float, default=0.5)

    args = parser.parse_args()

    config = load_config(f"./config/{args.source_data}/{args.target_data}.yaml")
    for key, value in config.items():
        setattr(args, key, value)

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    FType = torch.FloatTensor
    LType = torch.LongTensor

    seed = 1
    main(args)
