import torch
import networkx as nx
import numpy as np
import dgl
import random


def process_image(feature_extractor, model, image_batch, transform):
    device = model.device
    if transform is not None:
        image_batch = [transform(x) for x in image_batch]
    inputs = feature_extractor(image_batch, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.pooler_output


def build_map(dataset):
    graph_map = nx.DiGraph()

    # Add nodes
    for state in dataset.photo_names:
        u = int(state[5:])
        graph_map.add_node(u)

    # Add edges
    for state in dataset.photo_names:
        u = int(state[5:])
        for action in dataset.photo_names[state]:
            v = int(action[6:])
            graph_map.add_edge(u, v)
    return graph_map


def get_reward_table(graph_map):
    terminal_state = len(graph_map) - 1
    reward_table = dict(nx.single_target_shortest_path_length(graph_map, terminal_state))
    for node in reward_table:
        reward_table[node] *= -1
    reward_table = [x[1] for x in sorted(reward_table.items())]
    return reward_table


def set_random_seeds(seed_value=0, device='cpu'):
    '''source https://forums.fast.ai/t/solved-reproducibility-where-is-the-randomness-coming-in/31628/5'''
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    dgl.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False