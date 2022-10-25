import os
import torch
import numpy as np
import dgl
import networkx as nx
from dgl.data import DGLBuiltinDataset
from PIL import Image
from utils import process_image


class ObservationDataset(DGLBuiltinDataset):
    def __init__(self, data_dir, kind, feature_extractor, model, repeats=1, transform=None):
        self.data_dir = data_dir
        self.kind = kind
        self.feature_extractor = feature_extractor
        self.model = model
        self.photo_names = {}
        self.transform = transform
        self.train_test_ratio = 0.8
        self.repeats = repeats
        super().__init__(name='obs',
                         url=None)
    
    def process(self):
        # Traverse data_dir

        for state in os.listdir(self.data_dir):
            state_dict = {}
            state_dir = os.path.join(self.data_dir, state)
            if os.path.isdir(state_dir) and state.startswith('state'):
                for action in os.listdir(state_dir):
                    action_dir = os.path.join(state_dir, action)
                    if os.path.isdir(action_dir):
                        state_dict[action] = list(filter(lambda x : x.endswith('jpeg'), 
                                                         os.listdir(action_dir)))
                        dict_cutoff = int(self.train_test_ratio * len(state_dict[action]))
                        if self.kind == 'train':
                            state_dict[action] = state_dict[action][:dict_cutoff]
                        elif self.kind == 'test':
                            state_dict[action] = state_dict[action][dict_cutoff:]
                self.photo_names[state] = state_dict
    
    def __getitem__(self, idx):
        # Construct a complete graph
        idx = idx % len(self.photo_names)
        n = len(self.photo_names[f'state{idx}'])
        device = self.model.device
        graph = dgl.from_networkx(nx.complete_graph(n)).to(device)

        # Sample and load images
        actions = list(self.photo_names[f'state{idx}'])
        images = []
        for a in actions:
            sample = np.random.randint(0, len(self.photo_names[f'state{idx}'][a]))
            sample_path = os.path.join(self.data_dir,
                                       f'state{idx}',
                                       a,
                                       self.photo_names[f'state{idx}'][a][sample])
            images.append(Image.open(sample_path))
        
        # Process images
        feat = process_image(self.feature_extractor, self.model, images, self.transform)
        graph.ndata['x'] = feat

        return graph, idx
    
    def __len__(self):
        return self.repeats * len(self.photo_names)