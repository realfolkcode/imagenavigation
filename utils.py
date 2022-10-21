import torch
import networkx as nx


def process_image(feature_extractor, model, image_batch, transform):
    device = model.device
    if transform is not None:
        image_batch = [transform(x) for x in image_batch]
    inputs = feature_extractor(image_batch, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.pooler_output


def build_map(dataset):
    graph_map = nx.Graph()

    # Add nodes
    for state in dataset.photo_names:
        u = int(state[5:])
        graph_map.add_node(u)
    terminal_state = len(graph_map)
    graph_map.add_node(terminal_state)

    # Add edges
    for state in dataset.photo_names:
        u = int(state[5:])
        for action in dataset.photo_names[state]:
            if action == 'terminal':
                v = terminal_state
            else:
                v = int(action[6:])
            graph_map.add_edge(u, v)
    return graph_map