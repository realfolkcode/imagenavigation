import torch

def process_image(feature_extractor, model, image):
    device = model.device
    inputs = feature_extractor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.pooler_output