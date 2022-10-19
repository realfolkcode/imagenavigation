import torch

def process_image(feature_extractor, model, image_batch, transform):
    device = model.device
    if transform is not None:
        image_batch = [transform(x) for x in image_batch]
    inputs = feature_extractor(image_batch, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.pooler_output