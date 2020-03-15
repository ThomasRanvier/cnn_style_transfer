import matplotlib.pyplot as plt
import torch

def display_image(image, title):
    plt.figure()
    plt.axis("off")
    plt.imshow(image.permute(1, 2, 0))
    plt.title(title)
    plt.show()

def gram_matrix(y):
    (b, c, h, w) = y.size()
    features = y.view(b, c, w * h)
    gram = torch.bmm(features, features.transpose(1, 2)) / (c * h * w)
    return gram

def normalize_batch(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std