import torch

from transfer_net import TransferNet
from PIL import Image
from torchvision import transforms


class Stylizer(object):
    def __init__(self, trained_model_path):
        self._transfer_net = TransferNet()
        self._transfer_net.load_state_dict(torch.load(trained_model_path))
        self._transfer_net.eval()

    def stylize(self, content_image_path, resize=(256, 256), save=False, filename='outputs/images/default.jpg'):
        content_image = Image.open(content_image_path)
        transform = None
        if resize is None:
            transform = transforms.ToTensor()
        else:
            transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
        content_image = transform(content_image).unsqueeze(0)
        output = self._transfer_net(content_image).detach().squeeze()
        if save:
            transforms.ToPILImage()(output).save(filename)
        return output