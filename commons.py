import io
from PIL import Image
import torchvision.transforms as transforms
import torch

from params import *
from models import GeneratorResNet

PATH_G = 'saved_models/generator_199.pth'


def get_generator_model():
    generator = GeneratorResNet(img_shape=img_shape,
                                res_blocks=residual_blocks,
                                c_dim=c_dim)
    generator.load_state_dict(torch.load(PATH_G, map_location=torch.device('cpu')))
    generator.eval()
    return generator


def get_tensor(img_bytes):
    my_transforms = transforms.Compose(
        [transforms.Resize((img_height, img_width), Image.BICUBIC),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    img = Image.open(io.BytesIO(img_bytes))
    return my_transforms(img).unsqueeze(0)
