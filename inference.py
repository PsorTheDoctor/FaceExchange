import torch
import torchvision.transforms.functional as F
from torchvision.utils import save_image

from params import c_dim, label_changes
from commons import get_tensor, get_generator_model

generator = get_generator_model()


def generate_image(uploaded_img):
    prod_img = get_tensor(uploaded_img)
    prod_labels = torch.Tensor([[0, 0, 0, 0, 0]])

    img = prod_img[0]
    label = prod_labels[0]

    imgs = img.repeat(c_dim, 1, 1, 1)  # imgs shape = [5, 3, 128, 128]
    labels = label.repeat(c_dim, 1)  # labels shape = [5, 5]

    for sample_i, changes in enumerate(label_changes):
        for col, val in changes:
            labels[sample_i, col] = 1 - labels[sample_i, col] if val == -1 else val

    gen_imgs = generator(imgs, labels)

    # gen_imgs = torch.cat([x for x in gen_imgs.data], -1)
    # img_sample = torch.cat((img.data, gen_imgs), -1)
    # img_sample = F.to_pil_image(img_sample)

    save_image(gen_imgs, 'static/prod.png', normalize=True)
    return gen_imgs
