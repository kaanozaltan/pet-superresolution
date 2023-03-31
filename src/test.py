import os

import torch
from torchvision import transforms
from PIL import Image

from models import SRCNN, VDSR, EDSR


def test(model):
    model_name = model.get_name()
    print(f"Testing {model_name.upper()} model")
    model.load_state_dict(torch.load(f'pth/{model_name}.pth'))
    model.eval()

    for filename in os.listdir('dataset/test/lr'):
        input_img = Image.open(os.path.join('dataset/test/lr', filename).replace('\\', '/'))
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        input_tensor = transform(input_img)
        input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_tensor = output_tensor.squeeze(0).detach().numpy()
        output_tensor *= 255.0
        output_tensor = output_tensor.clip(0, 255)
        output_tensor = output_tensor.transpose(1, 2, 0).astype("uint8")

        output_img = Image.fromarray(output_tensor, mode='RGB')
        output_img.save(os.path.join('dataset/test/outputs', filename).replace('\\', '/'))


# test(model=SRCNN())
# test(model=VDSR())
test(model=EDSR())
