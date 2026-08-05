import os

import torch
from PIL import Image
from torchvision import transforms

from models import SRCNN, DeepSR


def generate_outputs(model):
    model_name = model.get_name()
    print(f"Performing inference for {model_name.upper()} model")
    model.load_state_dict(torch.load(f"models/{model_name}.pt"))
    model.eval()

    if not os.path.exists("dataset/inference/outputs"):
        os.makedirs("dataset/inference/outputs")

    for filename in os.listdir("dataset/inference/lr"):
        input_img = Image.open(os.path.join("dataset/inference/lr", filename))
        transform = transforms.ToTensor()
        input = transform(input_img).unsqueeze(0)

        with torch.no_grad():
            output = model(input)

        output = output.squeeze(0).detach().numpy()
        output *= 255.0
        output = output.clip(0, 255).astype("uint8")

        output_img = Image.fromarray(output[0], mode="L")
        output_img.save(os.path.join("dataset/inference/outputs", filename))


if __name__ == "__main__":
    generate_outputs(model=DeepSR())
