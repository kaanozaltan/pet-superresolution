from PIL import Image
import numpy as np
from torchvision import transforms

path = '../dataset_pet/train/hr/ATLATEPP1_ATLA.PT.TEP_IRM_CERVEAU.4.30.2023.04.12.16.31.59.773.69281413.jpg'
image = Image.open(path)
image_array = np.array(image)
print(image_array.shape)

transform = transforms.ToTensor()
print(transform(image).shape)