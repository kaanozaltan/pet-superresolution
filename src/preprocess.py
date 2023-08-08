import os

from PIL import Image
import pydicom
import numpy as np


def resize_all(src_path, dst_path, new_width, new_height):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for filename in os.listdir(src_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            img_path = os.path.join(src_path, filename).replace('\\', '/')
            img = Image.open(img_path)
            resized_img = img.resize((new_width, new_height))
            resized_img.save(os.path.join(dst_path, filename).replace('\\', '/'))


def read_dcm(file_path):
    dcm = pydicom.dcmread(file_path)
    pixel_array = dcm.pixel_array

    bits_stored = dcm.BitsStored
    bits_allocated = dcm.BitsAllocated
    bit_depth = bits_stored if bits_allocated == bits_stored else bits_allocated

    min_intensity = np.min(pixel_array)
    max_intensity = np.max(pixel_array)

    resolution = dcm.PixelSpacing

    num_rows = dcm.Rows
    num_columns = dcm.Columns

    orientation = dcm.ImageOrientationPatient

    print(f"Bit depth: {bit_depth} bits")
    print(f"Pixel intensity range: [{min_intensity}, {max_intensity}]")
    print(f"Pixel resolution: {resolution[0]}x{resolution[1]} mm")
    print(f"Dimensions: {num_rows}x{num_columns} px")
    print(f"Orientation: {orientation}")


def convert_all(src_path, dst_path):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for filename in os.listdir(src_path):
        if filename.endswith('.dcm'):
            output_filename = os.path.splitext(filename)[0] + '.jpg'
            dcm_path = os.path.join(src_path, filename).replace('\\', '/')
            dcm = pydicom.dcmread(dcm_path)
            pixel_array = dcm.pixel_array
            pixel_array = (pixel_array - pixel_array.min()) * (256 / (pixel_array.max() - pixel_array.min()))
            pixel_array = pixel_array.astype(np.uint8)
            img = Image.fromarray(pixel_array)
            img.save(os.path.join(dst_path, output_filename).replace('\\', '/'))    


# w, h = 256, 256

# resize_all('../dataset/train/original', '../dataset/train/hr', w, h)
# resize_all('../dataset/inference/original', '../dataset/inference/hr', w, h)

# resize_all('../dataset/train/original', '../dataset/train/lr', 32, 32)
# resize_all('../dataset/inference/original', '../dataset/inference/lr', 32, 32)

# resize_all('../dataset/train/lr', '../dataset/train/lr', w, h)
# resize_all('../dataset/inference/lr', '../dataset/inference/lr', w, h)

# path = 'C:/Users/Kaan/Desktop/ATLATEP_P21/MAC_FDG30min_Sharp/ATLAPTEPP21_AT.PT.TEP_IRM_CERVEAU.4.49.2023.04.06.17.13.12.179.69843961.dcm'
# dcm = pydicom.dcmread(path)
# pixel_array = dcm.pixel_array
# print("shape", pixel_array.shape)
# normalized_array = (pixel_array - pixel_array.min()) * (256 / (pixel_array.max() - pixel_array.min()))
# normalized_array = normalized_array.astype(np.uint8)
# # normalized_array = np.repeat(normalized_array, 3, axis=0)
# # normalized_array = np.repeat(normalized_array[:, :, np.newaxis], 3, axis=2)
# print("shape", normalized_array.shape)
# img = Image.fromarray(normalized_array)
# img.save('img.jpg')

convert_all('D:/dataset/train/original', 'D:/dataset/train/hr')
resize_all('D:/dataset/train/hr', 'D:/dataset/train/lr', 48, 48)
resize_all('D:/dataset/train/lr', 'D:/dataset/train/lr', 384, 384)

convert_all('D:/dataset/inference/original', 'D:/dataset/inference/hr')
#resize_all('../dataset/inference/hr', '../dataset/inference/lr', 48, 48)
#resize_all('../dataset/inference/lr', '../dataset/inference/lr', 384, 384)

#read_dcm('D:/siemens/NAMER_TP3/MEVI29839_02-06-2022/PI.1.3.12.2.1107.5.6.1.1592.30110122060811215189200000306')
#read_dcm('C:/Users/Kaan/Desktop/dcm_test/PI.1.3.12.2.1107.5.6.1.1592.30110122060811215189200000191')
