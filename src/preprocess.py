import os

from PIL import Image
import pydicom
import numpy as np


def resize_all(src_path, dst_path, new_width, new_height, resample=Image.BICUBIC):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for filename in os.listdir(src_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            img_path = os.path.join(src_path, filename).replace('\\', '/')
            img = Image.open(img_path)
            resized_img = img.resize((new_width, new_height), resample=resample)
            resized_img.save(os.path.join(dst_path, filename).replace('\\', '/'))


def print_metadata(file_path):
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
            output_filename = os.path.splitext(filename)[0] + '.png'
            dcm_path = os.path.join(src_path, filename).replace('\\', '/')
            dcm = pydicom.dcmread(dcm_path)
            pixel_array = dcm.pixel_array
            pixel_array = (pixel_array - pixel_array.min()) * (256 / (pixel_array.max() - pixel_array.min()))
            pixel_array = pixel_array.astype(np.uint8)
            img = Image.fromarray(pixel_array)
            img.save(os.path.join(dst_path, output_filename).replace('\\', '/'))


convert_all('../dataset_dcm/train', '../dataset/train/hr')
convert_all('../dataset_dcm/inference', '../dataset/inference/hr')

w_hr, h_hr = 384, 384
w_lr, h_lr = 48, 48

resize_all('../dataset/train/hr', '../dataset/train/lr', w_lr, h_lr)
resize_all('../dataset/inference/hr', '../dataset/inference/lr', w_lr, h_lr)

resize_all('../dataset/train/lr', '../dataset/train/lr', w_hr, h_hr)
resize_all('../dataset/inference/lr', '../dataset/inference/lr', w_hr, h_hr)
