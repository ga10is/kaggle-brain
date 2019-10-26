import os
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import pydicom
import cv2
import numpy as np

WINDOW_LIST = [
    (40, 80),
    (80, 200),
    (40, 380)
]


def get_first_of_dicom_field_as_int(x):
    # get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def get_windowing(data):
    dicom_fields = [data[('0028', '1050')].value,  # window center
                    data[('0028', '1051')].value,  # window width
                    data[('0028', '1052')].value,  # intercept
                    data[('0028', '1053')].value]  # slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def window_image(img, window_center, window_width, intercept, slope):
    img = (img * slope + intercept)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max

    img_float32 = img.astype(np.float32)
    # normalize image between 0.0 to 255.0
    img_float32 = (img_float32 - img_min) / (img_max - img_min) * 255
    img_uint8 = img_float32.astype(np.uint8)
    return img_uint8


def download_dicom(file_path, out_dir):
    try:
        id, _ = os.path.splitext(os.path.basename(file_path))
        out_filename = '%s.jpg' % id
        out_file_path = os.path.join(out_dir, out_filename)

        if os.path.exists(out_file_path):
            return True

        data = pydicom.read_file(file_path)
        image = data.pixel_array
        window_center, window_width, intercept, slope = get_windowing(data)

        windowed_images = []
        for window_center, window_width in WINDOW_LIST:
            image_windowed = window_image(
                image, window_center, window_width, intercept, slope)
            windowed_images.append(image_windowed)
        windowed_images = np.stack(windowed_images, axis=2)

        cv2.imwrite(out_file_path, windowed_images)

        return True
    except Exception as e:
        print('Could not write file: %s' % file_path)
        return False
        # raise


def download_wrapper(params):
    return download_dicom(*params)


def multi_download():
    dir_path = './data/raw/stage_1_test_images/'
    file_names = os.listdir(dir_path)
    out_dir = './data/data_512_v2/stage_1_test_images_v2'
    arg_list = [(os.path.join(dir_path, file_name), out_dir)
                for file_name in file_names]

    n_processes = multiprocessing.cpu_count()
    # ctx = multiprocessing.get_context('spawn')
    # with ctx.Pool(...)
    with Pool(processes=n_processes) as pool:

        imap_iter = pool.imap_unordered(download_wrapper, arg_list)
        results = list(tqdm(imap_iter, total=len(arg_list)))

    success = sum(1 for result in results if result)
    print('success: %d/%d' % (success, len(results)))


if __name__ == '__main__':
    # file_paths = os.listdir('./data/raw/stage_1_test_images/')
    # print(len(file_paths))
    multi_download()
    # file_path = './data/raw/stage_1_train_images/ID_0007d9aba.dcm'
    # download_dicom(file_path, out_dir='./')
