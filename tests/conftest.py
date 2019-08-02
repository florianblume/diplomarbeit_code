import shutil
import tempfile
import pytest
import numpy as np
import tifffile as tif

tmp_raw_dir = tempfile.TemporaryDirectory()
tmp_gt_dir = tempfile.TemporaryDirectory()
tmp_raw_dir_2 = tempfile.TemporaryDirectory()
tmp_gt_dir_2 = tempfile.TemporaryDirectory()
tmp_single_gt_dir = tempfile.TemporaryDirectory()

def create_raw_images():
    raw_image_1 = np.arange(900).reshape(30, 30)
    raw_image_2 = np.arange(900, 2 * 900).reshape(30, 30)
    raw_image_3 = np.arange(2 * 900, 3 * 900).reshape(30, 30)
    return [raw_image_1, raw_image_2, raw_image_3]

def create_raw_images_2():
    raw_image_1 = np.arange(900).reshape(30, 30) - 1
    raw_image_2 = np.arange(900, 2 * 900).reshape(30, 30) - 1
    raw_image_3 = np.arange(2 * 900, 3 * 900).reshape(30, 30) - 1
    return [raw_image_1, raw_image_2, raw_image_3]

def create_gt_images():
    gt_image_1 = np.arange(3 * 900, 4 * 900).reshape(30, 30)
    gt_image_2 = np.arange(4 * 900, 5 * 900).reshape(30, 30)
    gt_image_3 = np.arange(5 * 900, 6 * 900).reshape(30, 30)
    return [gt_image_1, gt_image_2, gt_image_3]

def create_gt_images_2():
    gt_image_1 = np.arange(3 * 900, 4 * 900).reshape(30, 30) - 1
    gt_image_2 = np.arange(4 * 900, 5 * 900).reshape(30, 30) - 1
    gt_image_3 = np.arange(5 * 900, 6 * 900).reshape(30, 30) - 1
    return [gt_image_1, gt_image_2, gt_image_3]

def create_single_gt_image():
    gt_image = np.arange(7 * 900, 8 * 900).reshape(30, 30)
    return gt_image

@pytest.fixture(scope='session', autouse=True)
def setup_module():
    raw_images = create_raw_images()
    gt_images = create_gt_images()
    raw_images_2 = create_raw_images_2()
    gt_images_2 = create_gt_images_2()
    for i, raw_image in enumerate(raw_images):
        tif.imsave(tmp_raw_dir.name + '/raw{}.tif'.format(i), raw_image)
        tif.imsave(tmp_gt_dir.name + '/gt{}.tif'.format(i), gt_images[i])
        tif.imsave(tmp_raw_dir_2.name + '/raw{}.tif'.format(i), raw_images_2[i])
        tif.imsave(tmp_gt_dir_2.name + '/gt{}.tif'.format(i), gt_images_2[i])
    single_gt_image = create_single_gt_image()
    tif.imsave(tmp_single_gt_dir.name + '/gt.tif', single_gt_image)
    def fin():
        shutil.rmtree(tmp_raw_dir.name)
        shutil.rmtree(tmp_gt_dir.name)
        shutil.rmtree(tmp_raw_dir_2.name)
        shutil.rmtree(tmp_gt_dir_2.name)