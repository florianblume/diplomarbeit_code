import os
import sys
import shutil
import numpy as np
import tifffile as tif

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

import util

##### Script to create the datasets as they were used in the experiments.

def expand_cells_dataset(base_dir, identifier):
    raw_sub_folders = ['avg2', 'avg4', 'avg8', 'avg16', 'raw']
    raw_files = ['test_noisy.npy', 'training_big_raw.npy']
    raw_ouput_folders = ['test', 'train']
    gt_files = ['test_gt.npy', 'training_big_GT.npy']
    gauss_noises = [15, 30, 60]
    # Size of SimSim images, we create cut outs of fish and mouse for that
    simsim_shape = [256, 256]

    # Store to compute mean and std first
    mean_std_data = []

    for i, gt_file in enumerate(gt_files):
        mean_std_data.extend(np.load(os.path.join(base_dir, 'data/raw',
                                                  identifier, 'gt',
                                                  gt_file)))
    mean = np.mean(mean_std_data)
    std = np.std(mean_std_data)
    mean_std_data = None

    # Extract gt images
    for i, gt_file in enumerate(gt_files):
        data = np.load(os.path.join(base_dir,
                                    'data/raw',
                                    identifier,
                                    'gt',
                                    gt_file))
        data = util.normalize(data, mean, std)
        for j, image in enumerate(data):
            # Insert as many leading 0s as there are digits in the number of
            # images in the data, e.g. 001 instead of 1 for up to 999 images
            pretty_index = str(j).zfill(len(str(abs(data.shape[0]))))
            filename = '{}_{}_{}.tif'.format(identifier,
                                             'gt',
                                             pretty_index)
            output_dir = os.path.join(base_dir,
                                      'data/processed',
                                      identifier,
                                      'gt',
                                      raw_ouput_folders[i])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            cropped_output_dir = os.path.join(base_dir,
                                              'data/processed',
                                              identifier,
                                              'cropped',
                                              'gt',
                                              raw_ouput_folders[i])
            if not os.path.exists(cropped_output_dir):
                os.makedirs(cropped_output_dir)
            tif.imsave(os.path.join(output_dir, filename), image)
            x = int(simsim_shape[0] / 2)
            y = int(simsim_shape[1] / 2)
            cropped_image = image[y:y+simsim_shape[0], x:x+simsim_shape[1]]
            tif.imsave(os.path.join(cropped_output_dir, filename), cropped_image)

    # Extract raw images
    for raw_sub_folder in raw_sub_folders:
        for i, raw_file in enumerate(raw_files):
            data = np.load(os.path.join(base_dir,
                                        'data/raw',
                                        identifier,
                                        raw_sub_folder,
                                        raw_file))
            for j, image in enumerate(data):
                # Insert as many leading 0s as there are digits in the number of
                # images in the data, e.g. 001 instead of 1 for up to 999 images
                pretty_index = str(j).zfill(len(str(abs(data.shape[0]))))
                filename = '{}_{}_{}.tif'.format(identifier,
                                                 raw_sub_folder,
                                                 pretty_index)
                output_dir = os.path.join(base_dir,
                                          'data/processed',
                                          identifier,
                                          raw_sub_folder,
                                          raw_ouput_folders[i])
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                cropped_output_dir = os.path.join(base_dir,
                                                  'data/processed',
                                                  identifier,
                                                  'cropped',
                                                  raw_sub_folder,
                                                  raw_ouput_folders[i])
                if not os.path.exists(cropped_output_dir):
                    os.makedirs(cropped_output_dir)
                image = util.normalize(image, mean, std)
                tif.imsave(os.path.join(output_dir, filename), image)
                x = int(simsim_shape[0] / 2)
                y = int(simsim_shape[1] / 2)
                cropped_image = image[y:y+simsim_shape[0], x:x+simsim_shape[1]]
                tif.imsave(os.path.join(cropped_output_dir, filename), cropped_image)
            # In case that we are processing the normal raw images also
            # add the artificially noised version
            if raw_sub_folder == 'raw':
                for gauss_noise in gauss_noises:
                    data = util.add_gauss_noise_to_images(data, 0, gauss_noise)
                    gauss_str = 'gauss' + str(gauss_noise)
                    for j, image in enumerate(data):
                        pretty_index = str(j).zfill(len(str(abs(data.shape[0]))))
                        filename = '{}_{}_{}.tif'.format(identifier,
                                                         gauss_str,
                                                         pretty_index)
                        output_dir = os.path.join(base_dir,
                                                  'data/processed',
                                                  identifier,
                                                  gauss_str,
                                                  raw_ouput_folders[i])
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        cropped_output_dir = os.path.join(base_dir,
                                                  'data/processed',
                                                  identifier,
                                                  'cropped',
                                                  gauss_str,
                                                  raw_ouput_folders[i])
                        if not os.path.exists(cropped_output_dir):
                            os.makedirs(cropped_output_dir)
                        image = image.astype(np.uint8)
                        image = util.normalize(image, mean, std)
                        tif.imsave(os.path.join(output_dir, filename), image)
                        x = int(simsim_shape[0] / 2)
                        y = int(simsim_shape[1] / 2)
                        cropped_image = image[y:y+simsim_shape[0], x:x+simsim_shape[1]]
                        tif.imsave(os.path.join(cropped_output_dir, filename), cropped_image)
        
def expand_simsim_dataset(base_dir):
    train_raw = tif.imread(os.path.join(base_dir,
                                        'data/raw/simsim/camsim_ccd_phot300_rn8_bgrd0.tif'))
    train_gt = tif.imread(os.path.join(base_dir,
                                       'data/raw/simsim/noise_free_32b.tif'))
    assert train_raw.shape[0] == train_raw.shape[0]
    # Adjust value ranges of simsim, the dataset is shifted and scaled
    train_gt, train_raw, range_ = adjust_raw_and_scaled_shifted_gt(train_gt,
                                                                   train_raw)
    mean, std = np.mean(train_gt), np.std(train_gt)
    train_gt = util.normalize(train_gt, mean, std)
    train_raw = util.normalize(train_raw, mean, std)
    # Three orientations in SimSim that's why we need to split it into 3 parts
    factor = int(train_raw.shape[0] / 3)

    out_dir = 'data/processed/simsim'
    part_dirs = ['part1', 'part2', 'part3']

    # Also store all together
    out_dir_all_raw_train = os.path.join(out_dir, 'all/raw/train')
    out_dir_all_raw_test = os.path.join(out_dir, 'all/raw/test')
    out_dir_all_gt_train = os.path.join(out_dir, 'all/gt/train')
    out_dir_all_gt_test = os.path.join(out_dir, 'all/gt/test')
    os.makedirs(out_dir_all_raw_train)
    os.makedirs(out_dir_all_raw_test)
    os.makedirs(out_dir_all_gt_train)
    os.makedirs(out_dir_all_gt_test)

    # And store as triple
    out_dir_all_3_times_raw_train = os.path.join(out_dir, 'all_3_times/raw/train')
    out_dir_all_3_times_raw_test = os.path.join(out_dir, 'all_3_times/raw/test')
    out_dir_all_3_times_gt_train = os.path.join(out_dir, 'all_3_times/gt/train')
    out_dir_all_3_times_gt_test = os.path.join(out_dir, 'all_3_times/gt/test')
    os.makedirs(out_dir_all_3_times_raw_train)
    os.makedirs(out_dir_all_3_times_raw_test)
    os.makedirs(out_dir_all_3_times_gt_train)
    os.makedirs(out_dir_all_3_times_gt_test)

    for i, part_dir in enumerate(part_dirs):
        out_dir_raw_train = os.path.join(out_dir, part_dir, 'raw/train')
        out_dir_raw_test = os.path.join(out_dir, part_dir, 'raw/test')
        out_dir_gt_train = os.path.join(out_dir, part_dir, 'gt/train')
        out_dir_gt_test = os.path.join(out_dir, part_dir, 'gt/test')
        os.makedirs(out_dir_raw_train)
        os.makedirs(out_dir_raw_test)
        os.makedirs(out_dir_gt_train)
        os.makedirs(out_dir_gt_test)
        train_raw_part = train_raw[i * factor:(i + 1) * factor]
        # Rotate images by 90 degrees to obtain artificial test images
        test_raw_part = np.array([np.rot90(image, k=2) for image in train_raw_part])
        train_gt_part = train_gt[i * factor:(i + 1) * factor]
        test_gt_part = np.array([np.rot90(image, k=2) for image in train_gt_part])
        assert train_raw_part.shape[0] == train_gt_part.shape[0]\
               == test_raw_part.shape[0] == test_gt_part.shape[0]

        for j, raw in enumerate(train_raw_part):
            pretty_index = str(j).zfill(len(str(abs(train_raw_part.shape[0]))))
            filename = '{}_{}_{}.tif'.format('simsim',
                                             'raw',
                                             pretty_index)
            # Somehow the training images are loaded as float 64 but the
            # data is actually only float 32
            tif.imsave(os.path.join(out_dir_raw_train, filename), raw.astype(np.float32))
            tif.imsave(os.path.join(out_dir_raw_test, filename), test_raw_part[j].astype(np.float32))

            # To store all together
            pretty_index = str(j).zfill(len(str(abs(train_raw_part.shape[0]))))
            filename = '{}_{}_{}_{}_{}.tif'.format('simsim',
                                                   'part',
                                                   (i + 1),
                                                   'raw',
                                                   pretty_index)
            tif.imsave(os.path.join(out_dir_all_raw_train, filename), raw.astype(np.float32))
            tif.imsave(os.path.join(out_dir_all_raw_test, filename), test_raw_part[j].astype(np.float32))

            for k in range(3):
                # To store three copies of all
                pretty_index = str(j).zfill(len(str(abs(train_raw_part.shape[0]))))
                filename = '{}_{}_{}_{}_{}_{}.tif'.format('simsim',
                                                          (k + 1),
                                                          'part',
                                                          (i + 1),
                                                          'raw',
                                                          pretty_index)
                tif.imsave(os.path.join(out_dir_all_3_times_raw_train, filename), raw.astype(np.float32))
                tif.imsave(os.path.join(out_dir_all_3_times_raw_test, filename), test_raw_part[j].astype(np.float32))

        for j, gt in enumerate(train_gt_part):
            pretty_index = str(j).zfill(len(str(abs(train_gt_part.shape[0]))))
            filename = '{}_{}_{}.tif'.format('simsim',
                                             'gt',
                                             pretty_index)
            tif.imsave(os.path.join(out_dir_gt_train, filename), gt)
            tif.imsave(os.path.join(out_dir_gt_test, filename), test_gt_part[j])
            # To store all together
            pretty_index = str(j).zfill(len(str(abs(train_raw_part.shape[0]))))
            filename = '{}_{}_{}_{}_{}.tif'.format('simsim',
                                                   'part',
                                                   (i + 1),
                                                   'raw',
                                                   pretty_index)
            tif.imsave(os.path.join(out_dir_all_gt_train, filename), gt.astype(np.float32))
            tif.imsave(os.path.join(out_dir_all_gt_test, filename), test_gt_part[j].astype(np.float32))

            for k in range(3):
                pretty_index = str(j).zfill(len(str(abs(train_raw_part.shape[0]))))
                filename = '{}+{}_{}_{}_{}_{}.tif'.format('simsim',
                                                          (k + 1),
                                                          'part',
                                                          (i + 1),
                                                          'raw',
                                                          pretty_index)
                tif.imsave(os.path.join(out_dir_all_3_times_gt_train, filename), gt.astype(np.float32))
                tif.imsave(os.path.join(out_dir_all_3_times_gt_test, filename), test_gt_part[j].astype(np.float32))

def zero_mean(data):
    return data - np.mean(data)

def adjust_raw_and_scaled_shifted_gt(gts, raws):
    range_ = 255.0 / np.std(gts)
    zero_gts = zero_mean(gts) / np.std(gts)
    zero_gts = zero_mean(zero_gts)
    zero_raws = zero_mean(raws)
    std = np.sum(zero_gts * zero_raws) / (np.sum(zero_raws * zero_raws))
    zero_raws *= std
    return zero_gts, zero_raws, range_

if __name__ == '__main__':
    MAIN_PATH = os.getcwd()
    
    ### Remove old data
    print('Removing old generated data.')
    if os.path.exists('data/processed'):
        shutil.rmtree(os.path.join(MAIN_PATH, 'data/processed'))

    print('Processing cells datasets.')
    print('\t Expanding fish dataset.')
    expand_cells_dataset(MAIN_PATH, 'fish')
    print('\t Expanding mouse dataset.')
    expand_cells_dataset(MAIN_PATH, 'mouse')
    print('Processing SimSim dataset.')
    expand_simsim_dataset(MAIN_PATH)
    