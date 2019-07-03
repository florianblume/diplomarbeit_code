import numpy as np
import os   

def normal_dense(x, m_=0.0, std_=None):
    import torch
    tmp = -((x-m_)**2)
    tmp = tmp / (2.0*std_*std_)
    tmp = torch.exp(tmp)
    tmp = tmp / torch.sqrt((2.0*np.pi)*std_*std_)
    return tmp


def img_to_tensor(img):
    img.shape = (img.shape[0], img.shape[1], 1)
    imgOut = torchvision.transforms.functional.to_tensor(img)
    return imgOut


def get_stratified_coords2D(box_size, shape):
    coords = []
    box_count_y = int(np.ceil(shape[0] / box_size))
    box_count_x = int(np.ceil(shape[1] / box_size))
    for i in range(box_count_y):
        for j in range(box_count_x):
            y = np.random.randint(0, box_size)
            x = np.random.randint(0, box_size)
            y = int(i * box_size + y)
            x = int(j * box_size + x)
            if (y < shape[0] and x < shape[1]):
                coords.append((y, x))
    return coords


def joint_shuffle(inA, inB):
    """Shuffles both numpy arrays consistently.
    This is useful to shuffle raw and ground-truth
    data together. Both arrays need to have the same
    dimensions.

    Arguments:
        inA {np.array} -- first array to shuffle
        inB {np.array} -- second array to shuffle

    Returns:
        np.array, np.array -- the shuffled arrays
    """
    assert inA.shape[0] == inB.shape[0]
    indices = np.arange(inA.shape[0])
    np.random.shuffle(indices)
    return inA[indices], inB[indices]


def random_crop_fri(data, width, height, box_size, dataClean=None, counter=None):

    if counter is None or counter >= data.shape[0]:
        counter = 0
        if dataClean is not None:
            data, dataClean = joint_shuffle(data, dataClean)
        else:
            np.random.shuffle(data)
    index = counter
    counter += 1

    img = data[index]
    if dataClean is not None:
        imgClean = dataClean[index]
    else:
        imgClean = None
    imgOut, imgOutC, mask = random_crop(
        img, width, height, box_size, imgClean=imgClean)
    return imgOut, imgOutC, mask, counter


def random_crop(img, width, height, box_size, imgClean=None, hotPixels=64):
    assert img.shape[0] >= height
    assert img.shape[1] >= width

    n2v = False
    if imgClean is None:
        imgClean = img.copy()
        n2v = True

    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)

    imgOut = img[y:y+height, x:x+width].copy()
    imgOutC = imgClean[y:y+height, x:x+width].copy()
    mask = np.zeros(imgOut.shape)
    maxA = imgOut.shape[1]-1
    maxB = imgOut.shape[0]-1

    if n2v:
        # Noise2Void training, i.e. no clean targets
        hotPixels = get_stratified_coords2D(box_size, imgOut.shape)

        for p in hotPixels:
            a, b = p[1], p[0]

            roiMinA = max(a-2, 0)
            roiMaxA = min(a+3, maxA)
            roiMinB = max(b-2, 0)
            roiMaxB = min(b+3, maxB)
            roi = imgOut[roiMinB:roiMaxB, roiMinA:roiMaxA]
          #  print(roi.shape,b ,a)
         #   print(b-2,b+3 ,a-2,a+3)
            a_ = 2
            b_ = 2
            while a_ == 2 and b_ == 2:
                a_ = np.random.randint(0, roi.shape[1])
                b_ = np.random.randint(0, roi.shape[0])

            repl = roi[b_, a_]
            imgOut[b, a] = repl
            mask[b, a] = 1.0
    else:
        # Noise2Clean
        mask[:] = 1.0

    rot = np.random.randint(0, 4)
    imgOut = np.array(np.rot90(imgOut, rot))
    imgOutC = np.array(np.rot90(imgOutC, rot))
    mask = np.array(np.rot90(mask, rot))
    if np.random.choice((True, False)):
        imgOut = np.array(np.flip(imgOut))
        imgOutC = np.array(np.flip(imgOutC))
        mask = np.array(np.flip(mask))

    return imgOut, imgOutC, mask


def PSNR(gt, pred, range_=255.0):
    mse = np.mean((gt - pred)**2)
    return 20 * np.log10((range_)/np.sqrt(mse))


def average_PSNR_of_stored_numpy_array(path_to_array, path_to_gt_array):
    data = np.load(path_to_array)
    gt = np.load(path_to_gt_array)
    img_factor = int(data.shape[0]/gt.shape[0])
    gt = np.repeat(gt, img_factor, axis=0)
    psnr = 0
    for index in range(data.shape[0]):
        psnr += PSNR(gt[index], data[index])
    psnr /= data.shape[0]
    return psnr


def normalize(img, mean, std):
    zero_mean = img - mean
    return zero_mean/std


def denormalize(x, mean, std):
    return x*std + mean


def load_config(config_path):
    import yaml
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
        return config
    raise "Config could not be loaded."


def add_shot_noise_to_images(images, defect_ratio):
    result = []
    for image in images:
        out = np.copy(image)
        num_salt = np.ceil(image.size * defect_ratio)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = np.random.randint(255, size=len(coords[0]))
        result.append(out)
    return result


def numpy_array_to_images(array_path, output_path):
    """To be able to visualize the numpy arrays that the networks handle this
    function can convert a dedicated numpy array file to images.

    Arguments:
        array_path {[type]} -- [description]
        output_path {[type]} -- [description]
    """
    import matplotlib.pyplot as plt
    data = np.load(array_path)
    for i, image in enumerate(data):
        plt.imsave(os.path.join(output_path, str(i).zfill(4) + '.png'), image)


def add_gauss_noise_to_images(images, mean, std):
    result = []
    for image in images:
        noisy = image + np.random.normal(mean, std, image.shape)
        noisy = np.clip(noisy, 0, 255)
        result.append(noisy)
    return result


def merge_two_npy_datasets(dataset_path_1, dataset_path_2, output_path):
    """Merges two datasets stored as numpy arrays. Only matching entries are
    matched. E.g. if at both paths there is a ./gt/gt_0.npy then this will be
    concatenated.

    Arguments:
        dataset_path_1 {str} -- path to first dataset
        dataset_path_2 {str} -- path to second dataset
        output_path {str} -- where to store the results, same folder structure
                             as the original datasets
    """
    import glob

    wd = os.getcwd()
    os.chdir(dataset_path_1)
    files_1 = glob.glob('**/*.npy', recursive=True)
    os.chdir(wd)
    os.chdir(dataset_path_1)
    files_2 = glob.glob('**/*.npy', recursive=True)
    os.chdir(wd)
    differences = set(files_1).symmetric_difference(set(files_2))
    if len(differences) > 0:
        print('There were unmatched entries in the dataset. Only processing similarities.')
        print(differences)
    files_1 = set(files_1)
    files_1 -= differences
    files_1 = list(files_1)
    for _file in files_1:
        print('Processing {}.'.format(_file))
        first = np.load(os.path.join(dataset_path_1, _file))
        second = np.load(os.path.join(dataset_path_2, _file))
        result = np.concatenate([first, second], axis=0)
        output_file = os.path.join(output_path, _file)
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_path, _file), result)
