import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def normal_dense(x, m_=0.0, std_=None):
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
    dataTmp = np.concatenate(
        (inA[..., np.newaxis], inB[..., np.newaxis]), axis=-1)
    np.random.shuffle(dataTmp)
    return dataTmp[..., 0], dataTmp[..., 1]


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
    imgOut, imgOutC, mask = random_crop(img, width, height, box_size, imgClean=imgClean)
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