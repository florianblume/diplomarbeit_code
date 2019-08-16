import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
                                mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class AbstractUNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """
    # * enforces named parameters to avoid accidential misconfigurations
    def __init__(self, config):
        super(AbstractUNet, self).__init__()
        up_mode = config['UP_MODE']
        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
        merge_mode = config['MERGE_MODE']
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")
        self.mean = config['MEAN']
        self.std = config['STD']
        self.in_channels = config['IN_CHANNELS']
        self.depth = config['DEPTH']
        self.patch_size = config['PRED_PATCH_SIZE']
        self.overlap = config['OVERLAP']
        self.augment_data = config['AUGMENT_DATA']
        self.start_filts = config['START_FILTS']
        self.up_mode = config['UP_MODE']
        self.merge_mode = config['MERGE_MODE']
        self.device = torch.device(config['DEVICE'])

        AbstractUNet._verify_depth_config(self.depth, self.patch_size, 
                                          self.overlap)

        self.down_convs = []
        self.up_convs = []

        self.noiseSTD = nn.Parameter(data=torch.log(torch.tensor(0.5)))
        self.noiseSTD.requires_grad = False

        # create the encoder pathway and add to a list
        for i in range(self.depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < self.depth-1 else False
            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)
            
        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(self.depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                             merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self._build_network_head(outs)

        self.reset_params()

        # Push network to respective device
        self.to(self.device)

    def _build_network_head(self, outs):
        raise NotImplementedError

    @staticmethod
    def _verify_depth_config(depth, patch_size, overlap):
        while depth > 0:
            patch_size = patch_size / 2.0
            depth -= 1
        assert patch_size.is_integer(), 'Patch size must be [depth]-many times'+\
                                        ' divisible by 2.'
        # No idea what to do with overlap...

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def loss_function(self, result):
        """The loss function of this network.
        
        Arguments:
            result {dict} -- dictionary containting at least the keys 'output',
                             'gt', 'mask'. Might contain more keys depending
                             one the implementation of the subclass.
        """
        raise NotImplementedError

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        """Runs this network on the given input x.
        
        Arguments:
            x {torch.Tensor} -- the input image to process, must be of shape
                                [batch_size, C, H, W]

        Returns:
            {torch.Tensor} -- the denoised image
            NOTE -- subclasses might return additional artifacts
        """
        raise NotImplementedError

    def training_predict(self, sample):
        """Performs a forward step during training.

        Arguments:
            train_data {dict} -- Dictionary containting at least the keys 'raw',
                                 'gt' and 'mask'. Subclasses might require
                                 additional keys.

        Returns:
            {dict} -- containing at leas the keys 'output', 'gt' and 'mask
        """
        raise NotImplementedError

    def predict(self, image):
        """Performs prediction on the whole image. Since we have certain memory
        constraints on the GPU it is important to define an appropriate patch
        size. The overlap defines the overlap between patches.
        
        Arguments:
            image {torch.Tensor} -- 2D image (grayscale or RGB), must be of
                                    shape [batch_size, C, H, W]

        Returns:
            {np.array} -- the denoised image of shape [H, W, C]
            NOTE -- subclasses might return additional artifacts
        """
        data = self._pre_process_predict(image)
        image_height = image.shape[-2]
        image_width = image.shape[-1]
        xmin = 0
        ymin = 0
        xmax = self.patch_size
        ymax = self.patch_size
        ovLeft = 0
        # Image is in [C, H, W] shape
        while (xmin < image_width):
            ovTop = 0
            while (ymin < image_height):
                # We do not receive anything from this method because it
                # modifies the data in-place
                self._process_patch(data, ymin, ymax, xmin, xmax, ovTop, ovLeft)
                ymin = ymin-self.overlap+self.patch_size
                ymax = ymin+self.patch_size
                ovTop = self.overlap//2
            ymin = 0
            ymax = self.patch_size
            xmin = xmin-self.overlap+self.patch_size
            xmax = xmin+self.patch_size
            ovLeft = self.overlap//2
        return self._post_process_predict(data)

    def _pre_process_predict(self, image):
        """Subclasses are supposed to prepare needed variables to perform
        the actual prediction.

        Returns:
            {dict} -- dictionary containing all the necessary variables
        """
        raise NotImplementedError

    def _process_patch(self, data, ymin, ymax, xmin, xmax, ovTop, ovLeft):
        """Performs prediction on the specified patch coordinates given the
        specified data dictionary. This method is for internal use of the
        prediction pipeline only.

        NOTE: Does not return anything because it modifies the data in-place.
        
        Arguments:
            data {dict} -- a dictionary with the data to work on
            ymin {int} -- y minimum of patch
            ymax {int} -- y maximum of patch
            xmin {int} -- x minimum of patch
            xmax {int} -- x maximum of patch
            ovTop {int} -- overlap top
            ovLef {int} -- overlap left
        """
        raise NotImplementedError

    def predict_patch(self, patch):
        """Performs prediction on the specified patch without any further
        processing.
        
        Arguments:
            patch {torch.Tensor} -- the patch to process of shape
                                    [batch_size, C, H, W]
        """
        raise NotImplementedError

    def _post_process_predict(self, result):
        """In this method subclasses can assemble the final results of the
        prediction.
        
        Arguments:
            result {dict} -- containing all the keys that the subclasses need

        Returns:
            {dict}        -- a dictionary containing the final results
        """
        raise NotImplementedError
