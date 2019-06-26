from dataloader import DataLoader
import util


class Trainer(Object):

    def load_data(self, config):
        # data_c=np.concatenate((data.copy(),dataTest.copy()))
        loader = DataLoader(config['DATA_BASE_PATH'])
        # In case the ground truth data path was not set we pass '' to
        # the loader which returns None to us
        data_raw, data_gt = loader.load_training_data(
            config['DATA_TRAIN_RAW_PATH'], config.get('DATA_TRAIN_GT_PATH', ''))
        if data_gt is not None:
            data_raw, data_gt = util.joint_shuffle(data_raw, data_gt)
            # If loaded, the network is trained using clean targets, otherwise it performs N2V
            data_train_gt = data_gt.copy()
            data_val_gt = data_gt.copy()
        else:
            data_train_gt = None
            data_val_gt = None

        data_train = data_raw.copy()
        data_val = data_raw.copy()
        return loader, data_raw, data_gt, data_train, data_train_gt, data_val, data_val_gt

    def create_checkpoint(model, optimizer, **kwargs):
        result = {'model_state_dict': model.state_dict(),
                'optimizier_state_dict': optimizer.state_dict()}
        result = {result, **kwargs}
        return result