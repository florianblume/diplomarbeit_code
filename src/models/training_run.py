class TrainingRun():

    def __init__(self):
        self.experiment_base_path = None

        self.epochs = 0
        self.virtual_batch_size = 0
        self.steps_per_epoch = 0
        self.patch_size = 0
        self.overlap = 0
        self.write_tensorboard_data = False

        self.current_step = 0
        self.running_loss = 0.0

        self.train_loss = 0.0
        self.avg_train_loss = 0.0
        self.train_hist = []
        self.train_losses = []

        self.val_loss = 0.0
        self.avg_val_loss = 0.0
        self.val_hist = []
        self.val_losses = []
        self.val_counter = 0

    @staticmethod
    def from_config(config, config_path):
        run = TrainingRun()

        run.experiment_base_path = config.get('EXPERIMENT_BASE_PATH',
                                                config_path)
        if run.experiment_base_path == "":
            run.experiment_base_path = config_path

        run.epochs = config['EPOCHS']
        run.virtual_batch_size = config['VIRTUAL_BATCH_SIZE']
        run.steps_per_epoch = config['STEPS_PER_EPOCH']
        run.patch_size = config['PRED_PATCH_SIZE']
        run.overlap = config['OVERLAP']
        run.write_tensorboard_data = config['WRITE_TENSORBOARD_DATA']
        return run

    def load_checkpoint(self, checkpoint):
        #TODO set data according to checkpoint
        pass

    def checkpoint(self):
        #TODO create checkpoint dict
        pass
