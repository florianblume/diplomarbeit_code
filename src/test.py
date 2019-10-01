import util
from models.baseline import UNet

config = util.load_config('../experiments/baseline/main_3/n2c/fish/raw/config.yml')
config['MEAN'] = 0
config['STD'] = 1
config['DEVICE'] = 'cpu'
config['DEPTH'] = 4

net = UNet(config)

print(util.compute_RF_numerical(net, (1, 1, 1024, 1024)))