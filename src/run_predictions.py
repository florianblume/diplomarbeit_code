import glob
import os

import util
import predict_model

with open('log.txt', 'w+') as log:
    for config in glob.glob('experiments_depth_4/**/*.yml', recursive=True):
        loaded_config = util.load_config(config)
        dirname = os.path.dirname(config)
        path = os.path.join(dirname, loaded_config['PRED_OUTPUT_PATH'])
        if not os.path.exists(path) or len(os.listdir(path)) == 0:
            try:
                #print(config)
                predict_model.main(config)
            except Exception as e:
                print(str(e))
                log.write(config)
                log.write('\n')
                log.write(str(e))
                log.write('\n')
                log.write('\n')
