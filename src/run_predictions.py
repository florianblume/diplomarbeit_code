import glob

import predict_model

with open('log.txt', 'w+') as log:
    for config in glob.glob('experiments/probabilistic/**/*.yml', recursive=True):
        try:
            predict_model.main(config)
        except Exception as e:
            print(str(e))
            log.write(config)
            log.write('\n')
            log.write(str(e))
            log.write('\n')
            log.write('\n')