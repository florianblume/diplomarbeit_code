import glob

import predict_model

old_log = open('log.txt', 'r')
lines = old_log.readlines()
old_log.close()

with open('log.txt', 'w+') as log:
    for line in lines:
        if '.yml' not in line:
            continue
        try:
            predict_model.main(line[:-1])
        except Exception as e:
            print(str(e))
            log.write(line)
            log.write('\n')
            log.write(str(e))
            log.write('\n')
            log.write('\n')