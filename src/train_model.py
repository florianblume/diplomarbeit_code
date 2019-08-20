import argparse
import logging
import os

import util

def main(config_path, debug):
    print('Training using config at \"{}\"'.format(os.path.abspath(config_path)))
    if debug:
        logging.basicConfig()
        logging.getLogger().setLevel(logging.DEBUG)
    config = util.load_config(config_path)
    trainer = util.load_trainer_or_predictor('Trainer', config, config_path)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the config.")
    parser.add_argument("--debug", "-d", action='store_true', help="Print debug info.")
    args = parser.parse_args()
    main(args.config, args.debug)
