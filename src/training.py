import argparse
import importlib

import util

def main(trainer, config):
    trainer_module = importlib.import_module(trainer)
    trainer = trainer_module.Trainer(config)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the config.")
    parser.add_argument("--trainer", "-t", help="The trainer to use [packapge/pyfile].")
    args = parser.parse_args()
    config = util.load_config(args.config)
    main(args.trainer, config)