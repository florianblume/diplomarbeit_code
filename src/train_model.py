import argparse
import importlib
import os
import sys

main_path = os.getcwd()
sys.path.append(os.path.join(main_path, 'src/models'))

def main(trainer, config):
    trainer_module = importlib.import_module('models.' + trainer)
    trainer = trainer_module.Trainer(config)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the config.")
    parser.add_argument("--trainer", "-t", 
        help="The trainer to use [packapge.pyfile(without extension)].")
    args = parser.parse_args()
    main(args.trainer, args.config)