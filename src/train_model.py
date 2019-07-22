import argparse
import importlib

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