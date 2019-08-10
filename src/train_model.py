import argparse

import util

def main(config_path):
    config = util.load_config(config_path)
    trainer = util.load_trainer_or_predictor('Trainer', config, config_path)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the config.")
    args = parser.parse_args()
    main(args.config)
