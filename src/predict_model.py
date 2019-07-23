import argparse

import util

def main(config_path):
    config = util.load_config(config_path)
    predictor = util.load_trainer_or_predictor('Predictor', config, config_path)
    predictor.predict()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the config.")
    args = parser.parse_args()
    main(args.config)
    