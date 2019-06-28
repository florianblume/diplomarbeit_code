import argparse
import importlib

import util

def main(predictor, config):
    predictor_module = importlib.import_module(predictor)
    predictor = predictor_module.Predictor(config)
    predictor.predict()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the config.")
    parser.add_argument("--predictor", "-p", 
        help="The predictor to use [packapge.pyfile(without extension)].")
    args = parser.parse_args()
    config = util.load_config(args.config)
    main(args.predictor, config)