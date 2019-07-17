import argparse
import importlib
import os
import sys

main_path = os.getcwd()
sys.path.append(os.path.join(main_path, 'src/models'))

def main(predictor, config):
    predictor_module = importlib.import_module('models.' + predictor)
    predictor = predictor_module.Predictor(config)
    predictor.predict()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the config.")
    parser.add_argument("--predictor", "-p", 
        help="The predictor to use [packapge.pyfile(without extension)].")
    args = parser.parse_args()
    main(args.predictor, args.config)