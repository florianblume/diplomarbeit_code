import argparse

import util
from baseline import training as base
from probabilistic import training as prob
from q_learning import training as q
from reinforce import training as reinf

trainings = {'base' : base,
             'prob' : prob,
             'q'    : q,
             'reinf':reinf}

def main(net_type, config):
    training = trainings[net_type]
    training.train(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the config.")
    parser.add_argument("--net_type", "-t", help="The network type to execute [base, prob, reinf, q].")
    args = parser.parse_args()
    config = util.load_config(args.config)
    main(args.net_type, config)