import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib
import json
import os

from models import abstract_predictor

class Predictor(abstract_predictor.AbstractPredictor):

    def _predict(self, image):
        image = self.net.predict(image, self.ps, self.overlap)
        return image