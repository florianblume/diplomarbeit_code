import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib
import json
import os

import predictor

class Predictor(predictor.Predictor):

    def _predict(self, image):
        image = self.net.predict(image, self.ps, self.overlap)
        return image