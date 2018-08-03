#from get_pulse import getPulseApp
#from lib.device import Camera
from lib.processors_noopenmdao import findFaceGetPulse
#from lib.interface import plotXY, imshow, waitKey, destroyWindow
#from liveplotter.plotter_impls import GeneralPlotter
import cv2, time
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

"""
Application to view the recorded HBR (heart beat rate) analyzed by PSD (power spectral density) and analyze the HBV (heart beat variability).
"""


def plotHBR(x, y, labels=[], title=[]):
    
    fig, ax = plt.subplots(1, 1)
    lines, = ax.plot(x, y)
    
    while True:
        x = x
        y = y
        lines.set_data(x, y)
        plt.draw()
        
        plt.pause(.1)