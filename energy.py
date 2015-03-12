#!/usr/bin/env python

__author__ = 'Aleksandar Gyorev'
__email__  = 'a.gyorev@jacobs-university.de'

import os
import cv2
import sys
import shutil
import random
import numpy as np

import eigenfaces as EF

"""
Iteratively setting different energy threshold to be used for the Eigenfaces
and recording the accuracy in a data text file, that is to be plotted with the
gnuplot script (./plot_energy.gpi).
"""
if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print 'Usage: python2.7 energy.py <att faces dir>'
        sys.exit(1)

    energy_values = range(5,101)[0:101:5]                                       # we consider all energies multiples of 5
    #energy_values = [85 85 85 85 85]                                            # run 5 times with 85% energy

    f = open('energy.dat', 'w')
    for energy_value in energy_values:
        efaces = EF.Eigenfaces(str(sys.argv[1]), float(energy_value / 100.0))
        efaces.evaluate()
        f.write('%d %.6lf\n' % (energy_value, efaces.accuracy))

    f.close()
