# SPDX-License-Identifier: MIT

"""
This script plots the spectrogram of the filter.
"""

import matplotlib.pyplot as plt
import numpy as np

import plot_factory

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals
    """
    The main function.
    """
    #***********************************************************************************************
    # Make the filters.
    #***********************************************************************************************
    width = 96 * 7

    frequencies = np.fft.fftfreq(width) + 1e-6

    # Butterworth parameters.
    cutoff = 0.06
    order = 10

    filter_dft = (1 / np.sqrt(1 + (cutoff / frequencies)**(2 * order))).astype(complex)

    #***********************************************************************************************
    # Set AeStHeTiCs.
    #***********************************************************************************************
    #fontsize = 60
    #plt.rc("axes", labelsize=fontsize)
    #plt.rc("axes", titlesize=fontsize)
    #plt.rc("figure", titlesize=fontsize)
    #plt.rc("xtick", labelsize=fontsize)
    #plt.rc("ytick", labelsize=fontsize)

    font = {"family": "monospace", "monospace": ["Times New Roman"]}
    plt.rc("font", **font)

    #***********************************************************************************************
    # Plot the filter spectrograms.
    #***********************************************************************************************
    (_, _) = plot_factory.make_periodogram_filter_plot(
        filter_dft=filter_dft, frequencies=frequencies
    )

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
