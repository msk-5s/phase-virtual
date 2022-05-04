# SPDX-License-Identifier: MIT

"""
This script plots the spectrogram of the virtual transformer measurements through the filtering
process.
"""

import matplotlib.pyplot as plt
import numpy as np

import convert
import data_factory
import plot_factory
import transform

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals
    """
    The main function.
    """
    #***********************************************************************************************
    # Load the data.
    #***********************************************************************************************
    start = 0
    width = 96 * 7

    data = data_factory.make_data()[start:(start + width), :]
    xfmr_labels = data_factory.make_xfmr_labels()

    data_v = convert.load_to_virtual(data=data, xfmr_labels=xfmr_labels)
    series = data_v[:, 42]

    #***********************************************************************************************
    # Make the series stationary.
    #***********************************************************************************************
    frequencies = np.fft.fftfreq(len(series)) + 1e-6

    order = 10
    cutoff = 0.06

    series_dft = np.fft.fft(series)

    series_dft_s = np.fft.fft(
        transform.filter_butterworth(
            data=series.reshape(-1, 1), cutoff=cutoff, order=order
        ).reshape(-1,)
    )

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
    # Plot the virtual measurement spectrograms.
    #***********************************************************************************************
    (fig_dft, _) = plot_factory.make_periodogram_voltage_plot(
        series_dft=series_dft, frequencies=frequencies, cutoff=cutoff
    )

    (fig_dft_s, _) = plot_factory.make_periodogram_voltage_plot(
        series_dft=series_dft_s, frequencies=frequencies, cutoff=cutoff
    )

    fig_dft.suptitle("Unfiltered Series")
    fig_dft_s.suptitle("Filtered Series")

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
