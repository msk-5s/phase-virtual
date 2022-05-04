# SPDX-License-Identifier: MIT

"""
This script plots the a virtual voltage measurement.
"""

import matplotlib.pyplot as plt

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
    # Make the time series' stationary.
    #***********************************************************************************************
    order = 10
    cutoff = 0.06

    series_s = transform.filter_butterworth(
        data=series.reshape(-1, 1), cutoff=cutoff, order=order
    ).reshape(-1,)

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
    # Plot correlation matrices.
    #***********************************************************************************************
    (fig_vn, _) = plot_factory.make_voltage_series_plot(series=series)
    (fig_vns, _) = plot_factory.make_voltage_series_plot(series=series_s)

    fig_vn.suptitle("Unfiltered Load")
    fig_vns.suptitle("Filtered Load")

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
