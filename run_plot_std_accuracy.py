# SPDX-License-Identifier: BSD-3-Clause

"""
This script plots the phase identification accuracy standard deviation for the load and virtual
measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals, too-many-statements
    """
    The main function.
    """
    model_name = ["gmm", "hierarchical", "kmeans", "kmedoids", "spectral"][0]
    noise_name = ["gauss", "laplace"][0]
    noise_percent = "0p005"

    #***********************************************************************************************
    # Load the results.
    #***********************************************************************************************
    results = {dataset: [] for dataset in ["correlation", "voltage"]}
    results_v = {dataset: [] for dataset in ["correlation", "voltage"]}

    for day in range(1, 8):
        for dataset in results.keys():
            frame = pd.read_csv(
                f"results/result-{day}-day-{noise_percent}-noise.csv", index_col=0
            )

            results[dataset].append(
                frame[f"accuracy_{model_name}_{dataset}_{noise_name}"].std()
            )

            results_v[dataset].append(
                frame[f"accuracy_{model_name}_{dataset}_v_{noise_name}"].std()
            )

    #***********************************************************************************************
    # Set AeStHeTiCs.
    #***********************************************************************************************
    sns.set_theme(style="whitegrid")

    fontsize = 60
    legend_fontsize = 20
    plt.rc("axes", labelsize=fontsize)
    plt.rc("legend", title_fontsize=legend_fontsize)
    plt.rc("legend", fontsize=legend_fontsize)
    plt.rc("xtick", labelsize=fontsize)
    plt.rc("ytick", labelsize=fontsize)

    font = {"family": "monospace", "monospace": ["Times New Roman"]}
    plt.rc("font", **font)

    #***********************************************************************************************
    # Plot the results.
    #***********************************************************************************************
    (fig, axs) = plt.subplots()
    fig.tight_layout()

    days = np.arange(1, 8)

    axs.bar(
        days -0.3, results["correlation"], color=(0.5, 0, 0), width=0.2, label="Load: Correlation"
    )

    axs.bar(
        days - 0.1, results_v["correlation"], color=(1, 0, 0), edgecolor="black", width=0.2,
        label="Virtual: Correlation"
    )

    axs.bar(days + 0.1, results["voltage"], color=(0, 0.5, 0), width=0.2, label="Load: Voltage")

    axs.bar(
        days + 0.3, results_v["voltage"], color=(0, 1, 0), edgecolor="black", width=0.2,
        label="Virtual: Voltage"
    )

    axs.legend()

    yticks = [round(x, 2) for x in np.arange(start=0.0, stop=0.13, step=0.02)]
    axs.set_yticks(yticks)
    axs.set_ylim([yticks[0], yticks[-1] + 0.00])
    axs.set_xticks(days)

    axs.set_xlabel("Window Width (Days)")
    axs.set_ylabel("Accuracy Spread")
    axs.legend(loc="upper right")

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
