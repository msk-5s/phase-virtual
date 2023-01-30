# SPDX-License-Identifier: BSD-3-Clause

"""
This script runs phase label correction using clustering for different combinations of parameters.
"""

import sys

from typing import Any, Dict, Mapping
from nptyping import NDArray

from rich.progress import track

import numpy as np
import pandas as pd
import sklearn

import clusterer
import convert
import data_factory
import model
import transform

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals
    """
    The main function.

    The reason for using command line arguments is so this script can be used in an array job on
    high performance computing resources, if available.

    Arguments
    ---------
    array_index : int
        The array job ID provided by the HPC tool.

    Examples
    --------
    The following will run the suite with array job ID 2:

    python3 run_suite.py 2
    """
    #***********************************************************************************************
    # Get command line arguemnts.
    #***********************************************************************************************
    # Depending on the HPC tool being used, the `array_index` will be the array job ID, which may
    # start at 1, rather than 0.
    array_index = int(sys.argv[1])
    param_index = array_index - 1

    #***********************************************************************************************
    # Select the appropriate parameters based on the array job ID.
    #***********************************************************************************************
    days_list = list(range(1, 8, 1))
    noise_percents = [0.005]

    params = [(days, noise_percent) for days in days_list for noise_percent in noise_percents]

    print(f"Parameter combinations: {len(params)}")

    # These can be set manually to run the suite using a specific set of parameters.
    days = params[param_index][0]
    noise_percent = params[param_index][1]

    #***********************************************************************************************
    # Load data and labels.
    #***********************************************************************************************
    # We want the results to be repeatable.
    rng = np.random.default_rng(seed=1337)
    random_state_labels = rng.integers(np.iinfo(np.int32).max)

    labels = {
        "phase_dict": data_factory.make_phase_labels(
            percent_error=0.4, random_state=random_state_labels
        ),
        "xfmr": data_factory.make_xfmr_labels()
    }

    data = data_factory.make_data()

    timesteps_per_day = 96
    width = days * timesteps_per_day

    #***********************************************************************************************
    # Calculate the index for each window in the data.
    #***********************************************************************************************
    start_indices = np.arange(
        start=0, stop=data.shape[0] - timesteps_per_day * (days - 1), step=timesteps_per_day
    )

    #start_indices = np.arange(
    #    start=0, stop=96*10, step=timesteps_per_day
    #)

    #***********************************************************************************************
    # Run the simulation case.
    #***********************************************************************************************
    results_df = _make_results_df(entry_count=len(start_indices))
    results_df["random_state_labels"] = np.repeat(random_state_labels, len(start_indices))

    for (i, start) in track(enumerate(start_indices), "Processing...", total=len(start_indices)):
        random_state = rng.integers(np.iinfo(np.int32).max)
        window = data[start:(start + width), :]

        results = _run_identification(
            data=window, labels=labels, noise_percent=noise_percent, random_state=random_state
        )

        # Append the timestep of the window start. We can use the window start and random seed for
        # examining the results at a specific spot using `run.py`.
        results["window_start"] = start
        results["random_state"] = random_state

        for (key, value) in results.items():
            results_df.at[i, key] = value

    #***********************************************************************************************
    # Save results.
    #***********************************************************************************************
    noise_string = str(noise_percent).replace(".", "p")

    results_df.to_csv(f"results/result-{days}-day-{noise_string}-noise.csv")

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def _run_identification( # pylint: disable=too-many-locals
    data: NDArray[(Any, Any), float], labels: Mapping[str, Any], noise_percent: float,
    random_state: int
) -> Dict[str, Any]:
    """
    Run phase identification.

    Parameters
    ----------
    data : numpy.ndarray of float, ndim=2, (n_timestep, n_load)
        The window of load data.
    labels: dict of Any
        phase_dict : dict of [str, NDArray[(Any,), int]]
            error : numpy.ndarray of int, (n_load,)
                The erroneous phase labels.
            true : numpy.ndarray of int, (n_load,)
                The true phase labels.
        xfmr: numpy.ndarray of int, ndim=1, (n_load,)
            The true transformer labels for each load.
    noise_percent : float
        The percent of noise to inject into the measurements.
    random_state : int
        The random state to use.

    Returns
    -------
    dict of {str: float}
        The accuracy results.
    """
    #***********************************************************************************************
    # Make the windows of noisy data.
    #***********************************************************************************************
    data_n_gauss = data_factory.make_data_gauss_noise(
        data=data, percent=noise_percent, random_state=random_state
    )

    data_n_laplace = data_factory.make_data_laplace_noise(
        data=data, percent=noise_percent, random_state=random_state
    )

    data_vn_gauss = convert.load_to_virtual(data=data_n_gauss, xfmr_labels=labels["xfmr"])
    data_vn_laplace = convert.load_to_virtual(data=data_n_laplace, xfmr_labels=labels["xfmr"])

    #***********************************************************************************************
    # Highpass filter data to make stationary.
    # Note: For this dataset, a Butterworth filter gives better results over a simple difference
    # filter, when using virtual measurements.
    #***********************************************************************************************
    data_ns_gauss = transform.filter_butterworth(data=data_n_gauss, cutoff=0.06, order=10)
    data_vns_gauss = transform.filter_butterworth(data=data_vn_gauss, cutoff=0.06, order=10)
    data_ns_laplace = transform.filter_butterworth(data=data_n_laplace, cutoff=0.06, order=10)
    data_vns_laplace = transform.filter_butterworth(data=data_vn_laplace, cutoff=0.06, order=10)

    #***********************************************************************************************
    # Make the clusterers.
    #***********************************************************************************************
    clusterers = {
        "gmm_correlation": clusterer.run_gmm_correlation,
        "gmm_voltage": clusterer.run_gmm_voltage,
        "hierarchical_correlation": clusterer.run_hierarchical_correlation,
        "hierarchical_voltage": clusterer.run_hierarchical_voltage,
        "kmeans_correlation": clusterer.run_kmeans_correlation,
        "kmeans_voltage": clusterer.run_kmeans_voltage,
        "kmedoids_correlation": clusterer.run_kmedoids_correlation,
        "kmedoids_voltage": clusterer.run_kmedoids_voltage,
        "spectral_correlation": clusterer.run_spectral_correlation,
        "spectral_voltage": clusterer.run_spectral_voltage,
    }

    #***********************************************************************************************
    # Get model predictions.
    #***********************************************************************************************
    results = {}

    items = zip(
        ["gauss", "laplace"], [data_ns_gauss, data_ns_laplace], [data_vns_gauss, data_vns_laplace]
    )

    for (noise_name, data_ns, data_vns) in items:
        for (clusterer_name, predict_clusters) in clusterers.items():
            predictions = model.predict(
                data=data_ns, phase_labels=labels["phase_dict"]["error"],
                predict_clusters=predict_clusters, random_state=random_state
            )

            predictions_v = model.predict(
                data=data_vns, phase_labels=labels["phase_dict"]["error"],
                predict_clusters=predict_clusters, random_state=random_state,
                xfmr_labels=labels["xfmr"]
            )

            accuracy = sklearn.metrics.accuracy_score(
                y_true=labels["phase_dict"]["true"], y_pred=predictions
            )

            accuracy_v = sklearn.metrics.accuracy_score(
                y_true=labels["phase_dict"]["true"], y_pred=predictions_v
            )

            results[f"accuracy_{clusterer_name}_{noise_name}"] = accuracy
            results[f"accuracy_{clusterer_name}_v_{noise_name}"] = accuracy_v

    return results

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def _make_results_df(entry_count: int) -> pd.DataFrame:
    """
    Make an empty results data frame..

    Parameters
    ----------
    entry_count : int
        The number of entries to initialize in the data frame.

    Returns
    -------
    pd.DataFrame
        A new results data frame.
    """
    results_df = pd.DataFrame(data={
        "accuracy_gmm_correlation_gauss": np.zeros(shape=entry_count, dtype=float),
        "accuracy_gmm_correlation_v_gauss": np.zeros(shape=entry_count, dtype=float),
        "accuracy_gmm_voltage_gauss": np.zeros(shape=entry_count, dtype=float),
        "accuracy_gmm_voltage_v_gauss": np.zeros(shape=entry_count, dtype=float),

        "accuracy_hierarchical_correlation_gauss": np.zeros(shape=entry_count, dtype=float),
        "accuracy_hierarchical_correlation_v_gauss": np.zeros(shape=entry_count, dtype=float),
        "accuracy_hierarchical_voltage_gauss": np.zeros(shape=entry_count, dtype=float),
        "accuracy_hierarchical_voltage_v_gauss": np.zeros(shape=entry_count, dtype=float),

        "accuracy_kmeans_correlation_gauss": np.zeros(shape=entry_count, dtype=float),
        "accuracy_kmeans_correlation_v_gauss": np.zeros(shape=entry_count, dtype=float),
        "accuracy_kmeans_voltage_gauss": np.zeros(shape=entry_count, dtype=float),
        "accuracy_kmeans_voltage_v_gauss": np.zeros(shape=entry_count, dtype=float),

        "accuracy_kmedoids_correlation_gauss": np.zeros(shape=entry_count, dtype=float),
        "accuracy_kmedoids_correlation_v_gauss": np.zeros(shape=entry_count, dtype=float),
        "accuracy_kmedoids_voltage_gauss": np.zeros(shape=entry_count, dtype=float),
        "accuracy_kmedoids_voltage_v_gauss": np.zeros(shape=entry_count, dtype=float),

        "accuracy_spectral_correlation_gauss": np.zeros(shape=entry_count, dtype=float),
        "accuracy_spectral_correlation_v_gauss": np.zeros(shape=entry_count, dtype=float),
        "accuracy_spectral_voltage_gauss": np.zeros(shape=entry_count, dtype=float),
        "accuracy_spectral_voltage_v_gauss": np.zeros(shape=entry_count, dtype=float),

        "accuracy_gmm_correlation_laplace": np.zeros(shape=entry_count, dtype=float),
        "accuracy_gmm_correlation_v_laplace": np.zeros(shape=entry_count, dtype=float),
        "accuracy_gmm_voltage_laplace": np.zeros(shape=entry_count, dtype=float),
        "accuracy_gmm_voltage_v_laplace": np.zeros(shape=entry_count, dtype=float),

        "accuracy_hierarchical_correlation_laplace": np.zeros(shape=entry_count, dtype=float),
        "accuracy_hierarchical_correlation_v_laplace": np.zeros(shape=entry_count, dtype=float),
        "accuracy_hierarchical_voltage_laplace": np.zeros(shape=entry_count, dtype=float),
        "accuracy_hierarchical_voltage_v_laplace": np.zeros(shape=entry_count, dtype=float),

        "accuracy_kmeans_correlation_laplace": np.zeros(shape=entry_count, dtype=float),
        "accuracy_kmeans_correlation_v_laplace": np.zeros(shape=entry_count, dtype=float),
        "accuracy_kmeans_voltage_laplace": np.zeros(shape=entry_count, dtype=float),
        "accuracy_kmeans_voltage_v_laplace": np.zeros(shape=entry_count, dtype=float),

        "accuracy_kmedoids_correlation_laplace": np.zeros(shape=entry_count, dtype=float),
        "accuracy_kmedoids_correlation_v_laplace": np.zeros(shape=entry_count, dtype=float),
        "accuracy_kmedoids_voltage_laplace": np.zeros(shape=entry_count, dtype=float),
        "accuracy_kmedoids_voltage_v_laplace": np.zeros(shape=entry_count, dtype=float),

        "accuracy_spectral_correlation_laplace": np.zeros(shape=entry_count, dtype=float),
        "accuracy_spectral_correlation_v_laplace": np.zeros(shape=entry_count, dtype=float),
        "accuracy_spectral_voltage_laplace": np.zeros(shape=entry_count, dtype=float),
        "accuracy_spectral_voltage_v_laplace": np.zeros(shape=entry_count, dtype=float),

        "random_state": np.zeros(shape=entry_count, dtype=int),
        "window_start": np.zeros(shape=entry_count, dtype=int)
    })

    return results_df

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
