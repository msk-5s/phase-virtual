# SPDX-License-Identifier: MIT

"""
This script runs phase identification for a single snapshot in time.

It's important to note that we can't rely on a single snapshot in time to determine overall
accuracy. However, this script is useful for observing algorithm performance at specific windows
with specific random seeds as determined by the results from running `run_suite.py`.
"""

import numpy as np
import sklearn.metrics

import clusterer
import convert
import data_factory
import model
import transform

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals, too-many-statements
    """
    The main function.
    """
    # We want the randomness to be repeatable.
    # The `random_state` in the results can be used instead, to examine a specific snapshot in the
    # year. Be sure to set the correct `start` of the window.
    rng = np.random.default_rng(seed=1337)
    random_state = rng.integers(np.iinfo(np.int32).max)

    #***********************************************************************************************
    # Load the data.
    #***********************************************************************************************
    days = 7
    timesteps_per_day = 96
    width = days * timesteps_per_day
    start = 0

    data = data_factory.make_data()[start:(start + width), :]
    data_n = data_factory.make_data_gauss_noise(data=data, percent=0.005, random_state=random_state)

    phase_dict = data_factory.make_phase_labels(percent_error=0.4, random_state=random_state)
    xfmr_labels = data_factory.make_xfmr_labels()

    #***********************************************************************************************
    # Transform the load measurements to virtual transformer measurements.
    #***********************************************************************************************
    data_v = convert.load_to_virtual(data=data, xfmr_labels=xfmr_labels)
    data_vn = convert.load_to_virtual(data=data_n, xfmr_labels=xfmr_labels)

    #***********************************************************************************************
    # Make the time series' stationary.
    #***********************************************************************************************
    cutoff = 0.06
    order = 10

    data_s = transform.filter_butterworth(data=data, cutoff=cutoff, order=order)
    data_ns = transform.filter_butterworth(data=data_n, cutoff=cutoff, order=order)

    data_vs = transform.filter_butterworth(data=data_v, cutoff=cutoff, order=order)
    data_vns = transform.filter_butterworth(data=data_vn, cutoff=cutoff, order=order)

    #***********************************************************************************************
    # Run phase identification.
    #***********************************************************************************************
    predict_clusters = clusterer.run_kmeans_voltage

    predictions = model.predict(
        data=data_s, phase_labels=phase_dict["error"], predict_clusters=predict_clusters,
        random_state=random_state
    )

    predictions_n = model.predict(
        data=data_ns, phase_labels=phase_dict["error"], predict_clusters=predict_clusters,
        random_state=random_state
    )

    predictions_v = model.predict(
        data=data_vs, phase_labels=phase_dict["error"], predict_clusters=predict_clusters,
        random_state=random_state, xfmr_labels=xfmr_labels
    )

    predictions_vn = model.predict(
        data=data_vns, phase_labels=phase_dict["error"], predict_clusters=predict_clusters,
        random_state=random_state, xfmr_labels=xfmr_labels
    )

    accuracies = {
        "data": sklearn.metrics.accuracy_score(y_true=phase_dict["true"], y_pred=predictions),
        "data_n": sklearn.metrics.accuracy_score(y_true=phase_dict["true"], y_pred=predictions_n)
    }

    accuracies_v = {
        "data_v": sklearn.metrics.accuracy_score(y_true=phase_dict["true"], y_pred=predictions_v),
        "data_vn": sklearn.metrics.accuracy_score(y_true=phase_dict["true"], y_pred=predictions_vn)
    }

    #***********************************************************************************************
    # Print out results.
    #***********************************************************************************************
    print("-"*50)

    for (name, accuracy) in accuracies.items():
        print(f"{name} Accuracy: {accuracy}")

    for (name, accuracy) in accuracies_v.items():
        print(f"{name} Accuracy: {accuracy}")

    print("-"*50)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
