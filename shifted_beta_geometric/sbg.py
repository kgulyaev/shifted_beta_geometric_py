"""
Implementation of the shifted beta geometric (sBG) model from "How to Project Customer Retention" (Fader and Hardie 2006)

http://www.brucehardie.com/papers/021/sbg_2006-05-30.pdf

Apache 2 License
"""

from math import log

import numpy as np

from scipy.optimize import minimize
from scipy.special import hyp2f1

__author__ = "JD Maturen"


def generate_probabilities(alpha, beta, x):
    """Generate probabilities in one pass for all t in x"""
    p = [alpha / (alpha + beta)]
    for t in range(1, x):
        pt = (beta + t - 1) / (alpha + beta + t) * p[t - 1]
        p.append(pt)
    return p


def probability(alpha, beta, t):
    """Probability function P"""
    if t == 0:
        return alpha / (alpha + beta)
    return (beta + t - 1) / (alpha + beta + t) * probability(alpha, beta, t - 1)


def survivor(probabilities, t):
    """Survivor function S"""
    s = 1 - probabilities[0]
    for x in range(1, t + 1):
        s = s - probabilities[x]
    return s


def log_likelihood(alpha, beta, data, survivors=None):
    """Function to maximize to obtain ideal alpha and beta parameters"""
    if alpha <= 0 or beta <= 0:
        return -1000
    if survivors is None:
        survivors = survivor_rates(data)
    probabilities = generate_probabilities(alpha, beta, len(data))
    final_survivor_likelihood = survivor(probabilities, len(data) - 1)

    return sum([s * log(probabilities[t]) for t, s in enumerate(survivors)]) + data[-1] * log(
        final_survivor_likelihood
    )


def log_likelihood_multi_cohort(alpha, beta, data):
    """Function to maximize to obtain ideal alpha and beta parameters using data across multiple (contiguous) cohorts.
    `data` must be a list of cohorts each with an absolute number per observed time unit."""
    if alpha <= 0 or beta <= 0:
        return -1000
    probabilities = generate_probabilities(alpha, beta, len(data[0]))

    cohorts = len(data)
    total = 0
    for i, cohort in enumerate(data):
        total += sum(
            [(cohort[j] - cohort[j + 1]) * log(probabilities[j]) for j in range(len(cohort) - 1)]
        )
        total += cohort[-1] * log(survivor(probabilities, cohorts - i - 1))
    return total


def survivor_rates(data):
    s = []
    for i, x in enumerate(data):
        if i == 0:
            s.append(1 - data[0])
        else:
            s.append(data[i - 1] - data[i])
    return s


def maximize(data):
    survivors = survivor_rates(data)
    func = lambda x: -log_likelihood(x[0], x[1], data, survivors)
    x0 = np.array([100.0, 100.0])
    res = minimize(func, x0, method="nelder-mead", options={"xtol": 1e-8})
    return res


def maximize_multi_cohort(data):
    func = lambda x: -log_likelihood_multi_cohort(x[0], x[1], data)
    x0 = np.array([1.0, 1.0])
    res = minimize(func, x0, method="nelder-mead", options={"xtol": 1e-8})
    return res


def predicted_retention(alpha, beta, t):
    """Predicted retention probability at t. Function 8 in the paper"""
    return (beta + t) / (alpha + beta + t)


def predicted_survival(alpha, beta, x):
    """Predicted survival probability, i.e. percentage of customers retained, for all t in x.
    Function 1 in the paper"""
    s = [predicted_retention(alpha, beta, 0)]
    for t in range(1, x):
        s.append(predicted_retention(alpha, beta, t) * s[t - 1])
    return s


def fit(data):
    res = maximize(data)
    if res.status != 0:
        raise Exception(res.message)
    return res.x


def fit_multi_cohort(data):
    res = maximize_multi_cohort(data)
    if res.status != 0:
        raise Exception(res.message)
    return res.x


def derl(alpha, beta, d, n):
    """discounted expected residual lifetime from "Customer-Base Valuation in a Contractual Setting: The Perils of
    Ignoring Heterogeneity" (Fader and Hardie 2009)"""
    return predicted_retention(alpha, beta, n) * hyp2f1(
        1, beta + n + 1, alpha + beta + n + 1, 1 / (1 + d)
    )


def predict_with_terminal_churn(data, future_periods, max_terminal_renewal=0.98):
    """
    For data with duplicate values such as this:
    0.695652174 0.695652174 0.304347826 0.130434783

    The sBG method breaks down, here will just use
    the terminal churn method, where the renewal rate in the future
    equals the last known renewal rate
    """

    future_data = data.copy()

    terminal_renewal_rate = min(data[-1] / data[-2], max_terminal_renewal)
    last_value = data[-1]
    for i in range(future_periods):
        next_value = last_value * terminal_renewal_rate
        last_value = next_value
        future_data.append(next_value)

    return future_data


def apply_smoothing(data, predicted_data):
    """
    For some predictions the first N predicted values are higher than the
    last known value, which cannot happen in reality with retention data.

    Example:

    A           B           C
    Known       Known       Predicted   Predicted
    0.12342216  0.119615308 0.135569514 0.132488536

    This smoothing method adjusts all future predictions down by:
    C - (B/A) * B

    The idea here is that (B/A) * B is our expected next period value, but C is our
    next value from the curve. Adjust the whole curve down by the difference of those
    values

    """

    last_known_datapoint = data[-1]
    first_predicted_datapoint = predicted_data[len(data)]

    if first_predicted_datapoint > last_known_datapoint:
        first_pred_val = predicted_data[len(data)]
        rev_first_pred_val = (last_known_datapoint / data[-2]) * last_known_datapoint
        adj_fact = first_pred_val - rev_first_pred_val

        for i in range(len(data), len(predicted_data)):
            predicted_data[i] = predicted_data[i] - adj_fact

    return predicted_data


def fit_predict(data, future_periods, include_past=False, smoothing=True, cohort_name=None):
    """Combine fit and predict_survival into one step"""

    sbg_fit_failed = False

    if data[-1] == 0.0:
        predicted_data = data.copy()
        for i in range(future_periods):
            predicted_data.append(0.0)
        sbg_fit_failed = True

    if not sbg_fit_failed:
        try:
            alpha, beta = fit(data)
        except:
            if cohort_name is not None:
                print("Switching to terminal churn rate for cohort: {}".format(cohort_name))
            predicted_data = predict_with_terminal_churn(data, future_periods)
            sbg_fit_failed = True

        if not sbg_fit_failed:
            # predict the next future_periods time samples:
            predicted_data = predicted_survival(alpha, beta, len(data) + future_periods)

            if smoothing:
                predicted_data = apply_smoothing(data, predicted_data)

    if include_past:
        return predicted_data
    else:
        return predicted_data[-future_periods:]


def test():
    """Test against the High End subscription retention data from the paper"""
    example_data = [0.869, 0.743, 0.653, 0.593, 0.551, 0.517, 0.491]
    ll11 = log_likelihood(1.0, 1.0, example_data)
    print(np.allclose(ll11, -2.115, 1e-3))

    res = maximize(example_data)
    alpha, beta = res.x
    print(res.status == 0 and np.allclose(alpha, 0.668, 1e-3) and np.allclose(beta, 3.806, 1e-3))
    print()

    print("real\t", ["{0:.1f}%".format(x * 100) for x in example_data])
    print("pred\t", ["{0:.1f}%".format(x * 100) for x in predicted_survival(alpha, beta, 12)])
    print()

    print(list(map("{0:f}".format, [derl(alpha, beta, 0.1, x) for x in range(12)])))
    print()

    multi_cohort_data = [
        [10000, 8000, 6480, 5307, 4391],
        [10000, 8000, 6480, 5307],
        [10000, 8000, 6480],
        [10000, 8000],
    ]
    alpha, beta = fit_multi_cohort(multi_cohort_data)
    print(np.allclose(alpha, 3.80, 1e-2) and np.allclose(beta, 15.19, 1e-2))


def added_tests():
    # two examples of duplicate values
    data1 = [0.695652174, 0.695652174, 0.304347826, 0.130434783]
    data2 = [1.0, 1.0, 0.666666667, 0.333333333]

    # handling zero retention at the end
    data3 = [0.662093647, 0.53146436, 0.432448554, 0.371905756, 0.0]

    # handling smoothing
    data4 = [
        0.662093647,
        0.53146436,
        0.432448554,
        0.371905756,
        0.326871458,
        0.294661497,
        0.267223382,
        0.240679988,
        0.225171488,
        0.204891142,
        0.190873844,
        0.174470623,
        0.16433045,
        0.1565762,
        0.144646585,
        0.136892335,
        0.130032806,
        0.12496272,
    ]

    # should switch to using terminal churn rate
    t1 = fit_predict(data1, 5, cohort_name="Test Cohort 1")
    t2 = fit_predict(data2, 5, cohort_name="Test Cohort 2")

    # should return all zeros at the end
    t3 = fit_predict(data3, 5, cohort_name="Test Cohort 3")
    print(t3)

    # test of smoothing, first smoothing is off
    t4 = fit_predict(data4, 5, cohort_name="Test Cohort 4", smoothing=False)
    print(f"Future value of {t4[0]} is greater than present value of {data4[-1]}")

    # now with smoothing turned on
    t5 = fit_predict(data4, 5, cohort_name="Test Cohort 4", smoothing=True)
    print(f"Future values like {t5[0]} are adjusted to be less than {data4[-1]}")


if __name__ == "__main__":
    test()
    added_tests()
