import numpy as np
from scipy.stats import norm


def get_95_ci_intervals(values): # 2D values
    intervals = []
    for _, row in values.iterrows():
        intervals.append(get_credible_interval(row, 2.5, 97.5))
    return intervals


def get_95_cb_intervals(values):
    intervals = []
    for _, row in values.iterrows():  # row shape should be (mean, std)
        intervals.append(get_credible_bounds(row[0], row[1], 2.5, 97.5))
    return intervals


def get_credible_interval(values, credible_lower_percent, credible_upper_percent):  # 1D values
    lower_bound = np.percentile(values, credible_lower_percent)
    upper_bound = np.percentile(values, credible_upper_percent)

    mask = (values >= lower_bound) & (values <= upper_bound)

    credible_values = values[mask]

    return credible_values


# For use with BNNs
def get_credible_bounds(mean, std, credible_lower_percent, credible_upper_percent):
    lower_bound = norm.ppf(credible_lower_percent / 100, loc=mean, scale=std)
    upper_bound = norm.ppf(credible_upper_percent / 100, loc=mean, scale=std)

    return lower_bound, upper_bound


def tvs_score(credible_values, target_ys):  # takes a 2d array for credible values
    true_positives = 0

    for i, credible_value_set in enumerate(credible_values):
        if np.max(credible_value_set) >= target_ys.iloc[i] >= np.min(credible_value_set):
            true_positives += 1

    return true_positives / len(target_ys)


def tvs_score_bnn(credible_bounds, target_ys):  # takes a 2d array for credible values
    true_positives = 0

    for i, bounds in enumerate(credible_bounds): # credible bounds should be shape (lower, upper)
        if bounds[1] >= target_ys.iloc[i] >= bounds[0]:
            true_positives += 1

    return true_positives / len(target_ys)


# If inputting BNN results, set as credible_values_a as (lower_bound, upper_bound) and preds_a for mean
def cds_score(credible_values_a, preds_a, credible_values_b, preds_b, bnn=False):
    if bnn == False:
        return cds_score_standard(credible_values_a, preds_a, credible_values_b, preds_b)
    else:
        return cds_score_bnn(credible_values_a, preds_a, credible_values_b, preds_b)


def cds_score_standard(credible_values_a, preds_a, credible_values_b, preds_b):
    total_resembling = 0

    for i, credible_value_set in enumerate(credible_values_a):
        if np.max(credible_value_set) >= preds_b[i] >= np.min(credible_value_set):
            total_resembling += 1

    for i, credible_value_set in enumerate(credible_values_b):
        if np.max(credible_value_set) >= preds_a[i] >= np.min(credible_value_set):
            total_resembling += 1

    return total_resembling / (2 * len(preds_a))


def cds_score_bnn(credible_values_a, preds_a, credible_values_b, preds_b):
    total_resembling = 0

    for i, credible_boundaries in enumerate(credible_values_a):
        if credible_boundaries[1] >= preds_b[i] >= credible_boundaries[0]:
            total_resembling += 1

    for i, credible_value_set in enumerate(credible_values_b):
        if np.max(credible_value_set) >= preds_a[i] >= np.min(credible_value_set):
            total_resembling += 1

    return total_resembling / (2 * len(preds_a))


def quality_score(credible_values, target_ys):
    total = 0

    for i, credible_value_set in enumerate(credible_values):
        upper_ci = np.max(credible_value_set)
        lower_ci = np.min(credible_value_set)

        if not (np.max(credible_value_set) >= target_ys.iloc[i] >= np.min(credible_value_set)):
            upper_diff = abs(target_ys.iloc[i] - upper_ci)
            lower_diff = abs(target_ys.iloc[i] - lower_ci)
            min_difference = np.min([upper_diff, lower_diff])
            total += min_difference ** 2

        total += upper_ci - lower_ci

    return total


# For use with BNNs
def quality_score_bnn(credible_bounds, target_ys):  # credible bounds should be shape (lower, upper)
    total = 0

    for i, bounds in enumerate(credible_bounds):
        if not (bounds[1] >= target_ys.iloc[i] >= bounds[0]):
            upper_diff = abs(target_ys.iloc[i] - credible_bounds[1])
            lower_diff = abs(target_ys.iloc[i] - credible_bounds[0])
            min_difference = np.min([upper_diff, lower_diff])
            total += min_difference ** 2

        total += bounds[1] - bounds[0]

    return total
