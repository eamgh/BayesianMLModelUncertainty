from sklearn.metrics import mean_absolute_error
import numpy as np


def get_best_model_idx(all_model_predictions, target_ys):
    model_mae_scores = np.empty(10)

    for i, model_results in enumerate(all_model_predictions):
        model_predictions = np.empty(len(target_ys))
        for j in range(0, len(target_ys)):
            model_predictions[j] = np.mean(model_results[:, j])
        model_mae_scores[i] = mean_absolute_error(target_ys, model_predictions)

    idx_of_best = np.argmin(model_mae_scores)

    return idx_of_best


def write_ensemble_model_results(results, filename):
    with open(filename, "a") as file:
        for row in results:
            for i, value in enumerate(row):
                file.write(str(value))
                if i != len(results)-1:
                    file.write(",")
            file.write("\n")


def bf_get_best_model_idx(all_results, target_ys):
    scores = []
    for result_set in all_results:
        means = np.mean(result_set, axis=1)
        scores.append(mean_absolute_error(target_ys, means))

    return np.argmin(scores)


def write_bf_model_results(results, filename):
    with open(filename, "a") as file:
        for row in results:
            for i, value in enumerate(row):
                file.write(str(value))
                if i != len(results)-1:
                    file.write(",")
            file.write("\n")


def bnn_get_best_model_idx(all_results, target_ys):
    scores = []
    for model_results in all_results:
        means = model_results[:, 0]
        scores.append(mean_absolute_error(target_ys, means))

    return np.argmin(scores)


def write_bnn_model_results(results, filename):
    with open(filename, "a") as file:
        for row in results:
            for i, value in enumerate(row):
                file.write(str(value))
                if i != 1:
                    file.write(",")
            file.write("\n")