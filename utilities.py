import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from neural_model import network_seq2point, seq2point_reduced


def ts2windows(arr, seqlen, step=1, stride=1, padding='after'):
    assert padding in ['before', 'after', 'output', None, ]
    arrlen = len(arr)  # original length of the array.
    if padding is None:
        n = seqlen // 2
        arr = arr.flatten()
        windows = [arr[i - n:i - n + seqlen:step] for i in range(n, arrlen - n, stride)]
    else:
        pad_width = (seqlen - 1, 0) if padding == 'before' else (0, seqlen - 1)
        if padding == 'output':
            pad_width = seqlen // 2
        arr = np.pad(arr.flatten(), pad_width, mode="constant", constant_values=0)

        windows = [arr[i:i + seqlen:step] for i in range(0, arrlen, stride)]

    windows = np.array(windows, dtype="float32")
    return windows


def normalize_per_series(ts_dict):
    scalers = {}
    transformed = {}
    for key, data in ts_dict.items():
        zscale = StandardScaler().fit(data)
        if isinstance(data, pd.DataFrame):
            transformed[key] = pd.DataFrame(zscale.transform(data),
                                            index=data.index, columns=data.columns)
        else:
            transformed[key] = zscale.transform(data)
        scalers[key] = zscale
    return transformed, scalers


def train(in_data, out_data, in_val, out_val, model, batchsz, epochs, patience):
    stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        verbose=0,
        mode="min",
        restore_best_weights=True,
    )
    history = model.fit(
        in_data, out_data,
        batch_size=batchsz,
        epochs=epochs,
        validation_data=(in_val, out_val),
        verbose=1,
        callbacks=[stop, ],
    )
    return history.history


def nilm_binary_confusion(y_true, y_pred, thr):
    pos_gt = y_true > thr
    pos_pred = y_pred > thr
    neg_gt = y_true <= thr
    neg_pred = y_pred <= thr
    positive = np.logical_and(pos_gt, pos_pred)
    negative = np.logical_and(neg_gt, neg_pred)
    truepos = np.count_nonzero(positive)
    trueneg = np.count_nonzero(negative)
    falseneg = np.count_nonzero(pos_gt) - truepos
    falsepos = np.count_nonzero(pos_pred) - truepos
    return truepos, trueneg, falsepos, falseneg


def nilm_f1score(y_true, y_pred, thr=1000, mean=0, std=1, ):
    """ Binarize the time series before computing the F1-score.
        If the data is normalized, use the mean and std args to normalize the
        threshold.
    """
    thr = (thr - mean) / std
    truepos, _, falsepos, falseneg = nilm_binary_confusion(y_true, y_pred, thr)
    denom = 2 * truepos + falseneg + falsepos
    f1 = 0.
    if denom > 0:
        f1 = 2 * truepos / denom
    return f1


def unscalerise(data, house, appliance, mode, scalers, output={}):
    assert mode in ['known', 'output', ]
    scaler = scalers[appliance][house]
    if mode == 'known':
        normalized_data = data
    if mode == 'output':
        normalized_data = data
        normalized_data[appliance] = output
    unscaled = pd.DataFrame(scaler.inverse_transform(normalized_data.values), columns=['mains', appliance])
    return unscaled


def def_model(mode, neural_mode, seqlen, folder_location, experience_title, appliance):
    if mode == 'train':
        if neural_mode == 'normal':
            model = network_seq2point(seqlen)
        else:
            model = seq2point_reduced(seqlen)
    if mode == 'test':
        model = keras.models.load_model(folder_location + experience_title + '/' + appliance)
    return model


def barchart(f1s, maes, houses, appliance, graph_location, experience_title, save):
    assert save in ['ON', 'OFF', None, ]
    plt.rcdefaults()
    n_groups = len(houses)  # len(APPLIANCES)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.4
    opacity = 0.8

    plt.bar(index, f1s, bar_width, alpha=opacity, color='b', label='f1 score')
    plt.bar(index + bar_width, maes / 100, bar_width, alpha=opacity, color='g', label='MAE (1e2 W)')
    plt.xlabel('Houses')
    plt.ylabel('Values')
    plt.title('Scores for ' + appliance)
    plt.xticks(index + 0.2, houses)
    plt.legend()
    plt.tight_layout()
    if save == 'ON':
        plt.savefig(graph_location + experience_title + '/' + 'BARCHART' + '_' + appliance + '.png')
    plt.show()


def plot_results(t, delta_t, unscaled_values, unscaled_values_output, appliance, test_kwargs, save, graph_location,
                 experience_title):
    plt.plot(np.array(unscaled_values_output[appliance])[t:t + delta_t])
    plt.plot(np.array(unscaled_values['mains'])[t:t + delta_t])
    plt.plot(np.array(unscaled_values[appliance])[t:t + delta_t])
    plt.legend(['output', 'mains', 'reference'])
    plt.title('Test: REFIT-' + test_kwargs['house'].split('g')[1] + ' on ' + appliance)
    plt.xlabel('Time')
    plt.ylabel('W')
    plt.show()

    if save == 'ON':
        plt.savefig(graph_location + experience_title + '/' + appliance + '.png')
    plt.show()
