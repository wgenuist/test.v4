from utilities import ts2windows
import numpy as np


def window_format_val(data, appliance, train_kwargs, seqlen):
    dictionary = data[appliance]
    in_data_dict, out_data_dict, in_val_dict, out_val_dict, n = {}, {}, {}, {}, seqlen // 2
    for house in dictionary.keys():
        length = len(dictionary[house])
        in_data_dict[house] = dictionary[house]['mains'][:80 * length // 100]
        out_data_dict[house] = dictionary[house][appliance][:80 * length // 100]
        in_val_dict[house] = dictionary[house]['mains'][80 * length // 100:]
        out_val_dict[house] = dictionary[house][appliance][80 * length // 100:]

    in_data_dict = {k: ts2windows(v.values, seqlen, padding=None) for k, v in in_data_dict.items()}
    out_data_dict = {k: np.reshape(v.values, (-1, 1))[n:-n] for k, v in out_data_dict.items()}
    in_val_dict = {k: ts2windows(v.values, seqlen, padding=None) for k, v in in_val_dict.items()}
    out_val_dict = {k: np.reshape(v.values, (-1, 1))[n:-n] for k, v in out_val_dict.items()}

    train_kwargs['in_data'], train_kwargs['out_data'] = np.vstack(in_data_dict.values()), np.vstack(
        out_data_dict.values())
    train_kwargs['in_val'], train_kwargs['out_val'] = np.vstack(in_val_dict.values()), np.vstack(
        out_val_dict.values())
    return in_data_dict, out_data_dict, in_val_dict, out_val_dict


def data_event_format(app_in, app_out, threshold, padding, sample_s, mode=None):
    """ formater les données d'entraînement en isolant les fenêtres d'event + trouver seqlen"""
    # app_out_appliances['appliance']['house'] => length 86400
    assert mode in ['perso', None, ]
    size = app_out.shape[0]
    data = np.array(app_out)
    data_reduced = np.array([app_out[2 * i] for i in range(size // 2)])
    seqlen = 0
    if mode is None:
        threshold = max(data_reduced) // 1.25
        padding = 50 * sample_s
    i, maximum, compt, index = 0, 0, np.array([0, 0]), np.array([])
    for point in data:
        if point > threshold:  # début de l'event
            index = np.append(index, i)
            compt = np.append(compt, 1)
            maximum = maximum + 1
        else:  # fin de l'event
            compt = np.append(compt, 0)
            if maximum > seqlen:  # on garde la sequence max (durée event)
                seqlen = maximum
            maximum = 0
        if compt[-1] < compt[-2]:  # padding après l'event
            index = np.append(index, np.linspace(i, i + padding, padding + 1))
        if compt[-1] > compt[-2]:  # padding avant l'event, on quitte l'event
            index = np.append(index, np.linspace(i - padding - 1, i - 1, padding + 1))
        i = i + 1
        if i > size:
            break
    index = list(set(index))
    isolated_events_in = [app_in[int(i)] for i in index]
    isolated_events_out = [app_out[int(i)] for i in index]
    return isolated_events_in, isolated_events_out  # return seqlen, index
