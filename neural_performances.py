from data_import import dict_import, select_data, paths, exceptions, specific_time, dates, normalize_data
from nilmtk.electric import get_activations
from utilities import train, ts2windows, unscalerise, nilm_f1score, def_model, plot_results
from sklearn.metrics import mean_absolute_error
from data_format import window_format_val
import matplotlib.pyplot as plt
from utilities import barchart
import numpy as np
import random


#########################################################
exp_title = 'testv4'
sample_s = 7
batchsz = 512
epochs = 10
patience = 6
#########################################################
#APPLIANCES = ['washing machine', 'kettle', 'dish washer', 'television', 'microwave']

APPLIANCES = ['washing machine', 'tumble dryer', 'microwave', 'television', 'fan', 'dish washer', 'audio system',
              'kettle', 'electric space heater', 'food processor', 'games console', 'washer dryer',
              'fridge', 'pond pump', 'dehumidifier', 'freezer', 'breadmaker', 'broadband router',
              'fridge freezer', 'toaster', 'computer']

APPLIANCES = ['washing machine', 'tumble dryer', 'microwave', 'television', 'fan', 'dish washer', 'audio system',
              'unknown', 'kettle', 'electric space heater', 'games console', 'washer dryer',
              'appliance', 'fridge', 'pond pump', 'dehumidifier', 'freezer', 'breadmaker', 'broadband router',
              'fridge freezer', 'toaster', 'computer']  # missing food processor

APPLIANCES = ['washing machine']

############################# a faire #######################################
# nettoyer les données
# regarder que les mae et les f1 soit normaux
# retirer pomp pump voir autres
# si il manque trop de données, on degage le trou (mains et appliance) ou on peut prolonger le dernier point !
# si consommation trop forte sur longtemps alors reduire ou degager


# pomp pump => val_loss: e-11
# fridge freezer/computer => val loss: 1

all_seqlen = [102, 102, 102, 102, 102, 171, 102,
              102, 85, 102, 102, 102,
              102, 102, 102, 102, 102, 102, 102,
              102, 102, 102, ]
#all_seqlen = [102, 102, 102, 102, 102, 102, 102, ]  # 102 better for dishwasher ? / [102, 85, 171, ]
#########################################################
mode = 'train'  # 'train' or 'test'
test_mode = 'auto'  # mode 'select' for a limited number of houses, else 'auto'
mode_norm = 'standard'  # mode for choosing scaler: 'standard', 'min_max' or 'unscale'
neural_mode = 'reduced'  # normal or reduced for neural model
#########################################################
nb = 3  # number of max random chosen houses
t = 6000 + 15000 + 4000  # graph display parameters
thr = 100  # f1 score threshold
delta = 1000
save = 'OFF'  # 'ON' for saving graph, else 'OFF' => create directory file before !
#########################################################


# format files and timestamps
experience_title, folder_location, graph_loc, REFIT_path = paths(exp_title)
reference, APPLIANCES_dict, ALL_APPLIANCES = dict_import(REFIT_path)
reference, APPLIANCES_dict = exceptions(reference, APPLIANCES_dict)  # remove corrupted data from the experience
dates = dates()  # gather all the selected dates
time_periods = specific_time(dates, reference['buildings_ids'])  # or use the function tim_periods
print(ALL_APPLIANCES)

# importing raw data
import_kwargs, train_kwargs = {'APPLIANCES_dict': APPLIANCES_dict, 'apps': APPLIANCES, 'path': REFIT_path,
                               'sample_s': sample_s, 'time_periods': time_periods, 'mode': test_mode,
                               'house_number': nb}, {'batchsz': batchsz, 'epochs': epochs, 'patience': patience, }
seqlens, test_kwargs, m = {}, {}, 0

data, bd_ids = select_data(**import_kwargs)
for k, appliance in enumerate(APPLIANCES):
    seqlens[appliance] = all_seqlen[k]

# correct data errors and normalize it
data, scalers = normalize_data(data, mode_norm)

outputs = {}

true_test_kwargs, i = {}, 1
for appliance in APPLIANCES:
    assert mode in ['test', 'train']

    # choosing model with appropriate sequence length
    seqlen = seqlens[appliance]
    model = def_model(mode, neural_mode, seqlen, folder_location, experience_title, appliance)
    train_kwargs['model'] = model

    # choosing the house of test and deleting it from train data then format all data
    houses = list(data[appliance])

    if len(houses) >= 2:  # cannot test on unknown house if only 1 house available
        test_kwargs = {houses[-1]: data[appliance][houses[-1]], houses[-2]: data[appliance][houses[-2]],
                       'timestamps': {}}
        test_kwargs['timestamps'][houses[-1]] = data[appliance][houses[-1]][appliance].index
        test_kwargs['timestamps'][houses[-2]] = data[appliance][houses[-2]][appliance].index
        if len(houses) > 2:
            for house in list(test_kwargs.keys())[:2]:
                del data[appliance][house]
    if len(houses) == 1:
        test_kwargs = {houses[-1]: data[appliance][houses[-1]], 'timestamps': {}}
        test_kwargs['timestamps'][houses[-1]] = data[appliance][houses[-1]][appliance].index
    outputs[appliance] = {}

    # for future usage
    true_test_kwargs[appliance] = test_kwargs

    window_format_val(data, appliance, train_kwargs, seqlen)

    # train & save model
    if mode == 'train':
        history = train(**train_kwargs)
        model.save(folder_location + experience_title + '/' + appliance)

    for house in list(test_kwargs.keys())[:-1]:
        # testing on an unknown house
        test = test_kwargs[house]
        output = model.predict(ts2windows(test['mains'].values, seqlen, padding='output'))
        unscaled_values = unscalerise(test, house, appliance, 'known', scalers)
        unscaled_values_output = unscalerise(test, house, appliance, 'output', scalers, output)

        outputs[appliance][house] = {'output': output, 'unscaled_values': unscaled_values,
                                     'unscaled_values_output': unscaled_values_output}

        # score calculation
        outputs[appliance][house]['f1'] = \
            nilm_f1score(unscaled_values[appliance], unscaled_values_output[appliance], thr)

        outputs[appliance][house]['mae'] = \
            mean_absolute_error(unscaled_values[appliance], unscaled_values_output[appliance])
        print(str(i) + '/' + str(len(APPLIANCES)) + 'done')


for appliance in list(outputs.keys()):
    test_kwargs = true_test_kwargs[appliance]
    maes, f1s = np.zeros([2]), np.zeros([2])
    for i, house in enumerate(outputs[appliance].keys()):
        f1s[i] = outputs[appliance][house]['f1']
        maes[i] = outputs[appliance][house]['mae']

        # reset the correct index on outputs
        outputs[appliance][house]['unscaled_values'] = \
            outputs[appliance][house]['unscaled_values'].set_index(test_kwargs['timestamps'][house])

        outputs[appliance][house]['unscaled_values_output'] = \
            outputs[appliance][house]['unscaled_values_output'].set_index(test_kwargs['timestamps'][house])

        dat = outputs[appliance][house]['unscaled_values'][appliance]
        thr_act = dat.max() // 2
        activation = get_activations(dat, seqlen // 3, 1, seqlens[appliance], thr_act)

        if not activation:
            timestamp = outputs[appliance][house]['unscaled_values_output'].index[:1000]
        else:
            # choose random activation time
            a = np.linspace(0, len(activation) - 1, len(activation))
            random.shuffle(a)
            k = int(a[0])
            timestamp = activation[k].index

        # select the event to display
        app = outputs[appliance][house]['unscaled_values'][appliance].loc[timestamp[0]: timestamp[-1]]
        mains = outputs[appliance][house]['unscaled_values']['mains'].loc[timestamp[0]: timestamp[-1]]
        out = outputs[appliance][house]['unscaled_values_output'][appliance].loc[timestamp[0]: timestamp[-1]]

        plt.plot(mains.values)
        plt.plot(app.values)
        plt.plot(out.values)

        plt.legend(['mains', 'reference', 'output'])
        plt.title('Test: REFIT-' + house.split('g')[1] + ' on ' + appliance)
        plt.xlabel('Time')
        plt.ylabel('W')

        plt.show()

        if save == 'ON':
            plt.savefig(graph_loc + experience_title + '/' + appliance + '_' + house + '.png')
        plt.show()

    barchart(f1s, maes, list(outputs[appliance].keys()), appliance, graph_loc, experience_title, save)
barchart(f1s, maes, list(outputs[appliance].keys()), appliance, graph_loc, experience_title, save)

bar = {'f1s': [], 'maes': []}
for appliance in list(outputs.keys()):
    mean_f1 = 0
    mean_mae = 0
    d = 0
    for i, house in enumerate(outputs[appliance].keys()):
        mean_f1 = outputs[appliance][house]['f1'] + mean_f1
        mean_mae = outputs[appliance][house]['mae'] + mean_mae
        d = d+1

    bar['f1s'].append(mean_f1 / d)
    bar['maes'].append(mean_mae / d)

print('---end---')

# test sur 2 maisons → peut-être random en prenant la taille exacte de maisons en mode random !
# ou random les ids de APPLIANCE_dict

# revoir les données d'entraînement
# changer le seuil
# faire un graph avec metriques cumulées par appareil


import pandas
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.pyplot import figure
figure(figsize=(3, 2), dpi=80)

df = pandas.DataFrame(dict(graph=list(outputs.keys()),
                           n=bar['f1s'], m=np.array(bar['maes'])))
ind = np.arange(len(df))
width = 0.4
fig, ax = plt.subplots()
ax.barh(ind, df.n, width, color='red', label='f1 score')
ax.barh(ind + width, df.m, width, color='green', label='mae')
ax.set(yticks=ind + width, yticklabels=df.graph, ylim=[2*width - 1, len(df)])
ax.legend()
plt.gcf().set_size_inches(10, 9)
plt.show()


