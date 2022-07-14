import matplotlib.pyplot as plt
from data_import import dict_import, select_data, paths, time_periods, exceptions, specific_time, dates, normalize_data
plt.close('all')

#########################################################
sample_s = 7
batchsz = 512
epochs = 5
patience = 6
#########################################################
APPLIANCES = ['washing machine', 'kettle', 'dish washer']
APPLIANCES = ['broadband router', 'freezer']
#  ['broadband router', 'food processor','games console','dish washer','washer dryer','unknown','freezer','breadmaker','dehumidifier','toaster','kettle','tumble dryer','electric space heater','computer','audio system','pond pump','television','fridge','microwave','fridge freezer','washing machine','appliance', 'fan']
all_seqlen = [102, 85, 171]
#########################################################
mode = 'train'  # train or test

mode_norm = 'unscale'

save = 'ON'  # ON for saving graph
#########################################################

experience_title, folder_location, graph_location, REFIT_path = paths()
reference, APPLIANCES_dict, ALL_APPLIANCES = dict_import(REFIT_path)
reference, APPLIANCES_dict = exceptions(reference, APPLIANCES_dict)  # remove corrupted data from the experience
dates = dates()
#time_periods = time_periods(date=date, weeks=weeks, bd_ids=reference['buildings_ids'], mode='fixed')
time_periods = specific_time(dates, reference['buildings_ids'])
print(ALL_APPLIANCES)

# APPLIANCES = APPLIANCES_dict['appliances']  #######

import_kwargs, seqlens = {'APPLIANCES_dict': APPLIANCES_dict, 'apps': APPLIANCES, 'path': REFIT_path,
                          'sample_s': sample_s, 'time_periods': time_periods, 'mode': 'auto', 'house_number': 2}, {}
buildings_ids = reference['buildings_ids']
data, bd_ids = select_data(**import_kwargs)
for k, appliance in enumerate(APPLIANCES):
    seqlens[appliance] = all_seqlen[k]

data, scalers = normalize_data(data, mode_norm)

import numpy as np

for appliance in APPLIANCES:
    for house in data[appliance].keys():
        test = np.array(data[appliance][house][appliance])
        mains = np.array(data[appliance][house]['mains'])

        plt.plot(test[:1000])
        plt.plot(mains[:1000])
        plt.title(appliance + ' ' + house)
        plt.legend([appliance, 'mains'])
        plt.show()

# si courbe discontinue alors 0 ? erreur possible dans le futur
# a faire: $eajouter les deux lignes de son code et trouver les bonnes fenêtres et normaliser