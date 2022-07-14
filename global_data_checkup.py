from data_import import dict_import, paths, load_data, time_periods, dates, exceptions
import matplotlib.pyplot as plt


sample_s, title = 7, 'data_import'
experience_title, folder_location, graph_location, REFIT_path = paths(title)
reference, APPLIANCES_dict, ALL_APPLIANCES = dict_import(REFIT_path)
reference, APPLIANCES_dict = exceptions(reference, APPLIANCES_dict)

APPLIANCES = ['washing machine', 'tumble dryer', 'microwave', 'television', 'fan', 'dish washer', 'audio system',
              'kettle', 'electric space heater', 'food processor', 'games console', 'washer dryer',
              'fridge', 'pond pump', 'dehumidifier', 'freezer', 'breadmaker', 'broadband router',
              'fridge freezer', 'toaster', 'computer']

appliance = 'washing machine'
dates = dates()

time_period = time_periods(date=dates[appliance], weeks=1, bd_ids=APPLIANCES_dict[appliance], mode='start_from_date')

data = load_data(APPLIANCES_dict[appliance], appliance, REFIT_path, sample_s, time_period)

for building in list(data.keys()):
    plt.plot(data[building]['mains'].values)
    plt.plot(data[building][appliance].values)
    plt.legend(['mains', appliance])
    plt.title(building)
    plt.show()

print('---end---')
# attention dans le programme exceptions, il faut tout changer aussi !
