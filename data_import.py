import nilmtk
import random
import pandas as pd
from nilmtk import DataSet
from sklearn.preprocessing import StandardScaler, MinMaxScaler

nilmtk.Appliance.allow_synonyms = False


def paths(title):
    experience_title = title  # no "/"
    folder_location = '/Users/wilfriedgenuist/PycharmProjects/desagregation/project/MODELS/'  # neural model
    graph_location = '/Users/wilfriedgenuist/PycharmProjects/desagregation/project/output_results/'
    REFIT_path = '/Users/wilfriedgenuist/PycharmProjects/desagregation/datasets/REFIT/REFIT.h5'
    return experience_title, folder_location, graph_location, REFIT_path


def dict_import(path):
    ds, APPLIANCES_dict, app = DataSet(path), {}, []
    reference = {'buildings_ids': set([building_id for building_id in ds.buildings.keys()]), }
    for b_id in reference['buildings_ids']:
        elec, b = ds.buildings[b_id].elec, 'building' + str(b_id)
        reference[b] = [str(elec[i + 2].appliances).split("'")[1] for i in range(len(elec.appliances))]
        for name in reference[b]:
            app.append(name)
    app = list(set(app))
    APPLIANCES_dict['appliances'] = app
    for name in app:
        APPLIANCES_dict[name] = []
    for app_name in app:
        for ID in reference['buildings_ids']:
            if app_name in list(reference['building' + str(ID)]):
                APPLIANCES_dict[app_name].append(ID)
    ds.store.close()
    return reference, APPLIANCES_dict, APPLIANCES_dict['appliances']


def load_data(bd_ids, appliance, path, sample_s, time_period,  pmin=0, pmax= 12000):  # select all appliances power for each ID given
    data_loaded, ds = {}, DataSet(path)
    physical = ('power', 'active',)
    load_kwargs = {'physical_quantity': physical[0], 'ac_type': physical[1], 'sample_period': sample_s, }
    for b_id in bd_ids:
        print("Loading {}, building {}".format(ds.metadata['name'], b_id))
        ds.set_window(time_period[str(b_id)]["start_time"], time_period[str(b_id)]["end_time"])
        elec, b = ds.buildings[b_id].elec, 'building' + str(b_id)

        data_loaded[b], data_loaded[b]['mains'] = {}, next(elec.mains().load(**load_kwargs))
        clip_values(data_loaded[b]['mains'], pmin, pmax)

        appgen = elec[appliance].load(**load_kwargs)
        data_loaded[b][appliance] = next(appgen)
        clip_values(data_loaded[b][appliance], pmin, pmax)
    ds.store.close()
    return data_loaded


def select_data(APPLIANCES_dict, apps, path, sample_s, time_periods, mode='auto', house_number=100):
    assert mode in ['auto', 'select', None, ]
    data = {}
    if mode == 'auto':  # selecting all available houses
        for appliance in apps:
            bd_ids = APPLIANCES_dict[appliance]
            data[appliance] = load_data(bd_ids, appliance, path, sample_s, time_periods[appliance])
    if mode == 'select':  # selecting a finite number of houses
        for appliance in apps:
            random.shuffle(APPLIANCES_dict[appliance])
            bd_ids = APPLIANCES_dict[appliance][:house_number]
            data[appliance] = load_data(bd_ids, appliance, path, sample_s, time_periods[appliance])
    return data, bd_ids  # returns data as data['APPLIANCE!!!'] = {'buildingX': {'mains' : ..., 'appliance' : ...}, ...}


def clip_values(series, pmin, pmax):
    series[series <= pmin] = pmin
    series[series >= pmax] = pmax


def time_periods(date='2014-03-07', weeks=1, bd_ids=[1], mode='start_from_date'):
    assert mode in ['custom', 'start_from_date', None, ]
    time = {}
    if mode == 'custom' or None:
        time = {
            '1': {"start_time": "2014-03-07", "end_time": "2014-03-14"},
            '2': {"start_time": "2014-03-07", "end_time": "2014-03-14"},  # original
            '3': {"start_time": "2014-09-08", "end_time": "2014-09-15"},  # original
            '4': {"start_time": "2014-09-08", "end_time": "2014-09-15"},
            '5': {"start_time": "2013-11-15", "end_time": "2013-11-22"},  # original
            '6': {"start_time": "2014-11-12", "end_time": "2014-11-19"},  # original
            '7': {"start_time": "2014-05-07", "end_time": "2014-05-14"},  # original
            '8': {"start_time": "2014-05-07", "end_time": "2014-05-14"},
            '9': {"start_time": "2014-05-03", "end_time": "2014-05-10"},  # original
            '10': {"start_time": "2014-05-03", "end_time": "2014-05-10"},
            '11': {"start_time": "2014-05-03", "end_time": "2014-05-10"},
            '12': {"start_time": "2014-05-03", "end_time": "2014-05-10"},
            '13': {"start_time": "2014-03-08", "end_time": "2014-03-15"},  # original
            '14': {"start_time": "2014-03-08", "end_time": "2014-03-15"},
            '15': {"start_time": "2014-03-08", "end_time": "2014-03-15"},
            '16': {"start_time": "2014-03-08", "end_time": "2014-03-15"},
            '17': {"start_time": "2014-03-08", "end_time": "2014-03-15"},
            '18': {"start_time": "2014-03-08", "end_time": "2014-03-15"},
            '19': {"start_time": "2014-06-01", "end_time": "2014-06-08"},  # original
            '20': {"start_time": "2014-06-01", "end_time": "2014-06-08"},
            '21': {"start_time": "2014-06-01", "end_time": "2014-06-08"}, }

    if mode == 'start_from_date':  # be careful when selecting a date so that a month doesn't change between two dates
        date_s = date.split('-')
        day = int(date.split('-')[2]) + weeks * 7
        if day < 10:
            day = '0' + str(day)
        else:
            day = str(day)
        end_time = date_s[0] + '-' + date_s[1] + '-' + day
        for building_id in bd_ids:
            time[str(building_id)] = {"start_time": date, "end_time": end_time}
    return time


def specific_time(date, bd_ids):
    true_time_period = {}
    for appliance in date.keys():
        true_time_period[appliance] = time_periods(date[appliance], 1, bd_ids, 'start_from_date')
    return true_time_period  # time_period[appliance][bd_id](start-end_time)


def normalize_per_series(ts_dict, mode):
    scalers = {}
    transformed = {}
    if mode == 'standard':
        for key, data in ts_dict.items():
            zscale = StandardScaler().fit(data)
            if isinstance(data, pd.DataFrame):
                transformed[key] = pd.DataFrame(zscale.transform(data), index=data.index, columns=data.columns)
            else:
                transformed[key] = zscale.transform(data)
            scalers[key] = zscale
    if mode == 'min_max':
        for key, data in ts_dict.items():
            zscale = MinMaxScaler().fit(data)
            if isinstance(data, pd.DataFrame):
                transformed[key] = pd.DataFrame(zscale.transform(data), index=data.index, columns=data.columns)
            else:
                transformed[key] = zscale.transform(data)
            scalers[key] = zscale
    if mode == 'unscale':
        for key, data in ts_dict.items():
            zscale = 1
            if isinstance(data, pd.DataFrame):
                transformed[key] = pd.DataFrame(data, index=data.index, columns=data.columns)
            else:
                transformed[key] = data
            scalers[key] = zscale
    return transformed, scalers


def normalize_data(data, mode):
    assert mode in ['standard', 'min_max', 'unscale']
    new_data, scalers = {}, {}
    for appliance in data.keys():
        new_data[appliance], scalers[appliance] = {}, {}
        for house in data[appliance].keys():
            data[appliance][house] = pd.concat(data[appliance][house], axis=1)

            mains_data = data[appliance][house]['mains']
            all_data = data[appliance][house]
            app_data = data[appliance][house][appliance]

            all_data = all_data.copy().fillna(0)
            mains_data = mains_data.copy().fillna(0)
            app_data = app_data.copy().fillna(0)

            mains_data = pd.DataFrame(mains_data.sum(axis=1), columns=['mains'])
            aggregate = pd.DataFrame(all_data.sum(axis=1), columns=['mains'])
            app_data = pd.DataFrame(app_data.sum(axis=1), columns=[appliance])

            mains_data[mains_data < aggregate] = aggregate[mains_data < aggregate]
            new_data[appliance][house] = pd.concat([mains_data, app_data], axis=1)
            mean = new_data[appliance][house]['mains'].mean()//10

            new_data[appliance][house].loc[new_data[appliance][house]['mains'] < 1, 'mains'] = mean  # fix NAN values

        new_data[appliance], scalers[appliance] = normalize_per_series(new_data[appliance], mode)
    return new_data, scalers


def dates():
    date = {}
    base = '2014-03-07'
    date['broadband router'] = base
    date['games console'] = base
    date['dish washer'] = '2014-04-11'
    date['washer dryer'] = base
    date['unknown'] = '2014-06-07'
    date['freezer'] = '2014-06-07'
    date['breadmaker'] = base
    date['dehumidifier'] = base
    date['toaster'] = '2014-10-14'
    date['kettle'] = '2014-10-07'
    date['tumble dryer'] = '2014-10-07'
    date['electric space heater'] = '2014-07-07'
    date['computer'] = '2014-10-14'
    date['audio system'] = '2014-10-07'
    date['pond pump'] = '2014-03-07'
    date['television'] = '2014-11-10'
    date['fridge'] = '2014-08-13'
    date['microwave'] = '2014-10-07'
    date['fridge freezer'] = '2014-10-07'
    date['washing machine'] = '2014-12-10'
    date['appliance'] = '2014-04-03'
    date['fan'] = base
    return date


def remove(building, appliance, reference, APPLIANCES_dict):
    # 'building X'
    reference[building].remove(appliance)
    APPLIANCES_dict[appliance].remove(int(building.split('g')[1]))


def exceptions(reference, APPLIANCES_dict):

    remove('building11', 'washing machine', reference, APPLIANCES_dict)

    reference['building11'].remove('broadband router')
    reference['building11'].remove('dish washer')
    reference['building20'].remove('food processor')
    reference['building12'].remove('toaster')
    reference['building12'].remove('television')
    reference['building1'].remove('electric space heater')

    reference['building10'].remove('dish washer')

#    reference['building1'].remove('audio system')

    APPLIANCES_dict['broadband router'].remove(11)
    APPLIANCES_dict['food processor'].remove(20)
    APPLIANCES_dict['dish washer'].remove(11)
    APPLIANCES_dict['toaster'].remove(12)
    APPLIANCES_dict['electric space heater'].remove(1)

#    APPLIANCES_dict['washing machine'].remove(16)
    APPLIANCES_dict['dish washer'].remove(10)

#    APPLIANCES_dict['audio system'].remove(1)
    APPLIANCES_dict['television'].remove(12)
    return reference, APPLIANCES_dict
