# SPDX-FileCopyrightText: : 2020 @JanFrederickUnnewehr, The PyPSA-Eur Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""

This rule downloads the load data from `Open Power System Data Time series <https://data.open-power-system-data.org/time_series/>`_. For all countries in the network, the per country load timeseries with suffix ``_load_actual_entsoe_transparency`` are extracted from the dataset. After filling small gaps linearly and large gaps by copying time-slice of a given period, the load data is exported to a ``.csv`` file.

Relevant Settings
-----------------

.. code:: yaml

    snapshots:

    load:
        url:
        interpolate_limit:
        time_shift_for_large_gaps:
        manual_adjustments:


.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`load_cf`

Inputs
------


Outputs
-------

- ``resource/time_series_60min_singleindex_filtered.csv``:


"""

import logging
logger = logging.getLogger(__name__)
from _helpers import configure_logging

import re
import pandas as pd
import numpy as np
import json

from shapely.geometry import LineString,Point

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error

#-----------------#
# utils functions #
#-----------------#
def unpack_param(df):
    if 'param' in df.columns:
        new = df.param.apply(string2list)
        df.lat=df.lat.apply(string2list)
        df.long=df.long.apply(string2list)
        return df.merge(pd.DataFrame(list(new)), left_index=True, right_index=True)
    else:
        return 0

def string2list(string, with_None=True):
    p = re.compile('(?<!\\\\)\'')
    string = p.sub('\"', string)

    if with_None:
        p2 = re.compile('None')
        string = p2.sub('\"None\"', string)

    return json.loads(string)

#use to create geo object
def change2linestring(df):
    # rows level process
    df['linestring']=[]
    for index in range(len(df['lat'])):
        df['linestring'].append((df['long'][index],df['lat'][index]))
    df['linestring']=LineString(df['linestring'])
    return df

def addLinestring(df):
    #dataframe level process
    df=df.reset_index(drop=True)
    df['linestring']='L'
    df=df.apply(change2linestring,axis=1)
    return df

#-----------------#
#  main functions #
#-----------------#

def load_preprocessing_dataset(IGGINL_df_path, entsog_df_path, EMAP_df_path):

    #load&prepocess IGGINL df
    IGGINL = pd.read_csv(IGGINL_df_path, sep=';')
    IGGINL = unpack_param(IGGINL)
    IGGINL['cap_uncertain'] = IGGINL.uncertainty.apply(lambda x: string2list(x)['max_cap_M_m3_per_d'])
    IGGINL['diameter_uncertain'] = IGGINL.uncertainty.apply(lambda x: string2list(x)['diameter_mm'])
    IGGINL['pressure_uncertain'] = IGGINL.uncertainty.apply(lambda x: string2list(x)['max_pressure_bar'])

    IGGINL['capacity_nan'] = IGGINL['max_cap_M_m3_per_d']
    IGGINL['diameter_nan'] = IGGINL['diameter_mm']
    IGGINL['pressure_nan'] = IGGINL['max_pressure_bar']

    IGGINL.loc[IGGINL[IGGINL['cap_uncertain'] > 0].index, 'capacity_nan'] = np.nan
    IGGINL.loc[IGGINL[IGGINL['diameter_uncertain'] > 0].index, 'diameter_nan'] = np.nan
    IGGINL.loc[IGGINL[IGGINL['pressure_uncertain'] > 0].index, 'pressure_nan'] = np.nan
    # add from to
    IGGINL.country_code = IGGINL.country_code.apply(string2list)
    IGGINL['from'] = IGGINL.country_code.str[0]
    IGGINL['to'] = IGGINL.country_code.str[1]
    # deal with whitespace
    IGGINL['from'] = IGGINL['from'].str.strip()
    IGGINL['to'] = IGGINL['to'].str.strip()
    # convert capacity
    IGGINL['max_capacity'] = IGGINL.max_cap_M_m3_per_d * 35.8 / 3.6 / 24  # gwh/h
    IGGINL.capacity_nan = IGGINL.capacity_nan * 35.8 / 3.6 / 24
    # add line string object
    IGGINL = addLinestring(IGGINL)
    # create new attributes
    IGGINL['From'] = IGGINL.country_code.str[0]
    IGGINL['To'] = IGGINL.country_code.str[1]
    try:
        IGGINL.node_id = IGGINL.node_id.apply(string2list)
    except:
        pass
    IGGINL['Node_0'] = IGGINL.node_id.str[0]
    IGGINL['Node_1'] = IGGINL.node_id.str[1]

    #load&preprocess entsog_df
    entsog_dataset = pd.read_csv(entsog_df_path)  # gwh/h
    entsog_dataset.columns = ['Point', 'Capacity', 'From_ID', 'To_ID', 'From', 'To', 'long', 'lat']
    entsog_dataset.drop(['From_ID', 'To_ID'], axis=1, inplace=True)
    entsog_dataset_wrapping = entsog_dataset.groupby(['long', 'lat', 'From', 'To']).max()['Capacity'].reset_index()
    #add shapely object
    entsog_dataset_wrapping['Point'] = entsog_dataset_wrapping.apply(lambda x: Point(x['long'], x['lat']), axis=1)

    #load&preprocess EMAP_df
    EMAP_Raw = unpack_param(pd.read_csv(EMAP_df_path, sep=';'))
    EMAP_Raw.pipe_class_EMap = EMAP_Raw.pipe_class_EMap.fillna('A')
    logger.info('finish loading')
    return IGGINL, entsog_dataset_wrapping, EMAP_Raw


def add_entsog_capacity(IGGINL_df, entsog_dataset_wrapping,
                        new_capacity_name='entsog_capacity_withdirection',
                        how='direct'):
    '''
    must add shapely object before

    '''
    IGGINL_df = IGGINL_df.copy()

    IGGINL_df[new_capacity_name] = 0.0
    IGGINL_df['distance_to_capacity_point'] = 10e10

    for i in range(len(entsog_dataset_wrapping)):
        min_distance = 10e10
        min_line_number = 0
        for k in range(len(IGGINL_df)):
            # calculate matching criterion
            distance = entsog_dataset_wrapping.Point[i].distance(IGGINL_df.linestring[k])

            if distance < 0.5 and IGGINL_df['distance_to_capacity_point'][k] > distance:
                if how == 'undirect':
                    IGGINL_df.at[k, new_capacity_name] = entsog_dataset_wrapping.Capacity[i]
                    IGGINL_df.at[k, 'distance_to_capacity_point'] = distance

                else:
                    # direct model
                    if IGGINL_df['is_bothDirection'][k] == 1:
                        IGGINL_df.at[k, new_capacity_name] = entsog_dataset_wrapping.Capacity[i]
                        IGGINL_df.at[k, 'distance_to_capacity_point'] = distance
                    else:
                        if IGGINL_df['From'][k] == entsog_dataset_wrapping['From'][i] and IGGINL_df['To'][k] == \
                                entsog_dataset_wrapping['To'][i]:
                            IGGINL_df.at[k, new_capacity_name] = entsog_dataset_wrapping.Capacity[i]
                            IGGINL_df.at[k, 'distance_to_capacity_point'] = distance

    # replace 0 in the new capacity column with np.nan
    IGGINL_df.entsog_capacity_withdirection = IGGINL_df.entsog_capacity_withdirection.apply(
        lambda x: np.nan if x == 0 else x)
    IGGINL_df.capacity_nan = IGGINL_df.capacity_nan.fillna(IGGINL_df.entsog_capacity_withdirection)
    logger.info('after adding entsog_2019 dataset, capactiy of {} rows are still missing capacity '\
                .format(IGGINL_df.capacity_nan.isna().sum()))
    logger.info('finish adding entsog_dataset')
    return IGGINL_df


def add_EMAP_diameter(IGGINL_df, EMAP_Raw, buffers=0.5):
    # calculate mean value of each class with original diameter data
    IGGINL_df_not_nan = IGGINL_df[IGGINL_df.diameter_nan.notna()].reset_index(drop=True)
    IGGINL_df_mean_diameter_s = IGGINL_df_not_nan[IGGINL_df_not_nan.diameter_nan < 600]['diameter_nan'].mean()
    IGGINL_df_mean_diameter_m = \
    IGGINL_df_not_nan[(IGGINL_df_not_nan.diameter_nan >= 600) & (IGGINL_df_not_nan.diameter_nan < 900)][
        'diameter_nan'].mean()
    IGGINL_df_mean_diameter_l = IGGINL_df_not_nan[IGGINL_df_not_nan.diameter_nan >= 900]['diameter_nan'].mean()
    class_dict = {'S': IGGINL_df_mean_diameter_s, 'M': IGGINL_df_mean_diameter_m, 'L': IGGINL_df_mean_diameter_l}

    IGGINL_df['EMAP_Class'] = 0

    # filter on EMAP, length>50, only keep S M L
    EMAP_Raw = EMAP_Raw[EMAP_Raw.length_km > 50]
    EMAP_Raw = EMAP_Raw[EMAP_Raw.pipe_class_EMap.isin(['S', 'M', 'L'])]
    EMAP_Raw = EMAP_Raw.reset_index(drop=True)

    # start matching
    IGGINL_df = addLinestring(IGGINL_df.copy())
    EMAP_Raw = addLinestring(EMAP_Raw.copy())
    IGGINL_df.linestring = IGGINL_df.linestring.apply(lambda x: x.buffer(buffers))
    IGGINL_df = IGGINL_df.apply(lambda x: match(x, EMAP_Raw, class_dict), axis=1)

    # fill original pipe diameter
    IGGINL_df.EMAP_Class = IGGINL_df.EMAP_Class.apply(lambda x: np.nan if x == 0 else x)
    IGGINL_df.diameter_nan = IGGINL_df.diameter_nan.fillna(IGGINL_df.EMAP_Class)
    logger.info('after adding EMAP dataset, capactiy of {} rows are still missing diameter ' \
                .format(IGGINL_df.diameter_nan.isna().sum()))
    logger.info('finish adding EMAP diameter')
    return IGGINL_df


def match(series, EMAP_Raw, class_dict={'S': 400, 'M': 700, 'L': 1000}):
    # use on IGGINL_df
    if np.isnan(series['capacity_nan']) and np.isnan(series['diameter_nan']):
        for i in range(len(EMAP_Raw)):
            if series['linestring'].contains(EMAP_Raw.iloc[i]['linestring']):
                # if match with several pipe in EMAP, choose the bIGGINL_dfest pipe class
                series['EMAP_Class'] = max(series['EMAP_Class'], class_dict[EMAP_Raw.iloc[i]['pipe_class_EMap']])
    return series


def train_lasso(IGG):
    # need use original data
    # ------------preprocessing----------------
    # find all pipe that have diameter data and capacity data
    train_data = IGG[IGG.diameter_nan.notna() & IGG.capacity_nan.notna() & IGG.pressure_nan.notna()]
    train_data = train_data.reset_index(drop=True)

    # add squared diameter_nan
    train_data['diameter_nan_2'] = train_data.diameter_nan * train_data.diameter_nan
    train_data_predicted = train_data

    # sort values, only helps when plotting
    train_data = train_data.sort_values('diameter_nan')

    # -------------start training--------------
    # train two model and return two model
    # (finally use normal lasso , becasue MAE of two models are really close, then choose simpler one)

    # train normal
    rg_model_normal = Lasso(alpha=0.001)
    rg_model_normal.fit(train_data.diameter_nan.values.reshape(-1, 1), train_data.capacity_nan)
    train_data['predict_normal'] = rg_model_normal.predict(train_data.diameter_nan.values.reshape(-1, 1))

    # calculate error
    MAE_normal = str(round(mean_absolute_error(train_data.capacity_nan, train_data.predict_normal), 3))

    logger.info('sucessful training lasso regression, MAE = {}' \
                .format(MAE_normal))

    # here will not show the plot, plot will be return
    logger.info('finish training lasso')
    return rg_model_normal

def filling_with_lasso(IGGINL_df, regression_model):
    # if diameter_nan of a pipe is still missing, use diameter data from diameter_mm of the pipe
    #IGGINL_df.diameter_nan = IGGINL_df.diameter_nan.fillna(IGGINL_df.diameter_mm)
    minimum_value = IGGINL_df.capacity_nan.min()

    IGGINL_df.capacity_nan = IGGINL_df.apply(
        lambda x: regression_model.predict(np.array([x['diameter_nan']]).reshape(-1, 1))[0]
        if (np.isnan(x['capacity_nan'])) & (not np.isnan(x['diameter_nan'])) else x['capacity_nan'], axis=1)

    #remove extremely small value
    IGGINL_df.capacity_nan = IGGINL_df.capacity_nan.apply(lambda x: np.nan if x < minimum_value else x)
    logger.info('finish filling with lasso')
    return IGGINL_df

def node_capacity_spread(df):
    iteration = 0
    while df.capacity_nan.isna().sum() > 0:
        capacity = node_capacity_update(df, how='max')
        df = df.apply(lambda x: add_capacity(capacity, x), axis=1)
        iteration += 1
        if iteration >= 20:
            logger.info('can\'t filling all missing capacity in 20 round\
                        still have {} missing value'.format(df.capacity_nan.isna().sum()))
            logger.info('fill rest missing capacity with capacity in max_capacity ')
            df.capacity_nan = df.capacity_nan.fillna(df['max_capacity'])
            return df
    logger.info('finish spreading')
    return df

def clean_save(df,output_path):
    Final_dataset = df[['id', 'name', 'source_id', 'node_id', 'lat', 'long',
                                   'country_code', 'is_bothDirection', 'lat_mean',
                                   'length_km', 'long_mean', 'max_pressure_bar',
                                   'num_compressor', 'start_year', 'capacity_nan', 'diameter_nan']]

    # rename columns
    Final_dataset.columns = ['id', 'name', 'source_id', 'node_id', 'lat', 'long',
                             'country_code', 'is_bothDirection', 'lat_mean',
                             'length_km', 'long_mean', 'max_pressure_bar',
                             'num_compressor', 'start_year', 'Capacity_GWh_h', 'diameter_mm',]
    Final_dataset.to_csv(output_path, sep=';', index=False)
    logger.info('sucessfully saving dataset to {}'.format(output_path))

def node_capacity_update(df, how='max'):
    df = df[df.capacity_nan.notna()].reset_index(drop=True)
    node_capacity = pd.concat([df[['Node_0', 'capacity_nan']], df[['Node_1', 'capacity_nan']]])
    node_capacity.fillna('NULL', inplace=True)
    node_capacity.Node_0 = node_capacity.apply(lambda x: x['Node_1'] if x['Node_0'] == 'NULL' else x['Node_0'], axis=1)
    if how == 'sum':
        node_capacity = node_capacity.groupby('Node_0').sum()['capacity_nan'].reset_index()
    elif how == 'mean':
        node_capacity = node_capacity.groupby('Node_0').mean()['capacity_nan'].reset_index()
    elif how == 'max':
        node_capacity = node_capacity.groupby('Node_0').max()['capacity_nan'].reset_index()

    node_capacity = node_capacity[node_capacity.capacity_nan > 0]
    node_capacity.set_index('Node_0', inplace=True)
    return node_capacity


def add_capacity(node_capacity, df):
    if not np.isnan(df['capacity_nan']):
        return df
    else:
        try:
            df['capacity_nan'] = node_capacity.loc[df['Node_0']]['capacity_nan']
            return df
        except KeyError:
            try:
                df['capacity_nan'] = node_capacity.loc[df['Node_1']]['capacity_nan']
                return df
            except KeyError:
                return df



if __name__ == "__main__":

    if 'snakemake' not in globals():
        IGGINL_path = '../data/bundle/gas_network/IGGINL_PipeSegments.csv'
        entsog_2019_path = '../data/bundle/gas_network/entsog_2019_dataset.csv'
        EMAP_path = '../data/bundle/gas_network/EMAP_Raw_PipeSegments.csv'
        output_path = '../resources/gas_network_IGGINLEE.csv'

    else:
        configure_logging(snakemake)
        IGGINL_path = snakemake.input.igginl_path
        entsog_2019_path = snakemake.input.entsog_2019_path
        EMAP_path = snakemake.input.emap_path
        output_path = snakemake.output.output_path

    IGGINL, entsog_dataset, EMAP = load_preprocessing_dataset(IGGINL_path,entsog_2019_path, EMAP_path)
    regression_model = train_lasso(IGGINL)
    IGGINL = add_entsog_capacity(IGGINL, entsog_dataset)
    IGGINL = add_EMAP_diameter(IGGINL,EMAP)
    IGGINL = filling_with_lasso(IGGINL, regression_model)
    IGGINL = node_capacity_spread(IGGINL)
    clean_save(IGGINL, output_path)

