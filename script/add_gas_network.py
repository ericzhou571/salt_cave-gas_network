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

import geopandas as gpd

#-----------------#
# utils functions #
#-----------------#
def concat_gdf(gdf_list, crs = 'EPSG:4326'):
    return gpd.GeoDataFrame(pd.concat(gdf_list),crs=crs)

def string2list(string, with_None=True):
    p = re.compile('(?<!\\\\)\'')
    string = p.sub('\"', string)

    if with_None:
        p2 = re.compile('None')
        string = p2.sub('\"None\"', string)

    return json.loads(string)


#-----------------#
#  main functions #
#-----------------#

def preprocessing(df_path):
    df = pd.read_csv(df_path, sep=';')

    df.long = df.long.apply(string2list)
    df.lat = df.lat.apply(string2list)
    df.node_id = df.node_id.apply(string2list)

    both_direct_df = df[df.is_bothDirection == 1].reset_index(drop=True)
    both_direct_df.node_id = both_direct_df.node_id.apply(lambda x: [x[1], x[0]])
    both_direct_df.long = both_direct_df.long.apply(lambda x: [x[1], x[0]])
    both_direct_df.lat = both_direct_df.lat.apply(lambda x: [x[1], x[0]])

    df_singledirect = pd.concat([df, both_direct_df]).reset_index(drop=True)
    df_singledirect.drop('is_bothDirection', axis=1)

    df['point1'] = df.apply(lambda x: Point((x['long'][0], x['lat'][0])), axis=1)
    df['point2'] = df.apply(lambda x: Point((x['long'][1], x['lat'][1])), axis=1)

    df['point1_name'] = df.node_id.str[0]
    df['point2_name'] = df.node_id.str[1]

    part1 = df[['point1', 'point1_name']]
    part2 = df[['point2', 'point2_name']]

    part1.columns = ['geometry', 'name']
    part2.columns = ['geometry', 'name']
    points = [part1, part2]

    points = concat_gdf(points)

    points = points.drop_duplicates()
    points.reset_index(drop=True, inplace=True)

    return df, points

def load_region(onshore_path, offshore_path):
    buses_region_offshore = gpd.read_file(offshore_path)
    buses_region_onshore = gpd.read_file(onshore_path)
    buses_region = concat_gdf([buses_region_offshore, buses_region_onshore])
    buses_region = buses_region.dissolve(by='name', aggfunc='sum')
    buses_region = buses_region.reset_index()

    return buses_region


def create_points2buses_map(input_points, buses_region):
    points = input_points.copy()
    points['buses'] = None
    buses_list = set(buses_region.name)
    for buses in buses_list:
        mask = buses_region[buses_region.name == buses]
        index = gpd.clip(points, mask).index
        if len(index) != 0:
            points.loc[index, 'buses'] = buses

    return points

def create_cross_regions_network(df,points2buses_map,output_path):
    # multi same direction pipes between two regions will combine to one pipe
    tmp_df = points2buses_map[['buses','name']]
    tmp_df.columns = ['buses_start','name']
    cross_buses_gas_network = df.merge(tmp_df,left_on='point1_name',right_on='name')
    tmp_df.columns = ['buses_destination', 'name']
    cross_buses_gas_network = cross_buses_gas_network.merge(tmp_df,left_on='point2_name',right_on='name')

    cross_buses_gas_network = cross_buses_gas_network[cross_buses_gas_network.buses_start \
                                                      != cross_buses_gas_network.buses_destination]
    cross_buses_gas_network.reset_index(drop=True,inplace=True)

    cross_buses_gas_network.drop(['point1','point2'],axis=1,inplace=True)

    cross_buses_gas_network.to_csv(output_path, sep=';', index=False)
    logger.info('sucessful store gas network to {}'.format(output_path))
    print('sucessful store gas network to {}'.format(output_path))



if __name__ == "__main__":

    if 'snakemake' not in globals():
        offshore_path = '../resources/regions_offshore_elec_s_37.geojson'
        onshore_path = '../resources/regions_onshore_elec_s_37.geojson'
        gas_network_path = '../resources/gas_network_IGGINLEE.csv'
        output_path = '../resources/gas_network_elec_s_37.csv'

    else:
        configure_logging(snakemake)
        offshore_path = snakemake.input.offshore_path
        onshore_path = snakemake.input.onshore_path
        gas_network_path = snakemake.input.gas_network_path
        output_path = snakemake.output.output_path

    gas_network, points = preprocessing(gas_network_path)
    buses_region = load_region(onshore_path, offshore_path)
    points2buses_map = create_points2buses_map(points, buses_region)
    create_cross_regions_network(gas_network, points2buses_map, output_path)
