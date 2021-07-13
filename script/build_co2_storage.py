# SPDX-FileCopyrightText: : 2017-2020 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Creates hydrogen storage capacity geojson file based on input buses strcuture

Relevant Settings
-----------------


Inputs
------

- ``resources/country_shapes.geojson``: country shapes out of country selection

    .. image:: ../img/country_shapes.png
        :scale: 33 %

- ``resources/offshore_shapes.geojson``: EEZ shapes out of country selection

    .. image:: ../img/offshore_shapes.png
        :scale: 33 %

- ``data/salt_cave/data/result/final.shp``: europe salt cave potential hydrogen storage capacity density shape file

- ``data/salt_cave/data/result/storage_potential_eu_kwh.csv``: europe salt cave potential hydrogen storage capacity per country

- ``data/salt_cave/data/result/new_energy_capacity_map_pwh.geojson``: europe salt cave potential hydrogen storage capacity shape file

- ``networks/xxxx``: buses structure

Outputs
-------
- ``resources/nodes_capacity_map_pwh.geojson``: buses' potential hydrogen storage capacity


Description
-----------

"""

import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import re
import pathlib
from pypsa import Network
#from _helpers import configure_logging
# snakemake is a global value snakemake.input.

logger = logging.getLogger(__name__)

#--------------------------------------------------------------------
# utility functions
#--------------------------------------------------------------------
def geo_concat(list_to_concat):
    # function: concat geodataframes in list_to_concat to one geodataframe
    # input: list of geodataframes
    # output: gpd.geodataframe
    return gpd.GeoDataFrame(pd.concat(list_to_concat, ignore_index=True)).set_crs('EPSG:4326')

def clean_dataframe(depth_df):
    # function: fill missing value np.nan with 0; remove inf
    # input: pd.DataFrame object
    # ouput: pd.DataFrame object
    depth_df.fillna(0, inplace=True)
    for column in depth_df.columns:
        if type(depth_df[column][0]) != str:
            depth_df[column] = depth_df[column].apply(
                lambda x: 0 if x > 1e5 else x)
    return depth_df

#--------------------------------------------------------------------
# important to assgin capacity to buses
#--------------------------------------------------------------------
def load_clusters_region(key_word,path2resources):
    '''
    load the nodes' area divisions

    Parameter
    ----------
    key_word: str
        is the name of node clusters. e.g., elec_s_128

    path2resources: str
        is the path to resources folder. e.g., /resources

    Return
    ------------
    _mask_list: list object
        a list of geodataframes, which will be used in add_storage2buses
    '''
    _mask_list=[]
    _key_word=key_word.lower()
    p=pathlib.Path(path2resources)
    for file in p.iterdir():
        if re.search(_key_word+'*.geojson',str(file.absolute()).lower()):
            _mask_list.append(file)
            logger.info('find mask {}'.format(str(file.absolute())))
    return _mask_list

def add_storge2buses(buses,mask_list,capacity_map_path,
                     conservative_capacity_name = 'conservative estimate Mt',
                     neutral_capacity_name = 'neutral estimate Mt',
                     capacity_name = 'co2_storage_Mt'):
    '''
    load the nodes' area divisions

    Parameter
    ----------
    buses: pypsa.Network().buses
        the csv table contains nodes to which we need assign hydrogen storage capacity
        nodes in buses must be the same as nodes in parameter mask_list

    mask_list: list
        a list of geodataframes, which will be used in add_storage2buses
        it need has same nodes as buses

    capacity_shape: gpd.geodataframe
        capacity density map (storage capacity per area)

    considered_locations: list
        which parts are needed to be considered: three possible choice: "onshore",
        "offshore","nearshore" can choose multiple choices at the same time.
        please input them as python list object. e.g., ['onshore','offshore']


    Return
    ------------
    buses with hydrogen storage capacity: pypsa.Network().buses
    '''
    _result = []
    _capacity_shape = gpd.read_file(capacity_map_path)

    # calculate area of polygons and store in new column named "area"
    _capacity_shape['area'] = _capacity_shape.to_crs(crs='EPSG:3395').area

    # filling missing value in conservative estimation with value in neutral estimation
    # before filling capacity EU total: 145Gt, after: 289Gt,
    # but it still much better than pure neutral: 1000+Gt
    _capacity_shape[conservative_capacity_name].replace(0, np.nan, inplace=True)
    _capacity_shape[conservative_capacity_name].fillna(_capacity_shape[neutral_capacity_name], inplace=True)

    for _mask in mask_list:
        mask_gdf = gpd.read_file(_mask)
        _capacity = []
        for row in range(len(mask_gdf)):
            part_map = gpd.clip(_capacity_shape,mask_gdf.iloc[row:row+1])
            # divide capacity
            # we suppose co2 storage distribute evenly in the polygon
            _capacity.append(sum(part_map[conservative_capacity_name]
                                 /part_map['area']
                                 *part_map.to_crs(crs='EPSG:3395').area))

        mask_gdf[conservative_capacity_name] = _capacity
        mask_gdf[conservative_capacity_name] = mask_gdf[conservative_capacity_name]/1e3
        _result.append(mask_gdf)

    _capacity_df = geo_concat(_result)
    # rename the capacity name with value of argument "capacity_name"
    _capacity_df = _capacity_df[['name',capacity_name]]
    _capacity_df = _capacity_df.groupby('name').sum()
    _total_capacity = _capacity_df[capacity_name].sum()
    logger.info('total capacity of nodes = {:.2e} mwh'.format(_total_capacity))

    return buses.merge(_capacity_df,left_index = True,right_index = True, how='left')


if __name__ == "__main__":
    if 'snakemake' not in globals():
        #---------------------------#
        #       test version        #
        #---------------------------#

        # logging configuration
        logging.basicConfig(filename="../logs/build_hydrogen_storage.log", level=logging.INFO)
        logging.info('test enviroment, not using snakemake')
        logging.info('start assgining hydrogen capacity potential to buses')

        # snakemake input
        onshore_path = "../resources/country_shapes.geojson"
        offshore_path = "../resources/offshore_shapes.geojson"
        network_input_path = "../networks/elec_s_256.nc"
        network_output_path = "../networks/with_hydrogen_stoage_elec_s_256.nc"
        path2resources = "../resources"
        capacity_map_path = "../resources/new_energy_capacity_map_kwhm2.geojson"

        # config yaml
        considered_locations = ['onshore','nearshore','offshore']
                               # ['onshore','nearshore','offshore']
    else:
        #---------------------------#
        #     snakemake version     #
        #---------------------------#
        configure_logging(snakemake)

        logging.info('start assgining hydrogen capacity potential to buses')
        #snakemake input
        onshore_path = snakemake.input.onshore_path
        offshore_path = snakemake.input.offshore_path
        network_input_path = snakemake.input.networks
        network_output_path = snakemake.output.output_path
        path2resources = snakemake.input.path2resources
        capacity_map_path = snakemake.input.capacity_map_path

        #config yaml
        config_dict= snakemake.config['hydrogen_storage']
        considered_locations = config_dict['considered_locations']

    # extract key word from network_input_path. example result: elec_s_128
    number_of_buses = network_input_path.split('/')[-1].split('.')[0]

    #load density map
    capacity_shape = gpd.read_file(new_capacity_per_area_path)

    # load target network
    network = Network()
    network.import_from_netcdf(network_input_path)
    logging.info('assgin hydro storage capacity to {}'.format(network_input_path))

    # assgin onshore co2 storage capacity to each buses
    # according to the onshore region data of each bus

    new_buses = add_storge2buses(buses = network.buses,
                                 mask_list = [onshore_path],
                                 capacity_map_path = capacity_map_path,
                                 conservative_capacity_name='conservative estimate Mt',
                                 neutral_capacity_name='neutral estimate Mt',
                                 capacity_name='onshore_co2_storage_Mt')

    # assgin offshore co2 storage capacity to each buses
    # according to the offshore region data of each bus
    new_buses = add_storge2buses(buses = new_buses,
                                 mask_list = [offshore_path],
                                 capacity_map_path = capacity_map_path,
                                 conservative_capacity_name='conservative estimate Mt',
                                 neutral_capacity_name='neutral estimate Mt',
                                 capacity_name='onshore_co2_storage_Mt')


    # update target network buses
    network.buses = new_buses
    # export to a new network file
    network.export_to_netcdf(network_output_path)

    logging.info('assgin capacity task finishing successful')