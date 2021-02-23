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
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.linestring import LineString
from fiona.errors import DriverError
import logging
import re
import pathlib
from pypsa import Network
from _helpers import configure_logging
# snakemake is a global value snakemake.input.

logger = logging.getLogger(__name__)

#--------------------------------------------------------------------
# utility functions
#--------------------------------------------------------------------
def geo_concat(list_to_concat):
    # concat geodataframes in list_to_concat to one geodataframe
    return gpd.GeoDataFrame(pd.concat(list_to_concat, ignore_index=True)).set_crs('EPSG:4326')

def clean_dataframe(depth_df):
    depth_df.fillna(0, inplace=True)
    for column in depth_df.columns:
        if type(depth_df[column][0]) != str:
            depth_df[column] = depth_df[column].apply(
                lambda x: 0 if x > 1e5 else x)
    return depth_df

#--------------------------------------------------------------------
# functions for creating capacity per area density map from capacity
# per volume map
#--------------------------------------------------------------------
def create_nearshore_50_pair(on_path, off_path):
    onshore = gpd.read_file(on_path)
    offshore = gpd.read_file(off_path)
    offshore_buffer = offshore.to_crs(crs='EPSG:3395')
    offshore_buffer.geometry = offshore_buffer.buffer(50000)  # 50km buffer
    offshore_buffer = offshore_buffer.to_crs(crs='EPSG:4326')
    nearshore = gpd.clip(onshore, offshore_buffer)
    realonshore = gpd.overlay(onshore, offshore_buffer, how='difference')
    return nearshore, realonshore


def calculate_depth(mask, depth_type, capacity_path, density_map, capacity_name='val_kwhm3'):
    capacity = pd.read_csv(capacity_path)
    density_map = gpd.read_file(density_map)
    density_map.geometry = density_map.geometry.buffer(0)

    capacity_per_depth = []
    # calculate capacity_per_depth
    # clip denstiy_map with mask

    for country in mask.name:
        # calculate onshore capacity
        mask_country = mask[mask.name == country]
        part_map = gpd.clip(density_map, mask_country)
        area_square_meter = part_map.to_crs(crs='EPSG:3395').area.values
        # add to list onshore capacity
        capacity_per_depth.append(
            sum(part_map[capacity_name].values * area_square_meter))
    mask['capacity_per_depth'] = capacity_per_depth
    df_capacity_per_depth = mask[['name', 'capacity_per_depth']]

    capacity = capacity.merge(df_capacity_per_depth,
                              left_on='name', right_on='name', how='left')

    capacity[depth_type + '_depth'] = capacity[depth_type] / \
        capacity['capacity_per_depth']

    return capacity[['name', depth_type + '_depth']]


def get_capacity_per_area(density_map, mask_list, depth_list, capacity_name='val_kwhm3'):
    capacity_per_country = []
    for mask, depth, storage_type in zip(mask_list, depth_list, ['onshore', 'nearshore', 'offshore']):
        depth = clean_dataframe(depth)
        for country in depth.name:
            # calculate onshore capacity
            mask_country = mask[mask.name == country]
            part_map = gpd.clip(density_map, mask_country)
            # calculate capacity per country
            part_map['capacity_per_area'] = depth[depth.name == country].iloc[0, -1] \
                                            * part_map[capacity_name]
            part_map['storage_type'] = storage_type
            # add to list onshore capacity
            capacity_per_country.append(part_map)
    return geo_concat(capacity_per_country)


def calculate_capacity_per_area_shape(onshore, nearshore, realoffshore, density_map_path, capacity_path,
                             capacity_name='val_kwhm3'):
    density_map = gpd.read_file(density_map_path)
    density_map.geometry = density_map.geometry.buffer(0)

    # calculate depths of onshore, nearshore(50km), and offshore
    onshore_depth = calculate_depth(
        onshore, 'onshore', capacity_path, density_map_path)
    nearshore_depth = calculate_depth(
        nearshore, 'nearshore', capacity_path, density_map_path)
    offshore_depth = calculate_depth(
        realoffshore, 'total_offshore', capacity_path, density_map_path)

    mask_list = [onshore, nearshore, realoffshore]
    depth_list = [onshore_depth, nearshore_depth, offshore_depth]
    return get_capacity_per_area(density_map, mask_list, depth_list, capacity_name)

#--------------------------------------------------------------------
# only function below is important to assgin capacity to buses
#--------------------------------------------------------------------
def load_clusters_region(key_word,path2resources):
    # key_word:
    #   is the name of node clusters. e.g., elec_s_128
    # path2resources:
    #   is the path to resources folder. e.g., /resources
    _mask_list=[]
    _key_word=key_word.lower()
    p=pathlib.Path(path2resources)
    for file in p.iterdir():
        if re.search(_key_word+'*.geojson',str(file.absolute()).lower()):
            _mask_list.append(file)
            logger.info('find mask {}'.format(str(file.absolute())))
    return _mask_list

def add_storge2buses(buses,mask_list,capacity_shape,considered_locations):
    # region file shouldn't have ovelap
    # mask_list:
    #   is the return of load_clusters_region
    # capacity_shape:
    #   capacity density map (storage capacity per area)
    # considered_locations:
    #   which parts are needed to be considered: three possible choice: "onshore", "offshore","nearshore"
    #   can choose multiple choices at the same time. please input them as python list object
    #   e.g., ['onshore','offshore']
    _result = []
    _capacity_shape = capacity_shape[capacity_shape['storage_type'].isin(
        considered_locations)]
    for _mask in mask_list:
        mask_gdf = gpd.read_file(_mask)
        _capacity = []
        for row in range(len(mask_gdf)):
            part_map = gpd.clip(_capacity_shape,mask_gdf.iloc[row:row+1])
            _capacity.append(sum(part_map['capacity_per_area']*part_map.to_crs(crs='EPSG:3395').area))

        mask_gdf['hydrogen_storage_potential_twh'] = _capacity
        # kwh to twh
        mask_gdf['hydrogen_storage_potential_twh'] = mask_gdf['hydrogen_storage_potential_twh']/1e9
        _result.append(mask_gdf)

    _capacity_df = geo_concat(_result)
    _capacity_df = _capacity_df[['name','hydrogen_storage_potential_twh']]
    _capacity_df = _capacity_df.groupby('name').sum()
    _total_capacity = _capacity_df.hydrogen_storage_potential_twh.sum()
    logger.info('total capacity of nodes = {}'.format(_total_capacity))
    if _total_capacity>90000:
        logger.warning('very serious overlap in region clusters geojson file')
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
        density_map_path = "../data/bundle/salt_cave/final.shp"
        capacity_path = "../data/bundle/salt_cave/storage_potential_eu_kwh.csv"
        path2resources = "../resources"
        new_capacity_per_area_path = "../data/bundle/salt_cave/new_energy_capacity_map_kwhm2.geojson"

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
        density_map_path = snakemake.input.density_map_path
        capacity_path = snakemake.input.capacity_path
        path2resources = snakemake.input.path2resources
        new_capacity_per_area_path = "/data/bundle/salt_cave/new_energy_capacity_map_kwhm2.geojson"

        #config yaml
        config_dict= snakemake.config['hydrogen_storage']
        considered_locations = config_dict['considered_locations']

    # extract key word from network_input_path. example result: elec_s_128
    number_of_buses = network_input_path.split('/')[-1].split('.')[0]

    # load capacity density map (capacity per area)
    #   this block make sure capacity_shape will be loaded
    #-------------------------------------------------
    try:
        capacity_shape = gpd.read_file(new_capacity_per_area_path)
    except(DriverError):
        logging.info('no new_capacity_per_area file, start create it')
        # steps in except take around 2 minute for a normal PC
        offshore = gpd.read_file(offshore_path)
        nearshore, realonshore = create_nearshore_50_pair(
            onshore_path, offshore_path)
        capacity_shape = calculate_capacity_per_area_shape(
            realonshore, nearshore, offshore, density_map_path, capacity_path)
        capacity_shape.to_file(
            new_capacity_per_area_path, driver='GeoJSON')
        logging.info('successful create new_capacity_per_area file!')
    #--------------------------------------------------

    # load target network
    network = Network()
    network.import_from_netcdf(network_input_path)
    logging.info('assgin hydro storage capacity to {}'.format(network_input_path))

    # load region data of each bus
    mask_list = load_clusters_region(number_of_buses,path2resources)
    # assgin hydrogen storage capacity to each buses according to the region data of each bus
    new_buses = add_storge2buses(network.buses,mask_list,capacity_shape,considered_locations)
    # update target network buses
    network.buses = new_buses
    # export to a new network file
    network.export_to_netcdf(network_output_path)

    logging.info('assgin capacity task finishing successful')