import geopandas as gpd
import pandas as pd
import fiona
import rasterio
import rasterio.features
from shapely.geometry import shape, mapping
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.linestring import LineString
import time
from shapely.geometry import Point

# snakemake is a global value snakemake.input.




def geo_concat(list_to_concat):
    '''
        concat geodataframes in list_to_concat to one geodataframe

        Parameters
        -----------

        list_to_concat : list
            a list of geodataframes

        Return
        ------------
        eu : geopandas.GeoDataFrame
            the result with capacity/depth
            unit: kwh/m per country
        '''
    return gpd.GeoDataFrame(pd.concat(list_to_concat, ignore_index=True)).set_crs('EPSG:4326')


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
    density_map=explode(density_map)
    density_map=density_map.set_crs(crs='EPSG:4326')

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

    capacity = capacity.merge(df_capacity_per_depth, left_on='name', right_on='name', how='left')

    capacity[depth_type + '_depth'] = capacity[depth_type] / capacity['capacity_per_depth']

    return capacity[['name', depth_type + '_depth']]


def get_capacity(capacity, density_map, mask_list, depth_list, capacity_name='val_kwhm3'):
    capacity_per_country = []
    for mask, depth, storage_type in zip(mask_list, depth_list, ['onshore', 'nearshore', 'offshore']):
        depth = clean_dataframe(depth)
        for country in depth.name:
            # calculate onshore capacity
            mask_country = mask[mask.name == country]
            part_map = gpd.clip(density_map, mask_country)
            # calculate capacity per country
            part_map['capacity'] = part_map.to_crs(crs='EPSG:3395').area * depth[depth.name == country].iloc[0, -1] \
                                   * part_map[capacity_name]
            part_map['storage_type'] = storage_type
            # add to list onshore capacity
            capacity_per_country.append(part_map)
    return geo_concat(capacity_per_country)


def calculate_capacity_shape(onshore, nearshore, realoffshore, density_map_path, capacity_path,
                             capacity_name='val_kwhm3'):
    capacity = pd.read_csv(capacity_path)
    density_map = gpd.read_file(density_map_path)
    density_map.geometry = density_map.geometry.buffer(0)

    # calculate depths of onshore, nearshore(50km), and offshore
    onshore_depth = calculate_depth(onshore, 'onshore', capacity_path, density_map_path)
    nearshore_depth = calculate_depth(nearshore, 'nearshore', capacity_path, density_map_path)
    offshore_depth = calculate_depth(realoffshore, 'total_offshore', capacity_path, density_map_path)

    mask_list = [onshore, nearshore, realoffshore]
    depth_list = [onshore_depth, nearshore_depth, offshore_depth]
    return get_capacity(capacity, density_map, mask_list, depth_list, capacity_name)


def clean_dataframe(depth_df):
    depth_df.fillna(0, inplace=True)
    for column in depth_df.columns:
        if type(depth_df[column][0]) != str:
            depth_df[column] = depth_df[column].apply(lambda x: 0 if x > 1e5 else x)
    return depth_df


def explode(indata):
    '''
    This function split multipolygons to polygons

    Parameters
    ----------
    indata : geopandas.GeoDataFrame
        geometry of indata is multipolygons

    Return
    ----------
    outdf : geopandas.GeoDataFrame
        geometry of outdf is polygons
    '''
    indf = indata
    outdf = gpd.GeoDataFrame(columns=indf.columns)
    for idx, row in indf.iterrows():
        if type(row.geometry) == Polygon:
            outdf = outdf.append(row, ignore_index=True)
        if type(row.geometry) == MultiPolygon:
            multdf = gpd.GeoDataFrame(columns=indf.columns)
            recs = len(row.geometry)
            multdf = multdf.append([row] * recs, ignore_index=True)
            for geom in range(recs):
                multdf.loc[geom, 'geometry'] = row.geometry[geom]
            outdf = outdf.append(multdf, ignore_index=True)
    return outdf


def assgin_capacity2node(nodes, country_path, offshore_path, capacity_path,
                         considered_locations=['onshore', 'nearshore', 'offshore']):
    capacity_df = gpd.read_file(capacity_path)
    onshore = gpd.read_file(country_path)
    offshore = gpd.read_file(offshore_path)
    nodes = nodes.set_crs('EPSG:4326')
    lines = []

    rest_result = capacity_df[capacity_df['storage_type'].isin(considered_locations)]

    full_table = []
    for country in onshore['name']:
        try:
            mask = gpd.GeoDataFrame(pd.concat([onshore[onshore['name'] == country],
                                                     offshore[offshore['name'] == country]], ignore_index=True))
            mask = mask.set_crs('EPSG:4326')
        except:
            mask = onshore[onshore['name'] == country]

        part_result = gpd.clip(rest_result, mask)
        #
        rest_result = rest_result[~rest_result.index.isin(part_result.index)]
        part_result = part_result.reset_index()
        part_nodes = gpd.clip(nodes, mask).reset_index()

        for i in range(len(part_result)):
            inx = part_nodes.geometry.apply(lambda x: x.distance(part_result.loc[i, 'geometry'])) \
                .argmin()
            part_nodes.loc[inx, 'capacity'] += part_result.loc[i, 'capacity']
            lines.append(LineString([part_nodes.loc[inx, 'geometry'], part_result.loc[i, 'geometry'].centroid]))
        full_table.append(part_nodes)

        # verify total capacity
    capacity_per_node = gpd.GeoDataFrame(pd.concat(full_table, ignore_index=True))
    capacity_per_node = capacity_per_node.set_crs('EPSG:4326')

    return capacity_per_node

if __name__ == "__main__":
    onshore_path = 'resources/country_shapes.geojson'
    offshore_path = 'resources/offshore_shapes.geojson'

    density_map_path = 'result/final.shp'
    capacity_path = 'result/storage_potential_eu_kwh.csv'

    offshore = gpd.read_file(offshore_path)
    nearshore, realonshore = create_nearshore_50_pair(onshore_path, offshore_path)
    new_capacity_map = calculate_capacity_shape(realonshore, nearshore, offshore, density_map_path, capacity_path)
    new_capacity_map.capacity=new_capacity_map.capacity/1e12
    new_capacity_map.to_file(
        'result/new_energy_capacity_map_pwh.geojson', driver='GeoJSON')

    nodes = pd.read_csv('example_nodes_45.csv')

    start= time.time()
    # convert lisa's data to GeoDataFrame
    capacity_per_node = gpd.GeoDataFrame(
        nodes, geometry=gpd.points_from_xy(nodes['x'], nodes['y']))

    # init capacity of each node with 0
    capacity_per_node['capacity'] = 0
    capacity_per_node=capacity_per_node[['name','capacity', 'geometry']]

    node_capacity=assgin_capacity2node(capacity_per_node,
                                       onshore_path,
                                       offshore_path,
                                       'result/new_energy_capacity_map_pwh.geojson',
                                       ['onshore', 'nearshore', 'offshore'])

    node_capacity.to_file(
        'result/nodes_capacity_map_pwh.geojson', driver='GeoJSON')
    print(time.time()-start)