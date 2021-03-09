import json
import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
from osgeo import gdal, osr
import geopandas as gpd
import fiona
import rasterio
import rasterio.features
from shapely.geometry import shape, mapping
from shapely.geometry.multipolygon import MultiPolygon
import logging
from fiona.errors import DriverError
import os
import shutil

logger = logging.getLogger(__name__)


def extract_with_threshold(color, mask, threshold=10, i=None, save_mode=False):
    '''
    extract capacity layer based on given color

    Parameter
    ----------
    threshold : float, default = 10
        threshold is variable use to juge a pixel in original raster
        whether it is background pixel or not
        The way of judgment is to calculate the veriance of a pixel's
        three color channels. Based on observations, background pixel
        in RGB color space always have a very low veriance (like black
        [0,0,0], white [255,255,255], grey [160,162,160]...)
        this function will treat the pixel with variance<threshold as
        background pixel

    i : int, default = None
        i is index of your target color in the color list. It controls
        the mode of this function.
        Mode 1 : i == None
        This function will extract all capacity layers;
        assgin the pixels in each layer with the representative color of the layer.
        Mode 2 : i == int
        This function will extract a layer of specific color in color list color[i]
        assgin the pixels in the layer with the representative color of the layer

    color : list, default = [[97,78,168],[58,197,163],[226,247,139],[255,224,122],[255,100,50],[177,0,67]]
        RGB color list of the capacity legend in original raster.

    mask : pd.DataFrame
        This is the original raster we are going to extract capacity layers from

    save_mode: Boole, default = False
        If this parameter set True, it will change output image to a image,
        that just have two type pixel {[0,0,0], [255,255,255]}

    Return
    ------------
    three channels (RGB) image array (np.array)
    '''

    # create clustering object, the color are the center
    cluster = KMeans(6)
    cluster.fit(np.array(color, dtype='float64'))
    cluster.cluster_centers_ = np.array(color, dtype='float64')

    # copy the data
    data = mask.copy()

    # before matching all pixel have same label 255
    data['predict'] = 255

    # match each pixel to a specific color, ignore background pixels
    data.loc[mask[mask.var(axis=1) > threshold].index, 'predict'] = cluster.predict(
        mask[mask.var(axis=1) > threshold])

    # uniform color in each layer
    if i is None:
        # make other pixel to white
        data.loc[data[data.predict == 255].index, :] = [255, 255, 255, 255]
    else:
        data.loc[data[data.predict != i].index, :] = [255, 255, 255, 255]
        # pixel without capacity data (Background) is white
        if not save_mode:
            data.loc[data[data.predict == i].index, :] = color[i] + [i]
        else:
            data.loc[data[data.predict == i].index, :] = [0, 0, 0, 0]
            # pixel with capacity data is black
    return data.iloc[:, :-1].values.astype('int').reshape(1800, 2880, 3)


def warp_with_gcps(input_path, output_path, gcps, gcp_epsg=4326, output_epsg=4326):
    '''
    This function is a tool for generating world files for rasters.
    It allows you to reference rasters to geographic or projected
    coordinate systems by creating a new GeoTiff with help of ground control points

    This function consits of two parts
    ----------------------------------
        part 1 is adapted from https://svn.osgeo.org/gdal/trunk/autotest/alg/warp.py

        But part 1 is not perfect. After it save the raster with coordinate system to a
        new GeoTiff. It will some how reduce the resolution of the original raster.
        Therefore There are no longer just two pixel value in new raster. {0,255}.
        This will cause a big problem in next step vectorization.

        So part 2 try to solve the problem caused by part 1.
        part 2 will let the pixel values of raster return to the state of just two value {0,255}
        The pixel with pixel value>20 will set it pixel value to 255. Other pixels' value will
        be 0.

    Parameters
    -----------
    input_path : str
        it should be relative path
        input image is .png file

    output_path : str
        it should be relative path
        output raster is .tif file (driver: GTiff)

    gcps: list
        it is a list of gdal.GCP objects (ground control points).
        It is manually created with QGIS and store in a json file :
        'original/GCP_Points.json'

    gcp_epsg: int, default = 4326
        it is the coordinate reference system of ground control points.
        4326 means crs = EPSG:4326 (latitude, longtitude)

    output_epsg: int, default = 4326
        it is the coordinate reference system of output raster.
        4326 means crs = EPSG:4326 (latitude, longtitude)

    Return
    ------------
    no return
    the result of these function will store in output_path

    '''
    # -----------------------part 1----------------------------------
    # Open the source dataset and add ground control points to it
    src_ds = gdal.OpenShared(str(input_path), gdal.GA_ReadOnly)
    gcp_srs = osr.SpatialReference()
    gcp_srs.ImportFromEPSG(gcp_epsg)
    gcp_crs_wkt = gcp_srs.ExportToWkt()
    src_ds.SetGCPs(gcps, gcp_crs_wkt)

    # Define target spatial reference system
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(output_epsg)
    dst_wkt = dst_srs.ExportToWkt()

    error_threshold = 0  # error threshold --> use same value as in gdalwarp
    resampling = gdal.GRA_Bilinear

    # Call AutoCreateWarpedVRT() to fetch default values for target raster dimensions and geotransform
    tmp_ds = gdal.AutoCreateWarpedVRT(src_ds,
                                      None,  # src_wkt : left to default value --> will use the one from source
                                      dst_wkt,
                                      resampling,
                                      error_threshold)
    dst_xsize = tmp_ds.RasterXSize
    dst_ysize = tmp_ds.RasterYSize
    dst_gt = tmp_ds.GetGeoTransform()
    tmp_ds = None

    # Now create the true target dataset
    dst_path = str(Path(output_path).with_suffix(".tif"))
    dst_ds = gdal.GetDriverByName('GTiff').Create(
        dst_path, dst_xsize, dst_ysize, src_ds.RasterCount)
    dst_ds.SetProjection(dst_wkt)
    dst_ds.SetGeoTransform(dst_gt)
    dst_ds.GetRasterBand(1).SetNoDataValue(255)

    # And run the reprojection
    gdal.ReprojectImage(src_ds,
                        dst_ds,
                        None,  # src_wkt : left to default value --> will use the one from source
                        None,  # dst_wkt : left to default value --> will use the one from destination
                        resampling,
                        0,  # WarpMemoryLimit : left to default value
                        error_threshold,
                        None,  # Progress callback : could be left to None or unspecified for silent progress
                        None)  # Progress callback user data
    dst_ds = None
    # -----------------------------------------------------------------

    # ----------------------part 2-------------------------------------
    with rasterio.open(output_path) as dataset:
        # get dirver of this raster type
        kwds = dataset.profile
        # reduce the band values into 0 and 255
        band = (dataset.read(1) > 20) * 255

    with rasterio.open(output_path, 'w', **kwds) as dataset:
        dataset.write_band(1, band.astype('uint8'))
    # -----------------------------------------------------------------


def raster2shp(input_path, output_path, capacity, background=255):
    '''
    This is the function convert raster to shapefile.
    This function will convert the original raster to two multipolygons.
    because our raster only have two type pixel values,
    so the result of this function are just two multipolygons:
    one is background (value: 0);
    another is salt cave location(value: capacity of the salt cave)

    Parameters
    ----------

    input_path : str
        It should be relative path.
        It is the path to input raster (GTiff file)


    output_path : str
        It should be relative path
        It is the path to store output shapefile (.shp file)

    capacity: int
        denstiy (unit : kwh/m-3 ) of the color in original raster legend.
        e.g.,200, 300,...

    background: int, default = 255
        This is the value of background pixel in the function warp_with_gcps()


    Return
    -----------
    no return
    output store in output_path
    '''
    with rasterio.open(input_path) as src:
        # metadata for shapefile coordinate reference system
        crs = src.crs
        # read raster as a numpy array
        src_band = src.read(1)
        # Keep track of unique pixel values in the input band
        unique_values = np.unique(src_band)
        # Polygonize with Rasterio. `shapes()` returns an iterable
        # of (geom, value) as tuples
        shapes = list(rasterio.features.shapes(
            src_band, transform=src.transform))

    # metadata for shapefile
    shp_schema = {
        'geometry': 'MultiPolygon',
        'properties': {'val_kwhm3': 'int'}
    }

    # Get a list of all polygons for a given pixel value
    # and create a MultiPolygon geometry with shapely.
    # Then write the record to an output shapefile with fiona.
    # We make use of the `shape()` and `mapping()` functions from
    # shapely to translate between the GeoJSON-like dict format
    # and the shapely geometry type.
    with fiona.open(output_path, 'w', 'ESRI Shapefile', shp_schema, crs) as shp:
        for pixel_value in unique_values:
            # combine the Geometries with same value in last step to multipolygon
            polygons = [shape(geom) for geom, value in shapes
                        if value == pixel_value]
            multipolygon = MultiPolygon(polygons)

            # redefine the value of each multipolygon
            # background multipolygon's value = 0
            # salt cave multipolygon's value = capacity denstiy of the salt cave
            if pixel_value == background:
                pixel_value = 0
            else:
                pixel_value = capacity

            # output shapefile
            shp.write({
                'geometry': mapping(multipolygon),
                'properties': {'val_kwhm3': int(pixel_value)}
            })


def save_img(path, img):
    '''
    Parameters
    ----------
    path : str
        should be relative path
    img : np.array
        should be three channels RGB image array
    '''
    img_save = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(path, img_save)
    return 1


def convert_original_png2shp(img_path, gcps_path, png_path, tif_path, shp_path,final_output_path, background=255):
    '''
    Parameters
    ----------
    img_path : str
        path to original hydrogen storage density map

    gcps_path : str
        path to json file that store gcp points data

    png_path : str
        path that use to store intermediate result

    tif_path : str
        path that use to store intermediate result

    shp_path :str
        path that use to store intermediate result

    final_output_path : str
        path that use to store hydrogen storage density map (capacity/area)
        final result of this funtion is store in this path

    background: int, default = 255
        This is the value of background pixel in the function warp_with_gcps()

    '''

    # read original path
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # flat picture to list and create pandas dataframe
    img_list = img.reshape((-1, 3))
    img_df = pd.DataFrame(img_list)

    # read gcps and modify it
    with open(gcps_path) as jsonfile:
        gcp = json.load(jsonfile)
    object_gcp = []
    for unit in gcp:
        object_gcp.append(gdal.GCP(unit[2], unit[3], 0, unit[0], unit[1]))

    # prepare useful variables
    capacity_list = [200, 250, 300, 350, 400, 450]
    color = [[97, 78, 168], [58, 197, 163], [226, 247, 139], [255, 224, 122], [255, 100, 50], [177, 0, 67]]

    # -------------------------#
    # -------start convert-----#
    # -------------------------#
    for i in range(len(capacity_list)):
        capacity = capacity_list[i]

        # Step1: split original raster into six raster by different colors
        current_layer = extract_with_threshold(color, img_df, 50, i=i, save_mode=True).astype('uint8')
        save_img(png_path.format(capacity),current_layer)

        # Step2: add gcp to each raster and create new raster with georeference
        warp_with_gcps(png_path.format(capacity),
                       tif_path.format(capacity),
                       object_gcp,
                       gcp_epsg=4326,
                       output_epsg=4326)

        # Step3: polygonize rasters in last step
        raster2shp(tif_path.format(capacity),
                   shp_path.format(capacity),
                   capacity, background)

    # Step4: create an aggreated shapefile which have six multipolygons with different capacity
    gdflist = []
    for capacity in capacity_list:
        gdflist.append(gpd.read_file(shp_path.format(capacity)))
    geodataframe = pd.concat(gdflist)
    geodataframe = geodataframe[geodataframe['val_kwhm3'] != 0]
    geodataframe = gpd.GeoDataFrame(
        geodataframe, crs="EPSG:4326").reset_index(drop=True)

    # fix invalid polygons
    geodataframe['geometry'] = geodataframe.geometry.buffer(0)

    # save final.shp in folder vector
    geodataframe.to_file(final_output_path)

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
# functions for creating capacity per area density map from capacity
# per volume map
#--------------------------------------------------------------------


def create_nearshore_50_pair(on_path, off_path):
    '''
    Parameters
    ----------
    on_path : str
        path to file that store onshore map
    off_path : str
         path to file that store offshore map

    Returns
    -----------
    nearshore: gpd.dataframe
        onshore regions in europe that within 50km from the seashore

    realonshore: gpd.dataframe
        onshore regions in europe that at least 50km away from the seashore
    '''
    onshore = gpd.read_file(on_path)
    offshore = gpd.read_file(off_path)
    offshore_buffer = offshore.to_crs(crs='EPSG:3395')
    offshore_buffer.geometry = offshore_buffer.buffer(50000)  # 50km buffer
    offshore_buffer = offshore_buffer.to_crs(crs='EPSG:4326')
    nearshore = gpd.clip(onshore, offshore_buffer)
    realonshore = gpd.overlay(onshore, offshore_buffer, how='difference')
    return nearshore, realonshore


def calculate_depth(mask, depth_type, capacity_path, density_map, capacity_name='val_kwhm3'):
    '''
    Parameters
    ----------
    mask : gpd.dataframe
        mask use to select area
    depth_type : str
        possible choices: 'onshore', 'nearshore', 'offshore'
    capacity_path: str
        path to the file that store hydrogen storage capacity of each country
    density_map: str
        path to the file that store hydrogen storage capacity density all over the europe (capacity per volume)
    capacity_name: str
        the name of the column that store hydrogen storage capacity density all over the europe

    Returns
    -----------
    depth table: pd.DataFrame
        metadata of the table = [country name, value of specific type depth]
    '''
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
    '''
    Parameters
    ----------
    density_map : geodataframe
        EU hydrogen storage capacity map (capacity per volume)
    mask_list : list
        list of gpd.geodataframe, in which each unit will use to select a specific area
    depth_list: list
        list of depth, in which each unit is the average value of a specific depth in a specific country
        e.g., onshore depth, offshore depth, nearshore depth
    capacity_name: str
        the name of the column that store hydrogen storage capacity density all over the europe

    Returns
    -----------
    depth table: pd.DataFrame
       capacity per area of a country.
    '''
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


def calculate_capacity_per_area_shape(onshore, nearshore, realoffshore, density_map_path, capacity_path, capacity_name='val_kwhm3'):
    '''
    Parameters
    ----------
    onshore : gpd.geodataframe
        onshore regions at least 50km away from the seeshore.

    nearshore : gpd.geodataframe
        onshore regions within 50km from the seeshore.

    realoffshore : gpd.geodataframe
        offshore regions

    density_map_path: str
        path to the file that store hydrogen storage capacity density all over the europe (capacity per volume)

    capacity_path: str
        path to the file that store hydrogen storage capacity of each country

    Returns
    -----------
    return value of function get_capacity_per_area
    '''
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


if __name__ == "__main__":
    if 'snakemake' not in globals():
        # test mode
        logging.basicConfig(filename="../logs/build_hydrogen_map.log", level=logging.INFO)
        for count in range(5):
            logging.info("------------------------------------------------------------ -")
        img_path = '../data/bundle/salt_cave/salt_cave.png'
        gcps_path = '../data/bundle/salt_cave/GCP_Points.json'
        onshore_path = "../resources/country_shapes.geojson"
        offshore_path = "../resources/offshore_shapes.geojson"
        capacity_path = "../data/bundle/salt_cave/storage_potential_eu_kwh.csv"
        new_capacity_per_area_path = "../resources/new_energy_capacity_map_kwhm2.geojson"
    else:
        configure_logging(snakemake)
        for count in range(5):
            logging.info("-"*30)
        img_path = snakemake.input.img_path
        gcps_path = snakemake.input.gcps_path
        onshore_path = snakemake.input.onshore_path
        offshore_path = snakemake.input.offshore_path
        capacity_path = snakemake.input.capacity_path
        new_capacity_per_area_path = snakemake.output.new_capacity_per_area_path
        # snakemake mode (normal mode)

    # define vital variables
    tmp_folder = '../resources/hydrogen_tmp/'
    # create tmp folder in resources
    try:
        os.mkdir(tmp_folder)
    except FileExistsError:
        shutil.rmtree(tmp_folder)
        os.mkdir(tmp_folder)
    except:
        logging.warning('tmp folder is not successfully created!')

    png_path = tmp_folder + 'capacity_{}.png'
    tif_path = tmp_folder + 'capacity_{}.tif'
    shp_path = tmp_folder + 'capacity_{}.shp'
    final_output_path = tmp_folder + 'final.shp'

    convert_original_png2shp(img_path, gcps_path, png_path, tif_path, shp_path, final_output_path)
    # load capacity density map (capacity per area)
    #   this block make sure capacity_shape will be loaded
    # -------------------------------------------------
    try:
        capacity_shape = gpd.read_file(new_capacity_per_area_path)
    except(DriverError):
        logging.info('no new_capacity_per_area file, start create it')
        # steps in except take around 2 minute for a normal PC
        offshore = gpd.read_file(offshore_path)
        nearshore, realonshore = create_nearshore_50_pair(
            onshore_path, offshore_path)
        capacity_shape = calculate_capacity_per_area_shape(
            realonshore, nearshore, offshore, final_output_path, capacity_path)
        capacity_shape.to_file(
            new_capacity_per_area_path, driver='GeoJSON')
        logging.info('successful create new_capacity_per_area file!')
    # --------------------------------------------------
    # remove tmp folder and files in it
    shutil.rmtree(tmp_folder)
