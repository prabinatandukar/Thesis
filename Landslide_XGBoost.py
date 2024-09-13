import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

from sklearn.ensemble import RandomForestClassifier
from glob import glob, iglob
from osgeo import gdal, gdal_array
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from write_geotif import CreateGeoTiff
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ConfusionMatrix

from yellowbrick.model_selection import FeatureImportances

from tqdm import tqdm
from joblib import Parallel, delayed
import earthpy.plot as ep
import gc
import os
import pickle


# Reading Data:This line reads a CSV file named "point_final.csv" into a pandas DataFrame (data_org) using pd.read_csv(). The delimiter used in the CSV file is ;.
data_org = pd.read_csv("point_final.csv", delimiter=";")


# Data Preprocessing:Here, a copy of the original DataFrame is created (data) to perform data preprocessing. In this step, commas in the data are replaced with periods. This is likely done to ensure consistency in representing decimal numbers.
data = data_org.copy()
data = data.replace(",", ".", regex=True)

# Removing First Column:This line removes the first column from the DataFrame data. It's common in data processing tasks to remove index columns or columns that are not needed for analysis.
data = data.iloc[:, 1:]

# Separate features and target variable: Here, the DataFrame X is created containing all the features (independent variables) except for the 'Landslide' column, while y contains only the 'Landslide' column, which is the target variable (dependent variable).
X = data.drop('Landslide', axis=1).copy()
y = data.loc[:, 'Landslide'].copy()

# Converting Data Types:This line converts all the columns in DataFrame X to float data type.
X = X.astype(float)


# Converting Categorical Variables:Here, specific columns ('aspect' and 'LULC') are converted to string data type. This might be done in preparation for one-hot encoding categorical variables.
#!!!!!uncomment when you have your aspect as categorical/factor variable fro example 1,2,3,4.....
X['aspect'] = X['aspect'].astype(str)
X['LULC'] = X['LULC'].astype(str)


# Identifying Categorical Columns:This line identifies categorical columns in the DataFrame X and stores their column names in the list cols_obj.
cols_obj = X.columns[X.dtypes == 'object'].values.tolist()
cols_obj
# One-Hot Encoding Categorical Variables: This line performs one-hot encoding on the categorical variables identified earlier and stores the result in the DataFrame X_encoded.
X_encoded = pd.get_dummies(X, columns=cols_obj)

# Train-Test Split:This line splits the data into training and testing sets using train_test_split() function from scikit-learn. The training set comprises 70% of the data, and stratification is applied based on the target variable y.
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, random_state=6768, train_size=0.70, stratify=y)

with open("./xgb_boost_finalized_model.sav", 'rb') as file:

    optimal_parameters = pickle.load(file)


# Defining XGBoost Classifier:
clf_xgb = xgb.XGBClassifier(**optimal_parameters)
# Fitting the Classifier: This line trains the XGBoost classifier (clf_xgb) on the training data (X_train, y_train) and evaluates it on the test set (X_test, y_test). The verbose=True argument prints the evaluation results during training.
clf_xgb.fit(X_train, y_train, verbose=True, eval_set=[(X_test, y_test)])

class_mapper = {0: "No", 1: "Yes"}


# Fitting the Classifier and Training Set:This line fits the XGBoost classifier to the training data (X_train, y_train) and evaluates it on the test set (X_test, y_test). The verbose=True argument prints the evaluation results during training.
clf_xgb.fit(X_train, y_train, verbose=True, eval_set=[(X_test, y_test)])

# Loading Image Files:
full_data_files = glob("F:/thesis/TEMP/project/data/*.tif")


files_sorted = ['F:/thesis/TEMP/project/data/dist_riv_mask.tif',
                'F:/thesis/TEMP/project/data/DTM_merged_crop1.tif',
                'F:/thesis/TEMP/project/data/ndvi_rescale.tif',
                'F:/thesis/TEMP/project/data/plan_curvature.tif',
                'F:/thesis/TEMP/project/data/profile_curvature.tif',
                'F:/thesis/TEMP/project/data/rain_resample.tif',
                'F:/thesis/TEMP/project/data/reclass_aspect111.tif',
                'F:/thesis/TEMP/project/data/resample_lulc.tif',
                'F:/thesis/TEMP/project/data/road_resample.tif',
                'F:/thesis/TEMP/project/data/slope_resample.tif']


def offset_chunk_dimension(arrayshape=None, chunk_size=None):
    xy_chunk = chunk_size
    if len(arrayshape) == 2:
        array_shape = (0, arrayshape[0], arrayshape[1])
    x_blocks = np.ceil(array_shape[-2]/xy_chunk[0]).astype(int)
    y_blocks = np.ceil(array_shape[-1]/xy_chunk[1]).astype(int)
    x_offsets = np.arange(x_blocks, dtype=int) * xy_chunk[0]
    y_offsets = np.arange(y_blocks, dtype=int) * xy_chunk[1]
    grid_offset = np.meshgrid(x_offsets, y_offsets)
    xy_offsets = np.c_[grid_offset[0].ravel(), grid_offset[1].ravel()]
    x_chunk = (np.repeat(xy_chunk[0], x_blocks).astype(int))
    rmd_x = array_shape[1] % xy_chunk[0]
    if rmd_x > 0:
        x_chunk[-1] = rmd_x
    y_chunk = (np.repeat(xy_chunk[1], y_blocks).astype(int))
    rmd_y = array_shape[2] % xy_chunk[1]
    if rmd_y > 0:
        y_chunk[-1] = rmd_y
    grid_chunk = np.meshgrid(x_chunk, y_chunk)
    xy_grid_chunk = np.c_[grid_chunk[0].ravel(), grid_chunk[1].ravel()]
    xy_grid_chunk = xy_grid_chunk.tolist()
    return np.hstack([xy_offsets, xy_grid_chunk])


def write_out_blk(gdal_dataset=None, idx=1, im_array=None, xoffset=None, yoffset=None, NDV=-9999, band_name="xgboost"):
    gdal_dataset.GetRasterBand(idx).WriteArray(
        im_array, xoff=xoffset, yoff=yoffset)
    gdal_dataset.GetRasterBand(idx).SetNoDataValue(NDV)
    gdal_dataset.SetDescription(band_name)


data_template = gdal.Open(files_sorted[2])
print(files_sorted[2])
print(data_template)
band = data_template.GetRasterBand(1)


data_type = gdal.GDT_Float32  # data type 6 (float32)

bands = 1

Name = "./xgboost/xgboost_pred_debug_full_2.tif"
DataType = 6
driver = gdal.GetDriverByName('GTIFF')
# Array[np.isnan(Array)] = NDV
DataSet = driver.Create(Name, band.XSize, band.YSize, bands, data_type)
DataSet.SetGeoTransform(data_template.GetGeoTransform())
DataSet.SetProjection(data_template.GetProjection())
fname = os.path.basename(Name)


blocks = offset_chunk_dimension(arrayshape=(
    band.XSize, band.YSize), chunk_size=(900, 900))

blocks = blocks.tolist()

geot = gdal.Open(files_sorted[0]).GetGeoTransform()


# Loading Image Data:Here, each TIFF file is opened using GDAL, and its data is read as an array within a specified subset.
block_id = 1
# hard_encoded:composite_block_array.columns,X_encoded.columns,clf_xgb, filename


def predict_model_blocks(sorted_image_list=files_sorted, block=None, geotransform=None, gdal_write_file=None):
    subset = block

    # Creating Longitude and Latitude Arrays: Longitude and latitude arrays are generated based on the geotransform information obtained from one of the TIFF files
    geotransform_x = geotransform[0]
    geotransform_y = geotransform[3]
    x_pixel = geotransform[1]
    y_pixel = geotransform[5]
    long = [geotransform_x + (i*x_pixel)
            for i in range(block[0], block[0]+block[2])]
    lat = [geotransform_y + (i*y_pixel)
           for i in range(block[1], block[1]+block[3])]

    xx, yy = np.meshgrid(np.array(long), np.array(lat))
    xx_yy_matrix = np.c_[xx.ravel(), yy.ravel()]

    composite_block_array = [gdal.Open(i).ReadAsArray(
        *subset) for i in sorted_image_list]

    composite_block_array = np.stack(composite_block_array)
    dimension = composite_block_array.shape

    # Reshaping Image Data:The 3D array is reshaped into a 2D array for further processing.
    composite_block_array = composite_block_array.reshape(
        dimension[0], (dimension[1]*dimension[2]))
    composite_block_array = composite_block_array.swapaxes(0, 1)

    # Concatenating Longitude, Latitude, and Image Data:Longitude, latitude, and image data arrays are concatenated horizontally to form a single array representing the entire dataset.
    composite_block_array = np.hstack((xx_yy_matrix, composite_block_array))
    composite_block_array = pd.DataFrame(
        composite_block_array, columns=X.columns)

    composite_block_array['aspect'] = composite_block_array['aspect'].astype(
        str)
    composite_block_array['LULC'] = composite_block_array['LULC'].astype(str)
    cols_obj_full = composite_block_array.columns[composite_block_array.dtypes == 'object'].values.tolist(
    )
    # encoded
    composite_block_array = pd.get_dummies(
        composite_block_array, columns=cols_obj_full)

    # Assuming X_encoded and full_data_array_df_encoded are your DataFrames
    missing_columns = [
        c for c in X_encoded.columns if c not in composite_block_array.columns]

    # Add missing columns to full_data_array_df_encoded and fill with false
    if len(missing_columns) > 0:
        composite_block_array[missing_columns] = False

    # Columns in the DataFrame are filtered to match those used during training, and their order is rearranged accordingly.
    composite_block_array = composite_block_array[X_encoded.columns]

    # Making predictions using the XGBoost model
    # predictions = clf_xgb.predict(composite_block_array)
    predictions = clf_xgb.predict_proba(composite_block_array)
    # probability of occurence
    predictions = predictions[:, 1]

    # Reshaping Predictions:The predicted probabilities are reshaped to match the dimensions of the original image.
    predictions = predictions.reshape(dimension[1:])

    write_out_blk(gdal_dataset=gdal_write_file, idx=1, im_array=predictions,
                  xoffset=subset[0], yoffset=subset[1], NDV=-9999, band_name="xgboost")
    return block


nworkers = 6

sorted_image_list = files_sorted


Parallel(n_jobs=nworkers, backend='threading')(delayed(predict_model_blocks)(sorted_image_list, block, geotransform=geot, gdal_write_file=DataSet)
                                               for block in tqdm(blocks, desc="Xgboost_model", colour='blue', initial=1))
