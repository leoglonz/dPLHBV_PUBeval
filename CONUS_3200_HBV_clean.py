import sys
sys.path.append('../../')

from hydroDL.model import crit, train
from hydroDL.model import rnn as rnn
import os
import numpy as np
import torch
import random
import pandas as pd
import json

from sklearn.model_selection import KFold
## fix the random seeds for reproducibility
randomseed = 111111
random.seed(randomseed)
torch.manual_seed(randomseed)
np.random.seed(randomseed)
torch.cuda.manual_seed(randomseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _basin_norm(
    x: np.array, basin_area: np.array, to_norm: bool
) -> np.array:
    """

    Because units of streamflow and area are ft^3/s and km^2 respectively,
    and we need (mm/day), we transform the equation as the code shows.

    """
    nd = len(x.shape)

    if nd == 3 and x.shape[2] == 1:
        x = x[:, :, 0]  # unsqueeze the original 3 dimension matrix
    temparea = np.tile(basin_area, (1, x.shape[1]))

    if to_norm is True:

        flow = (x * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10 ** 3

    else:

        flow = (
            x
            * ((temparea * (10**6)) * (10 ** (-3)))
            / (0.0283168 * 3600 * 24)
        )

    if nd == 3:
        flow = np.expand_dims(flow, axis=2)
    return flow


def trans_norm(x, var_lst, stat_dict, *, to_norm):
    """
    normalization, including denormalization code

    Parameters
    ----------
    x
        2d or 3d data
        2d：1st-sites，2nd-var type
        3d：1st-sites，2nd-time, 3rd-var type
    var_lst
        variables
    stat_dict
        a dict with statistics info
    to_norm
        if True, normalization; else denormalization

    Returns
    -------
    np.array
        normalized/denormalized data
    """
    if type(var_lst) is str:
        var_lst = [var_lst]
    out = np.zeros(x.shape)
    for k in range(len(var_lst)):
        var = var_lst[k]
        stat = stat_dict[var]
        if to_norm is True:
            if len(x.shape) == 3:
                out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]
            elif len(x.shape) == 2:
                out[:, k] = (x[:, k] - stat[2]) / stat[3]
        else:
            if len(x.shape) == 3:
                out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
            elif len(x.shape) == 2:
                out[:, k] = x[:, k] * stat[3] + stat[2]
    return out

def _trans_norm(
    x: np.array, var_lst: list, stat_dict: dict, log_norm_cols: list, to_norm: bool
) -> np.array:
    """
    Normalization or inverse normalization

    There are two normalization formulas:

    .. math:: normalized_x = (x - mean) / std

    and

     .. math:: normalized_x = [log_{10}(\sqrt{x} + 0.1) - mean] / std

     The later is only for vars in log_norm_cols; mean is mean value; std means standard deviation

    Parameters
    ----------
    x
        data to be normalized or denormalized
    var_lst
        the type of variables
    stat_dict
        statistics of all variables
    log_norm_cols
        which cols use the second norm method
    to_norm
        if true, normalize; else denormalize

    Returns
    -------
    np.array
        normalized or denormalized data
    """
    if type(var_lst) is str:
        var_lst = [var_lst]
    out = np.full(x.shape, np.nan)
    for k in range(len(var_lst)):
        var = var_lst[k]
        stat = stat_dict[var]
        if to_norm is True:
            if len(x.shape) == 3:
                if var in log_norm_cols:
                    x[:, :, k] = np.log10(np.sqrt(x[:, :, k]) + 0.1)
                out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]
            elif len(x.shape) == 2:
                if var in log_norm_cols:
                    x[:, k] = np.log10(np.sqrt(x[:, k]) + 0.1)
                out[:, k] = (x[:, k] - stat[2]) / stat[3]
        else:
            if len(x.shape) == 3:
                out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
                if var in log_norm_cols:
                    out[:, :, k] = (np.power(10, out[:, :, k]) - 0.1) ** 2
            elif len(x.shape) == 2:
                out[:, k] = x[:, k] * stat[3] + stat[2]
                if var in log_norm_cols:
                    out[:, k] = (np.power(10, out[:, k]) - 0.1) ** 2
    return out


def hargreaves(tmin, tmax, tmean, lat, trange):
    # calculate the day of year
    dfdate = pd.date_range(start=str(trange[0]), end=str(trange[1]), freq='D', closed='left') # end not included
    tempday = np.array(dfdate.dayofyear)
    day_of_year = np.tile(tempday.reshape(-1, 1), [1, tmin.shape[-1]])
    # Loop to reduce memory usage
    pet = np.zeros(tmin.shape, dtype=np.float32) * np.NaN
    for ii in np.arange(len(pet[:, 0])):
        trange = tmax[ii, :] - tmin[ii, :]
        trange[trange < 0] = 0

        latitude = np.deg2rad(lat[ii, :])

        SOLAR_CONSTANT = 0.0820

        sol_dec = 0.409 * np.sin(((2.0 * np.pi / 365.0) * day_of_year[ii, :] - 1.39))

        sha = np.arccos(np.clip(-np.tan(latitude) * np.tan(sol_dec), -1, 1))

        ird = 1 + (0.033 * np.cos((2.0 * np.pi / 365.0) * day_of_year[ii, :]))

        tmp1 = (24.0 * 60.0) / np.pi
        tmp2 = sha * np.sin(latitude) * np.sin(sol_dec)
        tmp3 = np.cos(latitude) * np.cos(sol_dec) * np.sin(sha)
        et_rad = tmp1 * SOLAR_CONSTANT * ird * (tmp2 + tmp3)

        pet[ii, :] = 0.0023 * (tmean[ii, :] + 17.8) * trange ** 0.5 * 0.408 * et_rad

    pet[pet < 0] = 0

    return pet

def cal_4_stat_inds(b):
    """
    Calculate four statistics indices: percentile 10 and 90, mean value, standard deviation

    Parameters
    ----------
    b
        input data

    Returns
    -------
    list
        [p10, p90, mean, std]
    """
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]


def cal_stat_gamma(x):
    """
    Try to transform a time series data to normal distribution

    Now only for daily streamflow, precipitation and evapotranspiration;
    When nan values exist, just ignore them.

    Parameters
    ----------
    x
        time series data

    Returns
    -------
    list
        [p10, p90, mean, std]
    """
    a = x.flatten()
    b = a[~np.isnan(a)]  # kick out Nan
    b = np.log10(
        np.sqrt(b) + 0.1
    )  # do some tranformation to change gamma characteristics
    return cal_4_stat_inds(b)

def cal_stat(x: np.array) -> list:
    """
    Get statistic values of x (Exclude the NaN values)

    Parameters
    ----------
    x: the array

    Returns
    -------
    list
        [10% quantile, 90% quantile, mean, std]
    """
    a = x.flatten()
    b = a[~np.isnan(a)]
    if b.size == 0:
        # if b is [], then give it a 0 value
        b = np.array([0])
    return cal_4_stat_inds(b)


def fill_Nan(array_3d):
    # Define the x-axis for interpolation
    x = np.arange(array_3d.shape[1])

    # Iterate over the first and third dimensions to interpolate the second dimension
    for i in range(array_3d.shape[0]):
        for j in range(array_3d.shape[2]):
            # Select the 1D slice for interpolation
            slice_1d = array_3d[i, :, j]

            # Find indices of NaNs and non-NaNs
            nans = np.isnan(slice_1d)
            non_nans = ~nans

            # Only interpolate if there are NaNs and at least two non-NaN values for reference
            if np.any(nans) and np.sum(non_nans) > 1:
                # Perform linear interpolation using numpy.interp
                array_3d[i, :, j] = np.interp(x, x[non_nans], slice_1d[non_nans], left=None, right=None)
    return array_3d

## GPU setting
# which GPU to use when having multiple
traingpuid = 4
torch.cuda.set_device(traingpuid)

## streamflow data I extracted from USGS
data_folder = "/data/yxs275/CONUS_data/FromGAGEII/generate_for_CONUS_3200/gages/dataCONUS3200/"


with open(data_folder+'train_data_dict.json') as f:
    train_data_dict = json.load(f)

## Time span of Daymet
AllTime = pd.date_range('1980-01-01', f'2020-12-31', freq='d')
index_start = AllTime.get_loc(train_data_dict["t_final_range"][0])
index_end = AllTime.get_loc(train_data_dict["t_final_range"][-1])

## Slice Daymet data for the time span where we have streamflow data
forcingAll = np.load("/data/yxs275/CONUS_data/FromGAGEII/generate_for_CONUS_3200/allBasin_localDaymet.npy")[:,index_start:index_end,:]
## IDs of gages where we have streamflow data
shapeID_str_lst= train_data_dict['sites_id']

## Fill the nans in Daymet with interpolation (This is for missing data in leap years)
forcingAll = fill_Nan(forcingAll)

forcingAllLst  = ['prcp', 'tmax', 'tmin']
##Streamflow data
streamflow = np.load(data_folder+"train_flow.npy")

## Attributes collected by Haoyu
attribute_file = '/data/yxs275/CONUS_data/attributes/CONUS_3254/attributes_haoyu.csv'
attributeALL_df = pd.read_csv(attribute_file,index_col=0)
attributeALL_df = attributeALL_df.sort_values(by='id')

attributeAllLst = attributeALL_df.columns

basin_area = np.expand_dims(attributeALL_df["area"].values,axis = 1)
lat =  attributeALL_df["lat"].values

##To find the sites where we have both streamflow and attributes
idLst_new = attributeALL_df["id"].values
idLst_old = [int(id) for id in shapeID_str_lst]

[C, ind1, SubInd] = np.intersect1d(idLst_new, idLst_old, return_indices=True)

forcingAll = forcingAll[SubInd,:,:]
streamflow = streamflow[SubInd,:,:]


log_norm_cols=['prcp', 'pr', 'total_precipitation', 'pre', 'potential_evaporation', 'LE',
                   'PLE', 'GPP', 'Ec', 'Es', 'Ei', 'ET_water', 'ET_sum', 'susm', 'smp', 'ssma', 'susma',
                   'usgsFlow', 'streamflow', 'qobs']

attributeLst = ['area','ETPOT_Hargr', 'FW', 'HWSD_clay', 'HWSD_gravel', 'HWSD_sand',
       'HWSD_silt', 'NDVI', 'Porosity', 'SoilGrids1km_clay',
       'SoilGrids1km_sand', 'SoilGrids1km_silt', 'T_clay', 'T_gravel',
       'T_sand', 'T_silt', 'aridity', 'glaciers', 'meanP', 'meanTa',
       'meanelevation', 'meanslope', 'permafrost', 'permeability',
       'seasonality_P', 'seasonality_PET', 'snow_fraction',
       'snowfall_fraction']

forcing_HBV_List = ['prcp',  'tmax', 'tmin', ]

##Caculate the statistics for forcing
stat_dict={}
for fid, forcing_item in enumerate(forcingAllLst) :
    if forcing_item in log_norm_cols:
        stat_dict[forcing_item] = cal_stat_gamma(forcingAll[:,:,fid])
    else:
        stat_dict[forcing_item] = cal_stat(forcingAll[:,:,fid])

## Slice the attributes we want from all attributes
[C, _, SubInd] = np.intersect1d(attributeLst, attributeAllLst, return_indices=True)
sorted_SubInd = [x for x, _ in sorted(zip(SubInd, ind1), key=lambda pair: pair[1])]
attribute = attributeALL_df.iloc[ind1, sorted_SubInd].values

attributeLst  = list(attributeAllLst[sorted_SubInd])

##Caculate the statistics for attributes
for aid, attribute_item in enumerate (attributeLst):
    stat_dict[attribute_item] = cal_stat(attribute[:,aid])

##Use tmean to caculate PET
Tex = [19801001, 19951001]

tmin = np.swapaxes(forcingAll[:,:,np.where(np.array(forcingAllLst) == "tmin")[0][0]], 0,1)
tmax = np.swapaxes(forcingAll[:,:,np.where(np.array(forcingAllLst) == "tmax")[0][0]], 0,1)

tmean = (tmin+tmax)/2

latarray = np.tile(lat, [tmin.shape[0], 1])
pet = hargreaves(tmin, tmax, tmean, lat=latarray, trange=Tex)

##Reorganize the forcing ['prcp','tmean','pet']
xTrain = np.zeros([forcingAll.shape[0],forcingAll.shape[1],3])
xTrain[:,:,0] = forcingAll[:,:,np.where(np.array(forcingAllLst) == "prcp")[0][0]]
xTrain[:,:,1] = np.swapaxes(tmean, 0,1)
xTrain[:,:,2] = np.swapaxes(pet, 0,1)


streamflow_trans = _basin_norm(
                        streamflow[:, :, 0 :  1].copy(), basin_area, to_norm=True
                    )  ## from ft^3/s to mm/day


## calculate the statistics for new forcing terms
stat_dict['tmean'] = cal_stat(xTrain[:,:,1])
stat_dict['pet'] = cal_stat(xTrain[:,:,2])

##Build the input for LSTM for parameterization
forcing_LSTM_norm = _trans_norm(
    xTrain.copy(), ['prcp','tmean','pet'], stat_dict, log_norm_cols=log_norm_cols, to_norm=True
)
xTrain[xTrain!=xTrain]  = 0
forcing_LSTM_norm[forcing_LSTM_norm!=forcing_LSTM_norm]  = 0

attribute_norm = trans_norm(attribute, list(attributeLst), stat_dict, to_norm=True)
attribute_norm[attribute_norm!=attribute_norm] = 0


Ninv = forcing_LSTM_norm.shape[-1]+attribute_norm.shape[-1]
EPOCH = 50 # total epoches to train the mode
BATCH_SIZE = 100
RHO = 365
saveEPOCH = 10
alpha = 0.25
HIDDENSIZE = 256
BUFFTIME = 365 # for each training sample, to use BUFFTIME days to warm up the states.
routing = True # Whether to use the routing module for simulated runoff
Nmul = 16 # Multi-component model. How many parallel HBV components to use. 1 means the original HBV.
comprout = False # True is doing routing for each component
compwts = False # True is using weighted average for components; False is the simple mean
pcorr = None # or a list to give the range of precip correc

tdRep = [1, 13]  # When using dynamic parameters, this list defines which parameters to set as dynamic
tdRepS = [str(ix) for ix in tdRep]
# ETMod: if True, use the added shape parameter (index 13) for ET. Default as False.
# Must set below ETMod as True and Nfea=13 when including 13 index in above tdRep list for dynamic parameters
# If 13 not in tdRep list, set below ETMod=False and Nfea=12 to use the original HBV without ET shape para
ETMod = True
Nfea = 13  # should be 13 when setting ETMod=True. 12 when ETMod=False
dydrop = 0.0  # dropout possibility for those dynamic parameters: 0.0 always dynamic; 1.0 always static
staind = -1  # which time step to use from the learned para time series for those static parameters

model = rnn.MultiInv_HBVTDModel(ninv=Ninv, nfea=Nfea, nmul=Nmul, hiddeninv=HIDDENSIZE, inittime=BUFFTIME,
                                routOpt=routing, comprout=comprout, compwts=compwts, staind=staind, tdlst=tdRep,
                                dydrop=dydrop, ETMod=ETMod)
lossFun = crit.RmseLossComb(alpha=alpha)



rootOut = '/data/lgl5139/DPL_HBV/CONUS_3200_Output/'+'/dPL_local_daymet_new_attr/'
if os.path.exists(rootOut) is False:
    os.mkdir(rootOut)
out = os.path.join(rootOut, f"exp_EPOCH{EPOCH}_BS{BATCH_SIZE}_RHO{RHO}_HS{HIDDENSIZE}_trainBuff{BUFFTIME}") # output folder to save results
if os.path.exists(out) is False:
    os.mkdir(out)

with open(out+'/dapengscaler_stat.json','w') as f:
    json.dump(stat_dict, f)


##K fold for spatial tests
# fold_number = 1
# # Define the number of folds (K)
# k = 6
#
# # Initialize the KFold cross-validator
# kf = KFold(n_splits=k, shuffle=True, random_state=randomseed)
#
# # Iterate through the folds
# for fold_number_idx, (train_index, test_index) in enumerate(kf.split(streamflow_trans)):
#     if fold_number_idx == fold_number:
#         xTrain_fold = xTrain[train_index,:,:]
#         forcing_LSTM_norm_fold = forcing_LSTM_norm[train_index,:,:]
#         streamflow_trans_fold = streamflow_trans[train_index,:,:]
#         attribute_norm_fold = attribute_norm[train_index,:]
# forcTuple_fold = [xTrain_fold,forcing_LSTM_norm_fold]
#
# trainedModel = train.trainModel(
#     model,
#     forcTuple_fold,
#     streamflow_trans_fold,
#     attribute_norm_fold,
#     lossFun,
#     nEpoch=EPOCH,
#     miniBatch=[BATCH_SIZE, RHO],
#     saveEpoch=saveEPOCH,
#     saveFolder=out,
#     bufftime=BUFFTIME)

# For temporal test
forcTuple = [xTrain,forcing_LSTM_norm]
trainedModel = train.trainModel(
    model,
    forcTuple,
    streamflow_trans,
    attribute_norm,
    lossFun,
    nEpoch=EPOCH,
    miniBatch=[BATCH_SIZE, RHO],
    saveEpoch=saveEPOCH,
    saveFolder=out,
    bufftime=BUFFTIME)