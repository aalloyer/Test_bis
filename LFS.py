import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os
import netCDF4 as nc
import statsmodels.api as sm
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import streamlit as st
from io import BytesIO
import matplotlib.pyplot as plt
import datetime
import matplotlib.pyplot as plt


@st.cache_data
def nearest_point_cmip(lon_, lat_, path_nc): 
    current_time = datetime.datetime.now()
    st.write(f"nearest_point_cmip en exécution à {current_time}")
    files = os.listdir(path_nc)
    nc_input = nc.Dataset(f"{path_nc}{files[0]}")
    lon = nc_input.variables["lon"][:]
    lat = nc_input.variables["lat"][:]
    time_var = nc_input.variables["time"]
    time = nc.num2date(time_var[:], 
                       units= time_var.units,
                       calendar = time_var.calendar )
    time = time.astype(str).tolist()
    time = pd.to_datetime(time)
    nearest_lon = np.argmin(np.abs(lon - lon_))
    nearest_lat = np.argmin(np.abs(lat - lat_))
    #st.write(lon[nearest_lon])
    #st.write(lat[nearest_lat])
    temperature = nc_input.variables["tas"][:,nearest_lat,nearest_lon]

    return pd.Series(temperature, index=time)


model_list = ['bcc_csm2_mr',
              'cnrm_esm2_1',
              'fgoals_g3',
              'gfdl_esm4',
              'ipsl_cm6a_lr',
              'mri_esm2_0']
  
path_data = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/Near_surface_air_temperature')) 
lon_ = 31,5
lat_ = 38,9

for model in model_list :
            
            #CMIP DATA TREATMENT - HISTORICAL 
            path_cmip_hist = f"{path_data}/historical/{model}_historical/"   
            temperature_hist_xts = nearest_point_cmip(lon_=lon_, lat_=lat_,path_nc = path_cmip_hist)
            plt.plot(temperature_hist_xts.index, temperature_hist_xts.values, label=model)

# Adding labels and title to the plot
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Near Surface Air Temperature')

# Adding legend to the plot
plt.legend()

# Displaying the plot
plt.show()
