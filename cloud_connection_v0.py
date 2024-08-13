import streamlit as st
from st_files_connection import FilesConnection

conn = st.connection('s3', type=FilesConnection)

model_list = ['bcc_csm2_mr',
              'cnrm_esm2_1',
              'fgoals_g3',
              'gfdl_esm4',
              'ipsl_cm6a_lr',
              'mri_esm2_0']

for model in model_list :
    path_cmip_hist = conn.read(f"Near_surface_air_temperature/historical/{model}_historical/"
    file_hist = os.listdir(path_cmip_hist)
