import streamlit as st
import cdsapi
import os

def download_data(model, period, experiment):

    key=st.secrets["key"]
    c = cdsapi.Client("https://cds.climate.copernicus.eu/api/v2", key)
   
    params = {
        'format': 'zip',
        'temporal_resolution': 'monthly',
        'experiment': experiment,
        'area': [38, #north
                 33, #west
                 33, #south
                 38], #east
        'variable': 'near_surface_air_temperature',
        'model': model,
        'year': period,
        'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
    }
    filename = f'{model}_{experiment}.zip'
    file = c.retrieve('projections-cmip6', params, filename)

    return file

# Sélection du modèle
model = st.selectbox('Select a climate model:', ['ipsl_cm6a_lr', 'bcc_csm2_mr', 'cnrm_esm2_1', 'fgoals_g3', 'gfdl_esm4', 'mri_esm2_0'])

# Sélection de la période
period = st.selectbox('Select the period:', ['historical', 'future'])
if period == 'historical':
    period_years = list(range(1950, 2015))
else:
    period_years = list(range(2015, 2100))
period_years_string = [str(year) for year in period_years]

# Bouton de téléchargement
if st.button('Download Data'):
    with st.spinner('Downloading...'):
        filename = download_data(model, period_years_string, period)
        st.success('Downloaded successfully!')
        # Lien de téléchargement
        with open(filename, "rb") as file:
            btn = st.download_button(
                label="Download ZIP",
                data=file,
                file_name=filename,
                mime="application/zip"
            )