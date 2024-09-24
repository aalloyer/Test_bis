import setuptools
import streamlit as st
import cdsapi
import os
import zipfile # RAJOUTER 

#pour l'instant uniquement pour near surface data
#c = cdsapi.Client()

# afficher hypothèses sur longitude et latitude
lon_site = st.sidebar.number_input("Longitude (from 0° to 360°) : ", step=0.1)
lat_site = st.sidebar.number_input("Latitude (from -90° [south] to 90° [north])", step=0.1) # intervalle ? 

# RAJOUTER LES IDENTIFIANTS API
# attention à pas déborder des limites

model_list = ['ipsl_cm6a_lr', 'bcc_csm2_mr', 'cnrm_esm2_1', 'fgoals_g3', 'gfdl_esm4', 'mri_esm2_0']

historical =list(range(1950, 2015))
historical_str = [str(year) for year in historical]
historical_duo =['historical', historical_str]

future = list(range(2015, 2100))
future_str = [str(year) for year in future]
future_duo = ['ssp3_7_0', future_str]

period_str =[historical_str, future_str]

def download_data(model_list, lon_site, lat_site, period, experiment):
    north = min(lat_site + 2, 90)
    west = max(lon_site +2, -180)
    south = max(lat_site -2, -90)
    east = min(lon_site -2, 180)
    key=st.secrets["key"]
    c = cdsapi.Client("https://cds.climate.copernicus.eu/api/v2", key)
    
    all_files = []
    
    for model in model_list :
        params = {
            'format': 'zip',
            'temporal_resolution': 'monthly',
            'experiment': experiment,
            'area': [north, #north
                     west, #west
                     south, #south
                     east], #east
            'variable': 'near_surface_air_temperature',
            'model': model,
            'year': period,
            'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
        }
        filename = f'{model}_{experiment}.zip'
        file = c.retrieve('projections-cmip6', params, filename)
        all_files.append([filename, file])
    
    return all_files

# Streamlit interface
st.title('Climate Data Download App')

if st.button('Lancer le téléchargement'):
    data = []
    for period in period_str :
        for model in model_list :
        # data.append(download_data(model_list, lon_site, lat_site, period[2], period[1]))
            zip_files_list = download_data(model_list, lon_site, lat_site, period[1], period[0])
            for file_info in zip_files_list :
                zip_file_name, file = file_info
                with zipfile.ZipFile(zip_file_name, "r") as zip_ref:
                    nc_filename = zip_file_name.replace('.zip', '.nc')
                    zip_ref.extract(nc_filename, path="extracted_files")
                    data.append(os.path.join("extracted_files", nc_filename))

    st.write(f'{data}')

# Sélection du modèle
# model = st.selectbox('Select a climate model:', ['ipsl_cm6a_lr', 'bcc_csm2_mr', 'cnrm_esm2_1', 'fgoals_g3', 'gfdl_esm4', 'mri_esm2_0'])

# Sélection de la période
# period = st.selectbox('Select the period:', ['historical', 'future'])
# if period == 'historical':
#     period_years = list(range(1950, 2015))
# else:
#     period_years = list(range(2015, 2100))
# period_years_string = [str(year) for year in period_years]

# Bouton de téléchargement
# if st.button('Download Data'):
#     with st.spinner('Downloading...'):
#         filename = download_data(model, period_years_string, period)
#         st.success('Downloaded successfully!')

#         # Lien de téléchargement
#         with open(filename, "rb") as file:
#             btn = st.download_button(
#                 label="Download ZIP",
#                 data=file,
#                 file_name=filename,
#                 mime="application/zip"
#             )
