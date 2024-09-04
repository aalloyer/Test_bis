import cdsapi


c = cdsapi.Client()

# TEMPERATURE - uniquement sur une petite zone géographique 

#  Download document check 

period_hist =  list(range(1950, 2015))
period_hist_string = [str(num) for num in period_hist]

period_future =  list(range(2015, 2100))
period_future_string = [str(num) for num in period_future]

models = ['ipsl_cm6a_lr','bcc_csm2_mr', 'cnrm_esm2_1', 'fgoals_g3', 'gfdl_esm4', 'mri_esm2_0']

params = { #objet dict, permet de rajouter et d'enlever des éléments facilement
     'format': 'zip',
        'temporal_resolution': 'monthly',   
        'experiment': 'historical',
        'area' :[
            33, 38, 30,
            40,
            ],
        'variable': 'near_surface_air_temperature',
        # 'model': 'cnrm_cm6_1_hr',
        # 'year': ['2015'],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],

}



# HISTORICAL

params['year'] = period_hist_string

for model in models:
    params['model'] = model
    # c.retrieve('projections-cmip6', params, f'data/historical/{model}_historical.zip')
    c.retrieve('projections-cmip6', params, f'{model}_historical.zip')
    print(f"{model}_historical_downloaded.zip")
    
# PROJECTION

params['experiment'] = 'ssp3_7_0'
params['year'] = period_future_string

for model in models:
    params['model'] = model
    c.retrieve('projections-cmip6', params, f'{model}_ssp3_7_0.zip')
    print(f"{model}_ssp3_7_0_downloaded.zip")
    
##Daily maximum near surface temperature 

models = ['ipsl_cm6a_lr', 'cnrm_esm2_1', 'fgoals_g3', 'gfdl_esm4', 'mri_esm2_0']
params['variable'] = 'daily_maximum_near_surface_air_temperature'

# historical

params['experiment'] = 'historical'
params['year'] = period_hist_string

for model in models:
    params['model'] = model
    # c.retrieve('projections-cmip6', params, f'data/historical/{model}_historical.zip')
    c.retrieve('projections-cmip6', params, f'{model}_historical.zip')
    print(f"{model}_historical_downloaded.zip")
    
# projection

params['experiment'] = 'ssp3_7_0'
params['year'] = period_future_string

for model in models:
    params['model'] = model
    c.retrieve('projections-cmip6', params, f'{model}_ssp3_7_0.zip')
    print(f"{model}_ssp3_7_0_downloaded.zip")
    
# Dézippage - code brut


import zipfile
import os


path = 'data'

list_files =['Near_surface_air_temperature', 'Daily_maximum_near_surface_air_temperature']
scenarios = [ 'historical', 'ssp3_7_0']
models = ['ipsl_cm6a_lr', 'cnrm_esm2_1', 'fgoals_g3', 'gfdl_esm4', 'mri_esm2_0','bcc_csm2_mr']

scenario = 'ssp3_7_0'
file = 'Daily_maximum_near_surface_air_temperature'

for file in list_files:
    for scenario in scenarios :
        for model in models:
        
             # file creation
            new_dir = os.path.join(f'{path}/{file}/{scenario}', f'{model}_{scenario}')
            os.makedirs(new_dir, exist_ok=True)
        
            # looking inside the zipped folder and list .nc files
            with zipfile.ZipFile(f'{path}/{file}/{scenario}/{model}_{scenario}.zip', 'r') as zip_ref:
                file_list = zip_ref.namelist()
                nc_files = [f for f in file_list if f.endswith('.nc')]
                
                # if presence of .nc file, we extract it and place it in the new file
                if nc_files:
                    with zipfile.ZipFile(f'{path}/{file}/{scenario}/{model}_{scenario}.zip', 'r') as zip_ref:
                        zip_ref.extract(nc_files[0]) # le chemin d'accès ne fonctionnait pas (ajout de //) donc on fait ça dans le working directory 
                        
# historical - historical plays the same role as scenario above 

scenario = 'historical'

for model in models:
    
    # file creation
    new_dir = os.path.join(f'{path}/{scenario}', f'{model}_{scenario}')
    os.makedirs(new_dir, exist_ok=True)
    
    # looking inside the zipped folder and list .nc files
    with zipfile.ZipFile(f'{path}/{scenario}/{model}_{scenario}.zip', 'r') as zip_ref:
        file_list = zip_ref.namelist()
        nc_files = [f for f in file_list if f.endswith('.nc')]
        
        # if presence of .nc file, we extract it and place it in the new file
        if nc_files:
            with zipfile.ZipFile(f'{path}/{scenario}/{model}_{scenario}.zip', 'r') as zip_ref:
                zip_ref.extract(nc_files[0], f'{path}/{scenario}/{model}_{scenario}')

# After downloading files for new years

year = 2014

for model in models:
    
    # file creation
    new_dir = os.path.join(f'{path}/{scenario}', f'{model}_{scenario}')
    os.makedirs(new_dir, exist_ok=True)
    
    # looking inside the zipped folder and list .nc files
    with zipfile.ZipFile(f'{path}/{scenario}/{model}_{scenario}_{year}.zip', 'r') as zip_ref:
        file_list = zip_ref.namelist()
        nc_files = [f for f in file_list if f.endswith('.nc')]
        
        # if presence of .nc file, we extract it and place it in the new file
        if nc_files:
            with zipfile.ZipFile(f'{path}/{scenario}/{model}_{scenario}_{year}.zip', 'r') as zip_ref:
                zip_ref.extract(nc_files[0], f'{path}/{scenario}/{model}_{scenario}')
                
                
# ERA5

import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': '2m_temperature',
        'year': [
            '1940', '1941', '1942',
            '1943', '1944', '1945',
            '1946', '1947', '1948',
            '1949', '1950', '1951',
            '1952', '1953', '1954',
            '1955', '1956', '1957',
            '1958', '1959', '1960',
            '1961', '1962', '1963',
            '1964', '1965', '1966',
            '1967', '1968', '1969',
            '1970', '1971', '1972',
            '1973', '1974', '1975',
            '1976', '1977', '1978',
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021', '2022', '2023',
            '2024',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
    },
    'download.nc')