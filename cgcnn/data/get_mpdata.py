import csv
import json
import requests

import pprint

 
#args
max_sites = 50


with open('./training-data-mp-ids.csv', 'r') as f:
    reader = csv.reader(f)
    mp_ids = [x[0] for x in list(reader)]

api_key = 'INSERT KEY HERE'
base_url = 'https://www.materialsproject.org/rest/v1/materials/'
csv_data = []
col_names = ['mp_id', 'full_formula', 'unit_cell_formula', 'formation_energy_per_atom', 
            'energy_per_atom', 'band_gap', 'bulk_modulus', 'shear_modulus', 'poisson', 
            'dielectric_constant', 'dielectric_tensor', 'refractive_index', 
            'eij_max', 'piezoelectric_tensor', 'total_magnetization', 'crystal_system', 'cif']
csv_data.append(col_names)

all_exceptions = []

for i, mp_id in enumerate(mp_ids):
    try:
        url = base_url + '{}/vasp'.format(mp_id)
        response = requests.get(url, headers={'X-API-KEY': api_key}, timeout=15)
        data = json.loads(response.content)['response'][0]
        
        if data['nsites'] > max_sites:
            continue

        material_id = data['material_id']
        full_formula = data['full_formula']
        unit_cell_formula = data['unit_cell_formula']
        formation_energy_per_atom = data['formation_energy_per_atom']
        energy_per_atom = data['energy_per_atom']
        band_gap = data['band_gap']

        # mechanical properties
        if data['elasticity']:
            if 'K_Voigt_Reuss_Hill' in data['elasticity']:
                bulk_modulus = data['elasticity']['K_Voigt_Reuss_Hill']
            else:
                bulk_modulus = None
            if 'G_Voigt_Reuss_Hill' in data['elasticity']:
                shear_modulus = data['elasticity']['G_Voigt_Reuss_Hill']
            else:
                shear_modulus = None
            if 'poisson_ratio' in data['elasticity']:
                poisson = data['elasticity']['poisson_ratio']
            else:
                poisson = None
        else:
            bulk_modulus, shear_modulus, poisson = None, None, None

        # dielectric properties
        if data['diel']:
            if data['diel']['poly_electronic']:
                dielectric_constant = data['diel']['poly_electronic']
            else:
                dielectric_constant = None
            if data['diel']['e_electronic']:
                dielectric_tensor = data['diel']['e_electronic']
            else:
                dielectric_tensor
            if data['diel']['n']:
                refractive_index = data['diel']['n']
            else:
                refractive_index = None
        else:
            dielectric_constant, dielectric_tensor, refractive_index = None, None, None

        # piezoelectric properties
        if data['piezo']:
            if data['piezo']['eij_max']:
                eij_max = data['piezo']['eij_max']
            else:
                eij_max = None
            if data['piezo']['piezoelectric_tensor']:
                piezoelectric_tensor = data['piezo']['piezoelectric_tensor']
            else:
                piezoelectric_tensor = None
        else:
            eij_max, piezoelectric_tensor = None, None

        if data['total_magnetization']:
            total_magnetization = data['total_magnetization']
        else:
            total_magnetization = None

        crystal_system = data['spacegroup']['crystal_system']

        cif = data['cif']

        csv_entry = [material_id, full_formula, unit_cell_formula, formation_energy_per_atom, 
                    energy_per_atom, band_gap, bulk_modulus, shear_modulus, poisson, 
                    dielectric_constant, dielectric_tensor, refractive_index, 
                    eij_max, piezoelectric_tensor, total_magnetization, crystal_system, cif]
        
        csv_data.append(csv_entry)

    except KeyError as e:
        pass

    print('{}/{}'.format(i+1, len(mp_ids)), end='\r', flush=True)

    
    # if i > 5:
    #     break

# print('EXCEPTIONS', all_exceptions)

with open('structure-property-data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

# import pandas as pd
# df = pd.read_csv('structure-property-data.csv')
# # print(df.head())
# print('\n\n\n', 'SHAPE', df.shape)

