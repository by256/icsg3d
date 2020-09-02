import csv
import json
import requests
import argparse
import pandas as pd
from pprint import pprint
from pymatgen import MPRester


parser = argparse.ArgumentParser(description="Query materialsproject API and download CIFs for training the CGCNN")
parser.add_argument('--key', metavar='key', type=str, help='Matrials Project API key')
parser.add_argument('--mp-ids', metavar='ids', type=str, help='path to file containing mp-ids to query.', default='./training-data-mp-ids.csv')
parser.add_argument('--saveto', metavar='saveto', type=str, help='target path to save data', default='./structure-property-data.csv')
namespace = parser.parse_args()

mp_ids = pd.read_csv(namespace.mp_ids, header=None).drop_duplicates()
mp_ids = list(mp_ids[0])

api_key = namespace.key
mpr = MPRester(api_key)

criteria = {'material_id': {'$in': mp_ids}}

properties = ['material_id', 'full_formula', 'pretty_formula', 'nsites', 'unit_cell_formula', 
            'formation_energy_per_atom', 
            'energy_per_atom', 'e_above_hull', 'band_gap', 'elasticity.K_Voigt_Reuss_Hill', 
            'elasticity.G_Voigt_Reuss_Hill', 'elasticity.poisson_ratio', 'diel.poly_electronic', 
            'diel.e_electronic', 'diel.n', 'piezo.eij_max', 'piezo.piezoelectric_tensor', 
            'total_magnetization', 'crystal_system', 'cif']

entries = mpr.query(criteria=criteria, properties=properties)

col_names = ['mp_id', 'full_formula', 'pretty_formula',  'nsites', 'unit_cell_formula', 
             'formation_energy_per_atom', 
            'energy_per_atom', 'e_above_hull', 'band_gap', 'bulk_modulus', 'shear_modulus', 'poisson', 
            'dielectric_constant', 'dielectric_tensor', 'refractive_index', 
            'eij_max', 'piezoelectric_tensor', 'total_magnetization', 'crystal_system', 'cif']

col_name_mapper = dict(zip(properties, col_names))

df = pd.DataFrame(entries).rename(mapper=col_name_mapper, axis=1)
df.to_csv(namespace.saveto, index=False)
