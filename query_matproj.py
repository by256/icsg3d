"""
ICSG3D/query_matproj,py
Script for querying CIFs from the materials project
Requires a valid API key
"""
import argparse
import json
import os

from pymatgen.ext.matproj import MPRester

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Query materialsproject API and download CIFs for all structures")
    parser.add_argument('--key', metavar='key', type=str, help='Matrials Project API key')
    parser.add_argument('--name', metavar='name', type=str, help='Name of the query')
    parser.add_argument('--anonymous_formula', metavar='anonymous_formula', type=str, help='Formula of the desired materials e.g. {"A": 1.0, "B": 1.0}', default='')
    parser.add_argument('--system', metavar='system', type=str, help='Desired crystal sytems', default='cubic')

    namespace = parser.parse_args()
    API_KEY = namespace.key
    query_str = namespace.name
    save_dir = os.path.join('data', query_str)
    af = "'anonymous_formula': {'$in': [" + namespace.anonymous_formula if namespace.anonymous_formula else ''
    af += ']}'
    cs = ", 'crystal_system': '" +namespace.system+"'" if namespace.system else ''
    query = eval("{" + af + cs + "}")
    print(query)
    fields = ['task_id', 'pretty_formula', 'formation_energy_per_atom', 'cif', 'band_gap', 'diel.poly_electronic', 'diel.refractive_index', 'piezo.eij_max', 'energy_per_atom', 'elasticity.K_Voigt_Reuss_Hill', 'elasticity.G_Voigt_Reuss_Hill', 'elasticity.poisson_ratio', 'nsites']
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'cifs'), exist_ok=True)

    
    with MPRester(API_KEY) as m:
        data = m.query(criteria=query, properties=fields)

    # Save/update json
    with open(os.path.join(save_dir, query_str + '.json'), 'w+') as wf:
        json.dump(data, wf)
    
    # # Write csv with file names and properties
    csv_keys = [k for k in data[0].keys() if k != 'cif']
    csv_header = ','.join(csv_keys)
    csv_file_name = query_str + '.csv'

    with open(os.path.join(save_dir, csv_file_name), 'w+') as wf:
        wf.write(csv_header)
        wf.write('\n')

    # Write the CIFs
    for d in data:
        cif_file_name = d['task_id'] + '.cif'
        with open(os.path.join(save_dir, 'cifs', cif_file_name), 'w+') as wf:
            wf.write(d['cif'])
        with open(os.path.join(save_dir, csv_file_name), 'a+') as wf:
            wf.write(','.join([str(d[k]) for k in csv_keys]))
            wf.write('\n')
