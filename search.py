#!/usr/bin/env python3
import sys
import os
from kwavelet import analysis
import minbar
from serialising import Serialising as S
from pathlib import Path
import numpy as np
import argparse

minbar.MINBAR_ROOT = '/u/kaho/minbar/minbar'
data_path='/home/kaho/mhz_QPOs_search_in_minbar'
parser=argparse.ArgumentParser(description="""restart""")
parser.add_argument('-re', default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

class Search(object):
    def __init__(self, restart=args.re, filename='search_table.gz', refile='search_results.txt'):
        if restart == True:
            o = minbar.Observations()
            b = minbar.Bursts()
            _allre = o.get_records()[o.instr_like('pca')]
            _allre.add_column('N', name='searched?')
            file = open(Path(data_path)/refile, 'w')	    
        else:
            _allre = S.load(filename=filename, path=data_path)
            file = open(Path(data_path)/refile, 'r+')	    
            file.read() # change the file object to located in a new line
        re_path = data_path+'/results/'

        # need to run in parallel later
        for i, j in enumerate(_allre[100:150]):
            _re = j
            detection = None
            if _re['searched?'] == 'N':
                try:
                    a = analysis(_re=_re, b=b, o=o, sims=0)
                    if a.bg is not None:
                        _allre['searched?'][_re.index] = 'Y'
                        for k in range(len(a.p)):
                            detection = np.any(a.np[k] > 1)
                        if detection:
                            file.write(f'{a.name} {a.obsid} yes\n')
                        try:
                            os.mkdir(re_path)
                            S.save(a, filename=f'{a.name}_{a.obsid}.gz', path=re_path)
                        except:
                            S.save(a, filename=f'{a.name}_{a.obsid}.gz', path=re_path)
                    else:
                        # skip observations with negatives
                        _allre['searched?'][_re.index] = '-'
                except:
                    _allre['searched?'][_re.index] = 'x'
        file.close()
        S.save(_allre, filename, data_path)

if __name__ == "__main__":
	Search()