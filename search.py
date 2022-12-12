#!/usr/bin/env python3
from kwavelet import analysis
import minbar
from serialising import Serialising as S
from pathlib import Path
minbar.MINBAR_ROOT = '/u/kaho/minbar/minbar'
data_path='~/mhz_QPOs_search_in_minbar'

class Search(object):
    def __init__(self, restart=False, filename='search_table.gz', refile='search_results.txt'):
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

        # need to run in parallel later
        for i, j in enumerate(_allre):
            _re = _allre[i]
            if _re['searched?'] == 'N':
                a = analysis(_re=_re, b=b)
                _allre['searched?'][_re.index] = 'Y'
                if any(a.p > 1):
                    file.write(f'{a.name} {a.obsid} yes\n')
                else:
                    file.write(f'{a.name} {a.obsid} no\n')
        file.close()
        S.save(a, filename=f'{a.name}_{a.obsid}', path=data_path+'/results')
        S.save(_allre, filename, data_path)



if __name__ == "__main__":
	Search()