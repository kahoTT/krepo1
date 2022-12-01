#!/usr/bin/env python3
from kwavelet import analysis
import minbar
from serialising import Serialising as S
from pathlib import Path
minbar.MINBAR_ROOT = '/u/kaho/minbar/minbar'
data_path='~/mhz_QPOs_search_in_minbar'

class Search(object):
    def __init__(restart=False, filename='search_table.gz', refile='search_results.txt'):
        if restart == True:
            o = minbar.Observations()
            b = minbar.Bursts()
            _allre = o.get_records()[o.instr_like('pca')]
            _allre.add_column('N', name='searched?')
        else:
            _allre = S.load(filename=filename, path=data_path)
        file = open(Path(data_path)/refile, 'r+')	    
        file.read() # change the file object to located in a new line

        # need to run in parallel later
        for i, j in enumerate(_allre):
            _re = _allre[i]
            a = analysis(_re=_re, b=b)
            _allre['searched?'][_re.index] = 'Y'
            file.write('something\n')
        file.close()
        S.save(_allre, filename, data_path)



if __name__ == "__main__":
	Search()