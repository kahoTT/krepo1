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
            _re = o.get_records()[o.instr_like('pca')]
        else:
            _re = S.load(filename=filename, path=data_path)
        file = open(Path(data_path)/refile, 'r+')	    
        file.read() # change the file object to located in a new line

        for i, j in enumerate(_re):
            obsid = _re[i]['obsid']
            a = analysis(obsid=obsid)
            _re.remove_row(0)
            S.save(_re, filename, data_path)

        file.close()



if __name__ == "__main__":
	Search()