#!/usr/bin/env python3
import sys
import os
from kwavelet import analysis
import minbar
from serialising import Serialising as S
from pathlib import Path
import numpy as np
import argparse
from multiprocessing import JoinableQueue, Process, cpu_count
from functools import partial

minbar.MINBAR_ROOT = '/u/kaho/minbar/minbar'
data_path='/home/kaho/mhz_QPOs_search_in_minbar'
parser=argparse.ArgumentParser(description="""restart""")
parser.add_argument('-re', default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

o = minbar.Observations()
b = minbar.Bursts()

class Search(object):
    def __init__(self, restart=args.re, filename='search_table.gz', refile='search_results.txt', nparallel=None):
        if restart == True:
            _allre = o.get_records()[o.instr_like('pca')]
            _allre.add_column('N', name='searched?')
            file = open(Path(data_path)/refile, 'w')	    
        else:
            _allre = S.load(filename=filename, path=data_path)
            file = open(Path(data_path)/refile, 'r+')	    
            file.read() # change the file object to located in a new line

        qi = JoinableQueue()
        qo = JoinableQueue()
        if nparallel is None:
            nparallel = cpu_count()
        for i in range(nparallel):
            p = ParallelSearch(qi, qo, task=analysis(b=b, o=o, sims=0))
            p.daemon = False
            p.start()

        for k, j in enumerate(_allre[100:150]):
            _re = j
            if _re['searched?'] == 'N':
                qi.put(_re)
        for _ in range(nparallel):
            qi.put(None)
        qi.close()

        file.close()
        S.save(_allre, filename, data_path)

def task():
    detection = None
    sign = None
    re_path = data_path+'/results/'
    try:
        a = analysis(_re=_re, b=b, o=o, sims=0)
        if a.bg is not None:
            sign = 'Y'
            for k in range(len(a.p)):
                detection = np.any(a.np[k] > 1)
            try:
                os.mkdir(re_path)
                S.save(a, filename=f'{a.name}_{a.obsid}.gz', path=re_path)
            except:
                S.save(a, filename=f'{a.name}_{a.obsid}.gz', path=re_path)
        else:
            # skip observations with negatives
            sign = '-'
    except:
        sign = 'x'
    return sign, detection


class ParallelSearch(Process):
    def __init__(self, qi, qo, nice=19, task=None):
        super().__init__()
        self.qi = qi
        self.qo = qo
        self.nice = nice
        self.task = task

    def run(self):
        os.nice(self.nice)
        while True:
            data = self.qi.get() # remove and reture an item from the queue
            if data is None:
                self.qi.task_done() # indicate tasks are completed
                self.qo.close()
                break
            task = self.task(**data)
            self.qo.put((data, task))
            # self.qi.task_done()

if __name__ == "__main__":
	Search()