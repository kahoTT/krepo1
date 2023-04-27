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
o = minbar.Observations()
b = minbar.Bursts()

class Search(object):
    def __init__(self, i0=0, i1=None, restart=False, filename='search_table.gz', refile='search_results.txt', nparallel=None):
        if restart == True:
            # _rawre = o.get_records()[o.instr_like('pca')]
            # _int = np.where((_allre['sflag'] == '-') | (_allre['sflag'] == 'd')|
                        #    (_allre['sflag'] == 'a') | (_allre['sflag'] == 'f')
                            # )
            # _allre = _rawre[_int]
            _allre = S.load(filename='table.gz', path=data_path)
            _allre.add_column('.', name='result')
            file = open(Path(data_path)/refile, 'w')	    
        else:
            _allre = S.load(filename=filename, path=data_path)
            file = open(Path(data_path)/refile, 'r+')	    
            file.read() # change the file object to located in a new line
        if i1 == None:
            i1 == len(_allre)        

        qi = JoinableQueue()
        qo = JoinableQueue()
        if nparallel is None:
            nparallel = cpu_count()
        for i in range(nparallel):
            p = ParallelSearch(qi, qo, task=task)
            p.daemon = False
            p.start()

        # for k, j in enumerate(data):
        for k in range(i0, i1):
            _re = _allre[k]
            if _re['result'] == '.':
                data = dict(_re=_re, k=k)
                qi.put(data)
        for _ in range(nparallel):
            qi.put(None)
        qi.close()

        for _ in range(i0, i1):
            (data, sign) = qo.get()
            k = data['k']
            _re = _allre[k]
            _re['result'] = sign
            if sign == 'Y':
            # file = open(...)
               file.write(f"{k}_{_re['name']}_{_re['obsid']} \n") 
            # file.close()
               file.flush()
            S.save(_allre, filename, data_path)
            qo.task_done()
        qi.join()
        qo.join()
        file.close()
        # S.save(_allre, filename, data_path)

def task(_re, k):
    sign = None
    re_path = data_path+'/results/'
    try:
        a = analysis(_re=_re, b=b, o=o, sims=60, _5sigma=False)
        if a.bg is not None:
            sign = 'N'
            for k in range(len(a.p)):
                if np.any(a.np[k] > 1):
                    sign = 'Y'
                    break
            # filename = "a.name_a.obsid.gz" % name
            try:
                result_dir = re_path+f"{a.name}_{a.obsid}"
                os.mkdir(result_dir)
                S.save(a, filename=f"{k}_{a.name}_{a.obsid}.gz", path=result_dir)
                a.plot_wspec(savef_path=result_dir)
            except:
                S.save(a, filename=f"{k}_{a.name}_{a.obsid}.gz", path=result_dir)
                a.plot_wspec(savef_path=result_dir)
        else:
            # skip observations with negatives
            sign = '-'
    except Exception as e:
        # raise e
        sign = 'x'
    return sign


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
            self.qi.task_done()

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="""restart""")
    parser.add_argument('-re', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('-i0', default="0", action="store")
    parser.add_argument('-i1', default="None", action="store")
    args = parser.parse_args()
    i0 = int(args.i0)
    if args.i1 == "None":
        i1 = None
    else:
        i1 = int(args.i1)
    Search(i0, i1, restart=args.re)