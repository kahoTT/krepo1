import os
import itertools
import uuid

from multiprocessing import JoinableQueue, Process, cpu_count
from kkepshot import Shot
#from starshot.kepshot import Shot
from numpy import iterable
from kepler.code import make
from abuset import AbuSet
from isotope import ion as I
from types import FunctionType
from pathlib import Path
from isotope import ion as I, ufunc_A, ufunc_Z, ufunc_idx, ufunc_ion_from_idx
import numpy as np
from serialising import Serialising

class ParallelShot(Process):
    def __init__(self, qi, qo, nice=19, task=Shot):
        super().__init__()
        self.qi = qi
        self.qo = qo
        self.nice = nice
        self.task = task

    def run(self):
        os.nice(self.nice)
        while True:
#            breakpoint()
#            print(data)
            data = self.qi.get() # remove and reture an item from the queue
            if data is None:
                self.qi.task_done() # indicate tasks are completed
                self.qo.close()
                break
            task = self.task(**data)
            self.qo.put((data, task))
            self.qi.task_done()
            
class ParallelProcessor(Serialising):
    def __init__(self, nparallel=None, task=Shot, **kwargs):
        make() # just once
        processes = list()
        qi = JoinableQueue()
        qo = JoinableQueue()
        if nparallel is None:
            nparallel = cpu_count()
        for i in range(nparallel):
            p = ParallelShot(qi, qo, task=task)
            p.daemon = False
            p.start()
            processes.append(p)

        for k,v in kwargs.items():
            if isinstance(v, (str, dict, AbuSet, Path, FunctionType, type)):
                kwargs[k] = (v,)

        base = dict()
        data = list()
        values = list()
        keys = list()
        for k,v in kwargs.items():
            if iterable(v):
                values.append(v)
                keys.append(k)
            else:
                base[k]=v
        for vec in itertools.product(*values):
            data.append(base | dict(zip(keys, vec)))
        for d in data:
            qi.put(d)
        for _ in range(nparallel):
            qi.put(None)
        qi.close()
        qi.join()

        results = list()
        sortre = list()
        for _ in range(len(data)):
            allresults = Result(qo.get())
            results.append(allresults)
            sortre.append(allresults.data.get('Q'))
            qo.task_done()
        qo.join()
        
        results = [x for _,x in sorted(zip(sortre, results))]
        self.results = results

class Result(Serialising):
    def __init__(self, data):
        self.result = data[1]
        self.data = data[0] 
    
    def __repr__(self):
        return f'{self.data}'