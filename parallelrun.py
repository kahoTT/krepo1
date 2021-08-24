import os
import itertools
import uuid

from multiprocessing import JoinableQueue, Process, cpu_count
import color
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
import matplotlib.pyplot as plt
from ioncolor import IonColor

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
        sortre1 = list()
        sortre2 = list()
        for _ in range(len(data)):
            allresults = Result(qo.get())
            results.append(allresults)
            sortre1.append(allresults.data.get('Q'))
            sortre2.append(allresults.data.get('mdot'))
            qo.task_done()
        qo.join()
        
        results = [x for _,_,x in sorted(zip(sortre1, sortre2, results))]
        self.results = results

    def plot_abu(self, lim = 1e-3, mdot=None):
        i1 = slice(1, None)

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('Mass fraction')
        ax.set_ylim(1.e-3, 1.5)
        c = IonColor()

        if mdot is True:
            ax.set_xlabel('Accretion rate ($\dot{m}_{\mathrm{Edd}}$)')
            for j in range(0, len(self.results), 1):
                for i,a in self.results[j].result.abub:
                    am = np.max(a[1])
                    if am > lim:
                        ax.plot(self.results[j].data.get('mdot'), a[1], color=c(i), label=i.mpl)
                        maxabu = np.argmax(a[1])
#                        ax.text(
#                           self.y_m[i1][maxabu], a[i1][maxabu], i.mpl,
#                          ha='center', va='center', clip_on=True)
        else:
            ax.set_xlabel('Surface Flux ($\mathrm{MeV nucleon}^{-1}$)')
            for j in range(0, len(self.results), 1):
                for i,a in self.results[j].result.abub:
                    am = np.max(a[1])
                    if am > lim:
                        ax.plot(self.results[j].data.get('Q'), a[1],'.', color=c(i)) 
#                        maxabu = np.argmax(a[1])
#                        ax.text(
#                           self.results[j].data.get('Q'), a[1], i.mpl,
#                          ha='center', va='center', clip_on=True)

class Result(Serialising):
    def __init__(self, data):
        self.result = data[1]
        self.data = data[0] 
    
    def __repr__(self):
        return f'{self.data}'