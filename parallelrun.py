import os
import itertools
import uuid

from multiprocessing import JoinableQueue, Process, cpu_count
import color
from kkepshot import Shot
#from starshot.kepshot import Shot
from numpy import iterable
from kepler.code import make
from abuset import AbuSet, AbuData
from isotope import ion as I
from types import FunctionType
from pathlib import Path
from isotope import ion as I, ufunc_A, ufunc_Z, ufunc_idx, ufunc_ion_from_idx, ioncacheza, ufunc_idx_ZA
import numpy as np
from serialising import Serialising
import matplotlib.pyplot as plt
from ioncolor import IonColor
from utils import index1d

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

        results = list()
        sortre1 = list()
        sortre2 = list()
        for _ in range(len(data)):
            allresults = Result(qo.get())
            results.append(allresults)
            sortre1.append(allresults.data.get('Q'))
            sortre2.append(allresults.data.get('mdot'))
            qo.task_done()
        qi.join()
        qo.join()
        
        results = [x for _,_,x in sorted(zip(sortre1, sortre2, results))]
        self.results = results
        self.Q = sorted(sortre1)
        self.mdot = sorted(sortre2)

        # get results
        Qb = list()
#        C12abu = list()
        for i in results:
            Qb.append(i.result.Qb)
#            C12abd.append(i.result.)
        a = len(np.unique(self.Q))
        b = len(np.unique(self.mdot))
        Qba = np.array(Qb).reshape(b, a)
        self.Qb = Qb
        self.Qba = Qba

        # Qba = np.empty((len(self.Q), len(self.mdot)))
        # k = 0
        # for j in range(len(self.mdot)):
        #     for i in range(len(self.Q)):
        #         Qba[j,i] = Qb[k]
        #         k += 1
        # self.Qba = Qba

        # map all ions (from Shot)
        print(f'[{self.__class__.__name__}] Mapping ions....')
        ions = set()
        ionsa = list()
        for b in self.results:
            idx = ufunc_idx(b.result.abub.ions)
            ions |= set(idx)
            ionsa.append(idx)
        ions = np.array(sorted(ions))
        nions = len(ions)
        abub = np.zeros((len(self.results), nions))
        for i, b in enumerate(self.results):
            ii = index1d(ionsa[i], ions)
            abub[i,ii] = b.result.abub.data[1]
        # ions = ufunc_ion_from_idx(ions)
        ions = ioncacheza(*ufunc_idx_ZA(ions))
        self.abub = AbuData(
            abub,
            ions,
            molfrac = False,
            )

    def plot_qqb(self, mdot = None):
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        if mdot == True:
            ax.plot(self.mdot, self.Qb, '.')
            ax.set_xlabel('Accretion rate ($\dot{M}_{\mathrm{Edd}}$)')
        else:
            ax.plot(self.Q, self.Qb, '.')
            ax.set_xlabel('Surface Flux ($\mathrm{MeV\,nucleon}^{-1}$)')
        ax.set_ylabel('Base Flux ($\mathrm{MeV\,nucleon}^{-1}$)')

    def plot_abu(self, lim = 1e-5, mdot=None, surfaceflux = False):
        i1 = slice(1, None)

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

        ax.set_yscale('log')
        ax.set_ylabel('Mass fraction')
        ax.set_ylim(1.e-5, 1.5)
        c = IonColor()

        if mdot is True:
            ax.set_xlabel('Accretion rate ($\dot{M}_{\mathrm{Edd}}$)')
        else:
            ax.set_xlabel('Surface Flux ($\mathrm{MeV\,nucleon}^{-1}$)')
        for i,a in self.abub:
            am = np.max(a)
            if am > lim:
                maxabu = np.argmax(a)
                if mdot is True:
                    ax.plot(self.mdot, a, color=c(i)) 
                    ax.text(
                            self.mdot[maxabu], a[maxabu], i.mpl, color=c(i),
                            ha='center', va='center', clip_on=True, size=12)
                else:
                    ax.plot(self.Q, a, color=c(i)) 
                    ax.text(
                            self.Q[maxabu], a[maxabu], i.mpl, color=c(i),  
                            ha='center', va='center', clip_on=True, size=12)

    def plot_contour_Q_mdot_Qb(self, Q=None, mdot=None, Qb=None):
        fig, ax = plt.subplots()
        # self.fig = fig
        # self.ax = ax

        if Q is None:
            Q = self.Q
        if mdot is None:
            mdot = self.mdot
        if Qb is None:
            qba = self.Qba
        else:
            a = len(np.unique(Q))
            b = len(np.unique(mdot))
            qba = np.array(Qb).reshape(b, a)

        cm = ax.contourf(np.unique(Q), np.unique(mdot), qba, cmap=plt.cm.viridis)
        plt.colorbar
        fig.colorbar(cm, ax=ax)
        ax.set_ylabel('$\dot{m}_{\mathrm{Edd}}$')
        ax.set_xlabel('Surface luminosity / $\dot{m}$ (MeV/u)')



class Result(Serialising):
    def __init__(self, data):
        self.result = data[1]
        self.data = data[0] 
    
    def __repr__(self):
        return f'{self.data}'