import os
import io
import itertools
from multiprocessing import JoinableQueue, Process, cpu_count
import multiprocessing
from kkepshot import Shot
from numpy import iterable
from kepler.code import make
from abuset import AbuSet
from isotope import ion as I
from utils import cpickle
from types import FunctionType
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from isotope import ion as I, ufunc_A, ufunc_Z, ufunc_idx, ufunc_ion_from_idx

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
            data = self.qi.get()
            if data is None:
                self.qi.task_done()
                break
            task = self.task(**data)
            if self.qo is not None:
                self.qo.put((data, task))
            self.qi.task_done()

class ParallelProcessor(object):
    def __init__(self, nparallel=None, task=Shot, **kwargs):
        make() # only one Kepler make process before we start... WTF is that?
        processes = list()
        qi = JoinableQueue()
        qo = JoinableQueue()
        if nparallel is None:
            nparallel = cpu_count()
        for i in range(nparallel):
            p = ParallelShot(qi, qo, task=task)
#            p.run(run)
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

        # we could collect up results

        results = list()
        while not qo.empty():
            results.append(Result(*qo.get()))
            qo.task_done()
        qo.join()

        self.results = sorted(results)

# may not necessarily do the following command if better understanding the parallel code
        l_Qb = list()
        l_mdot = list()        
        l_scaled_sol_abu = list()
        l_max_mass_no = list()
        l_abu = list()
        for i in range(0, len(self.results), 1):
            l_Qb.append(results.Qb[1])
            l_mdot.append(results.mdot)
#            l_scaled_sol_abu.append(results.Qb)
            l_max_masso_no.append(results.max_mass_no[1])
            l_abu = results.abu[1]
        self.l_Qb = l_Qb
        self.l_mdot = l_mdot
#        self.l_scaled_sol_abu = l_scaled_sol_abu
        self.l_max_mass_no = np.max(l_max_mass_no)
        y = np.r_[1:max_mass_no+1]
        for j in range(0, len(l_max_mass_no), 1):
            z1 = np.zeros(len(y))
            for ii in y:
                i = int(ii)
                _int = np.where(i == ufunc_A(l_abu[j].iso))
                if _int[0].size == 0:
                    z1[i-1] = 0 
                else:
                    z1[i-1] = sum(l_abu[j].abu[_int])
            if j == 0:
                z = z1
            else:
                z = np.vstack((z,z1))

    def plot_qb(self):
        

class Results(object):
    def __init__(self, results=None):
        if results is None:
            results = list()
        self.results = sorted(results)
        self.version = VERSION
    def add(self, result):
        self.results.append(result)
        self.results.sort()
    def __add__(self, other):
        assert othert.__class__.__name__ == self.__class__.__name__
        results = self.results + other.results
        return self.__class__(results)
    def __call__(self, **kwargs):
        results = list()
        for r in self.results:
            ok = True
            for k,v in kwargs.items():
                vr = r.data[k]
                try:
                    if vr != v:
                        ok = False
                        break
                except:
                    vr = str(vr)
                    v = str(v)
                    if vr != v:
                        ok = False
                        break
            if ok:
                results.append(r)
        return Results(results)
    def __iter__(self):
        for r in self.results:
            yield r
    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.__class__(self.results[key])
        return self.results[key]
    def to_list(self):
        return self.results.copy()
    def data(self):
        return [r.data for r in self.results]
    def result(self):
        return [r.result for r in self.results]
    def __repr__(self):
        return (
            f'{self.__class__.__name__}(\n' +
            ',\n'.join(repr(r) for r in self.results) +
            ')'
            )

class Result(object):
    def __init__(self, data, result):
        self.result = result
        self.data = data
    def __getattr__(self, attr):
        if hasattr(self, 'data'):
            if attr in self.data:
                return self.data[attr]
        raise AttributeError()
    def __iter__(self):
        yield self.data
        yield self.result
    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.data).replace(" ","")} : {repr(self.result)})'
    def __lt__(self, other):
        for k,v in self.data.items():
            vo = other.data[k]
            try:
                if v < vo:
                    return True
                elif v > vo:
                    return False
            except:
                v = str(v)
                vo = str(vo)
                if v < vo:
                    return True
                elif v > vo:
                    return False
        return False
