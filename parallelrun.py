import os
import itertools
import uuid

from multiprocessing import JoinableQueue, Process, cpu_count
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
import numpy as np
import matplotlib.pyplot as plt
from utils import cpickle
from starshot.base import Base

def abu_format(parameters):
    parameters = parameters.copy()
    abu = parameters.get('abu', None)
    if abu is not None:
        Abu = AbuSet(abu)
        parameters['Abu'] = repr(Abu).replace(' ','')
        abu = Abu.as_dict(massfrac=True)
        parameters['ABU'] = (
            '(' +
            ','.join(f'{I(k).Name():s}={v:<8G}'.strip() for k,v in abu.items())
            + ')'
            )
    return parameters

def save_model(data, task, filename, path):
    data = abu_format(data)
    if isinstance(filename, FunctionType):
        filename = filename(**data)
    elif isinstance(filename, str):
        filename = filename.format(**data)
    elif filename is Ellipsis:
        filename = '_'.join(f'{k}={v:<5}'.replace(' ','') for k,v in data.items() if not k in ('abu','Abu')) +'.pickle.xz'
    if hasattr(task, 'save'):
        task.save(filename, path)
    else:
        if filename is None:
            filename = uuid.uuid1().hex + '.pickle.xz'
        if path is not None:
            filename = Path(path) / filename
        filename = Path(filename).expanduser()
        cpickle(task, filename)

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
            data = self.qi.get() # remove and reture an item from the queue
            if data is None:
                self.qi.task_done() # indicate tasks are completed
                self.qo.close()
                break
            path = data.pop('path', None)
            filename = data.pop('filename', None)

            task = self.task(**data)

            if filename is not None or path is not None:
                save_model(data, task, filename, path)
            elif self.qo is not None:
                self.qo.put((data, task))
            self.qi.task_done()

class ParallelProcessor(Base):
    def __init__(self, nparallel=None, task=Shot, **kwargs):
        make() # only one Kepler make process before we start... WTF is that?
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

        # we could collect up results

        results = list()
        for _ in range(len(data)):
            results.append(Result(*qo.get()))
        qo.task_done()
        qo.join()

        self.results = sorted(results)

# may not necessarily do the following command if better understanding the parallel code
#    def list_result(object):
#        l_Qb = list()
#        l_Lb = list()
#        l_mdot = list()        
#        l_scaled_sol_abu = list()
#        l_max_mass_no = list()
#        l_abu = list()
#        for i in range(0, len(results), 1):
#            l_Qb.append(self.results[i].result.Qb)
#            l_Lb.append(self.results[i].result.Lb)
#            l_mdot.append(self.results[i].result.mdot)
#            l_scaled_sol_abu.append(results.Qb)
#            l_max_mass_no.append(self.results[i].result.max_mass_no[1])
#            l_abu.append(self.results[i].result.abu[1])
#        self.l_Qb = l_Qb
#        self.l_Lb = l_Lb
#        self.l_mdot = l_mdot
#        self.l_abu = l_abu
#        self.l_scaled_sol_abu = l_scaled_sol_abu
#        self.l_max_mass_no = np.max(l_max_mass_no)
#        y = np.r_[1:self.l_max_mass_no+1]
#        self.y = y
#        for j in range(0, len(l_max_mass_no), 1):
#            z1 = np.zeros(len(y))
#            for ii in y:
#                i = int(ii)
#                _int = np.where(i == ufunc_A(l_abu[j].iso))
#                if _int[0].size == 0:
#                    z1[i-1] = 0 
#                else:
#                    z1[i-1] = sum(l_abu[j].abu[_int])
#            if j == 0:
#                z = z1
#            else:
#                z = np.vstack((z,z1))
#        self.z = z

#    def plot_lmap(self, unit=None):
#        fig, ax = plt.subplots()
#        self.fig = fig
#        self.ax = ax 
#        ax.set_ylabel('Mass number')
#        if unit == 'q':
#            ax.set_xlabel('Bottom heat flux ($\mathrm{MeV\,nucleons}^{-1}\,\mathrm{s}^{-1}$)')
#            pcm = ax.pcolor(self.l_Qb, self.y, self.z.T, cmap = 'binary', norm=colors.LogNorm(vmin = 1e-10, vmax = max(map(max, self.z.T))))
#        else:
#            ax.set_xlabel('Bottom heat flux ($\mathrm{erg\,s}^{-1}$)')
#            pcm = ax.pcolor(self.l_Lb, self.y, self.z.T, cmap = 'binary', norm=colors.LogNorm(vmin = 1e-10, vmax = max(map(max, self.z.T))))
#        fig.colorbar(pcm, ax=ax, extend='max')
#        ytick_labels = ax.yaxis.get_ticklocs()
#        yticks = ax.yaxis.get_ticklocs()
#        ax.set_yticks(yticks+.5)
#        ax.set_yticklabels(yticks)

    def __iter__(self):
        for r in self.results:
            yield r

class Results(Base):
    def __init__(self, results=None):
        if results is None:
            results = list()
        self.results = sorted(results)
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

    def save_results(self, filename=None, path=None):
        """Bulk-save all completed results/tasks.

        If you provide a 'path' then output files will be written into
        that directory.

        'filename' should be a string with format instructions so that
        variables in there can be replaced from the keywords.
        e.g., 'Q={Q:g}_Mdot={ymax:g}_{Abu}.xz'

        If no filename is provided, the UUID will be used.

        'Abu' will be replaced by AbuSet object representation
        'ABU' will use nice dictionary representation
        """
        for p,r in self.results:
            save_model(p, r, filename, path)

class Result(object):
    def __init__(self, data, result):
        self.result = result
        self.data = data
    def __getattr__(self, attr):
        if attr != 'data' and hasattr(self, 'data'):
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
