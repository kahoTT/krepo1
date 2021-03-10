import os
import io
import itertools
from multiprocessing import JoinableQueue, Process, cpu_count
import multiprocessing
from kkepshot import Shot
from numpy import iterable
from kepler.code import make

class ParallelShot(Process):
    def __init__(self, qi, qo, nice=19, task=Shot, run=None):
        super().__init__()
        self.qi = qi
        self.qo = qo
        self.nice = nice
        self.task = task
        self.run = run

    def run(self):
        os.nice(self.nice)
        while self.run:
            data = self.qi.get()
            if data is None:
                self.qi.task_done()
                break
            task = self.task(**data)
            if self.qo is not None:
                self.qo.put((data, task))
            self.qi.task_done()

class ParallelProcessor(object):
    def __init__(self, nparallel=None, task=Shot, run=True, **kwargs):
        make() # only one Kepler make process before we start... WTF is that?
        processes = list()
        qi = JoinableQueue()
        qo = JoinableQueue()
        if nparallel is None:
            nparallel = cpu_count()
        for i in range(nparallel):
            p = ParallelShot(qi, qo, task=task, run=run)
            p.daemon = True
            p.start()
            processes.append(p)

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
            results.append(*qo.get())
            qo.task_done()
        qo.join()

        self.results = sorted(results)
