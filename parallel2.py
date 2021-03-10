import os
import io
import itertools
from multiprocessing import JoinableQueue, Process, cpu_count
import multiprocessing
from kkepshot import Shot
from numpy import iterable
from kepler.code import make

class ParallelShot(Process):
    def __init__(self, qi, qo, nice=19):
        super().__init__()
        self.qi = qi
        self.qo = qo
        self.nice = nice

    def run(self):
        os.nice(self.nice)
        while True:
            data = self.qi.get()
            if data is None:
                self.qi.task_done()
                break
#            filename = data.pop('filename', None)
#            task = self.task(**data)
            s = Shot(**data)
#            if filename is not None:
#                s.save(filename)
#            elif self.qo is not None:
#                out = BytesIO()
#                s.save(out)
#                self.qo.put(out)
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
        self.data = data
        self.results = sorted(results)
