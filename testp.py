from multiprocessing import JoinableQueue, Process, cpu_count
import multiprocessing
import time

class Main(object):
    def __init__(self, data):
        self.data = data

    def side_thread(self, t):
        while True:                                       
            print(f"{t} started!") 
            time.sleep(t)
            print(f"{t} finished!")
            break

    def run(self):
        qi = JoinableQueue()
        processes = list()
        for _ in range(len(self.data)):
            qi.put(self.data[_])
#        for _ in range(len(self.data)):
        qi.put(None)
        while True:
            data2 = qi.get()
            if data2 is None:
                qi.task_done()
                break
            p = Process(target=self.side_thread, args=(data2,))
            p.daemon = False
            p.start()
            processes.append(p)
            qi.task_done()
        qi.join()
        print('job started')    
        for p in processes: # kernal starts again until all the processes have been finished
            p.join() 
        print('job finished')
