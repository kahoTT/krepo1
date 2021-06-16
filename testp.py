from multiprocessing import JoinableQueue, Process, cpu_count
import time

class Main(object):

    def side_thread(self, t):
        while True:                                       
            print(f"{t} started!") 
            time.sleep(t)
            print(f"{t} finished!")
            break

    def run(self):
        processes = list()
        for i in (1,2,3,4,5):
            p = Process(target=self.side_thread, args=(i,))
            p.daemon = False
            p.start()
            processes.append(p)
        print('job started')    
#        for p in processes:
#            p.join()
        print('job finished')
