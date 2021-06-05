from multiprocessing import JoinableQueue, Process, cpu_count
import time

class _main(object):

    def side_thread(self, t):
        while True:                                       
            print("1 started!") 
            time.sleep(t)
            print("1 finished!")
            break

    def _run(self):
        t = 1
        p = Process(target=self.side_thread, args=(t,))
        p.daemon = False
        p.start()
        p.join()