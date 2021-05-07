import threading
import time

class main_(object):
    def __init__(self, a):
        self.a = a
        t = threading.Thread(target=self.main_thread)
        t.daemon = True
        t.start()
#        time.sleep(10)
#        t.stop()

    def main_thread(self):
        for i in range(0,10,1):
            t1 = threading.Thread(target = self.side_thread)
            if self.a == 2:
                t1.deamon = True
                t1.start()
                print(threading.active_count())
                time.sleep(1)
            else:
                print(i)
                time.sleep(1)
        while True:
            if t1.is_alive() == False:
                break

    def side_thread(self):
        while True:                                       
            print("1 started!") 
            time.sleep(8)
            print("1 finished!")
            break

    
