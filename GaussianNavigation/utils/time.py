
import time


def print_run_time(func):  
    def wrapper(*args, **kw):  
        local_time = time.time()  
        temp = func(*args, **kw) 
        print ('current Function [%s] run time is %.2f' % (func.__name__ ,time.time() - local_time))
        return temp
    return wrapper 
