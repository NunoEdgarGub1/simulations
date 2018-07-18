
import multiprocessing as mp
import sys

def multi_run_wrapper(args):
    x = args[0]
    y = args[1]
    print (mp.current_process().name)
    return x+y

if __name__ == "__main__":

    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    pool = mp.Pool(3)
    print(pool.map(multi_run_wrapper,[(1,2),(2,3),(3,4)]))

