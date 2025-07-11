"""Module for running generic functions in parallel"""

import sys
import os
import pickle
import functools
import torch
from RAMP.nnp_calculations import aimnet2sph
from RAMP.utils import calculations, multiproc


def run_func(func_name, calc=None):
    '''
    Run a function with a list of arguments and dump the results to a pickle file (to be used for multinode calculations)

    Returns:
        None
    '''
    workers = int(os.cpu_count())
    func = eval( "calculations." + func_name)
    if calc is not None:
        # Load model for calculations
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = torch.jit.load(calc)
        model.share_memory()
        #torch.set_num_threads(1)
        calc = aimnet2sph.AIMNet2Calculator(model)
        func = functools.partial(func, calc=calc)

    with open("input.pickle", "rb") as f:
        input_list = pickle.load(f)
    print(len(input_list))
    results = multiproc.parallel_run_proc(func, input_list, workers, calc)
    print("finished?")
    with open("output.pickle", "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        run_func(sys.argv[1])
    else:
        run_func(sys.argv[1], sys.argv[2])
