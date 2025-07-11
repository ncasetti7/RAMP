"""Module for performing predictions with Chemprop model"""

import os

def make_prediction(infile, model_file, outfile):
    '''
    Make predictions with Chemprop model

    Args:
        infile (str): Path to input file
        model_file (str): Path to model file
        outfile (str): Path to output file

    Returns:
        None
    '''
    os.system("chemprop_predict --test_path " + infile + " --checkpoint_path " + model_file + " --preds_path " + outfile + " 2>/dev/null")
