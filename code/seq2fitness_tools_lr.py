# Import necessary libraries here
from os.path import abspath, join
import sys

#module_path = abspath("nn4dms_nn-extrapolate/code")
#if module_path not in sys.path:
#    sys.path.append(module_path)


import numpy as np
import encode as encoding
import inference_lr as inference  # use this inference for LR model only - removed training_ph

# define relative path to pretrained_model_dir - this is fixed - it is set relative to
# working directory set in 02_run_sa.py
pretrained_model_dir = "../nn-extrapolation-models/pretrained_models"
# define model path - change this to select model
model_file_path = "other_models/gb1_lr.pb"

# import inference  # use this inference for any model that is not LR
class seq2fitness_handler:
    def __init__(self):
        
        model_path = join(pretrained_model_dir, model_file_path)
        self.model = inference.restore_sess_from_pb(model_path)

    def seq2fitness(self, seq):
        encoded_seq = encoding.encode(encoding="one_hot,aa_index", char_seqs=[seq])

        # since model is LR, use `run_inference_lr` - otherwise use `run_inference`
        prediction = inference.run_inference_lr(encoded_data=encoded_seq, sess=self.model)
        
        return prediction
