import os
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

import config
import train
from training import dataset
from training import misc
from metrics import metric_base

def mixing(resume_run_id, resume_snapshot=None):
	network_pkl = misc.locate_network_pkl(resume_run_id, resume_snapshot)
    print('Loading networks from "%s"...' % network_pkl)
    G, D, Gs = misc.load_pkl(network_pkl)

    latents_1 = np.random.randn((1,*G.input_shape[1:]))
    labels_1 = [1,0,0,0,0,0]

    latents_2 = np.random.randn((1,*G.input_shape[1:]))
	labels_2 = [0,1,0,0,0,0]

	w_1 = Gs.components.mapping.get_output_for(latents_1, labels_1, is_validation=True)
	w_2 = Gs.components.mapping.get_output_for(latents_2, labels_2, is_validation=True)

	print(w1)
	print(w2)


# def main():
#     kwargs = EasyDict(train)
#     kwargs.update(G_args=G, D_args=D, G_opt_args=G_opt, D_opt_args=D_opt, G_loss_args=G_loss, D_loss_args=D_loss)
#     kwargs.update(dataset_args=dataset, sched_args=sched, grid_args=grid, metric_arg_list=metrics, tf_config=tf_config)
#     kwargs.submit_config = copy.deepcopy(submit_config)
#     kwargs.submit_config.run_dir_root = dnnlib.submission.submit.get_template_from_path(config.result_dir)
#     kwargs.submit_config.run_dir_ignore += config.run_dir_ignore
#     kwargs.submit_config.run_desc = desc
#     dnnlib.submit_run(**kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()