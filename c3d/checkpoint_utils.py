import os

import numpy      as np
import tensorflow as tf

def load_checkpoint(loaded_checkpoint):
    """
    Function to checkpoint file (both ckpt text file, numpy and dat file)
    Args:
        :model:                 String indicating selected model
        :dataset:               String indicating selected dataset
        :experiment_name:       Name of experiment folder
        :loaded_checkpoint:     Number of the checkpoint to be loaded, -1 loads the most recent checkpoint
        :preproc_method:     The preprocessing method to use, default, cvr, rr, sr, or any other custom preprocessing

    Return:
        numpy containing model parameters, global step and learning rate saved values.
    """
    filename = 'checkpoint-'+str(loaded_checkpoint)
    try:
        ckpt = np.load(os.path.join('c3d',filename+'.npy'), encoding='bytes')

        return ckpt

    except:
        print("Failed to load saved checkpoint numpy file: ", filename)
        exit()



def _assign_tensors(sess, curr_dict, tensor_name):
    """
    Function recursively assigns model parameters their values from a given dictionary
    Args:
        :sess:        Tensorflow session instance
        :curr_dict:   Dictionary containing model parameter values
        :tensor_name: String indicating name of tensor to be assigned values

    Return:
       Does not return anything
    """
    try:
        if type(curr_dict) == type({}):
            for key in curr_dict.keys():
                _assign_tensors(sess, curr_dict[key], tensor_name+'/'+key)

            # END FOR

        else:
            if ':' not in tensor_name:
                tensor_name = tensor_name + ':0'

            # END IF

            if 'weights' in tensor_name:
                tensor_name = tensor_name.replace('weights', 'kernel')

            # END IF

            sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name(tensor_name), curr_dict))

        # END IF

    except:
        if 'Momentum' not in tensor_name:
            print("Notice: Tensor " + tensor_name + " could not be assigned properly. The tensors' default initializer will be used if possible. Verify the shape and name of the tensor.")

        #END IF

    # END TRY


def initialize_from_dict(sess, data_dict):
    """
    Function initializes model parameters from value given in a dictionary
    Args:
        :sess:        Tensorflow session instance
        :data_dict:   Dictionary containing model parameter values

    Return:
       Does not return anything
    """

    print('Initializing model weights...')
    try:
        data_dict = data_dict.tolist()
        for key in data_dict.keys():
            print(key)
            _assign_tensors(sess, data_dict[key], key)

        # END FOR

    except:
        print("Error: Failed to initialize saved weights. Ensure naming convention in saved weights matches the defined model.")
        exit()

    # END TRY
