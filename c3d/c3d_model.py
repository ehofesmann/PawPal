" C3D MODEL IMPLEMENTATION FOR USE WITH TENSORFLOW "


"""
Model weights found at https://github.com/hx173149/C3D-tensorflow. The model used was C3D UCF101 TF train - finetuning on UCF101 split1 use C3D sports1M model by @ hdx173149.
"""
import tensorflow as tf
import numpy      as np

from c3d.models_abstract      import Abstract_Model_Class
from c3d.layers_utils          import *

from c3d.default_preprocessing         import preprocess

class C3D(Abstract_Model_Class):

    def __init__(self, **kwargs):
        """
        Args:
            Pass all arguments on to parent class, you may not add additional arguments without modifying abstract_model_class.py and Models.py. Enter any additional initialization functionality here if desired.
        """
        super(C3D, self).__init__(**kwargs)



    def inference(self, inputs, is_training, input_dims, output_dims, seq_length, scope, dropout_rate = 0.5, return_layer=['logits'], weight_decay=0.0):
        """
        Args:
            :inputs:       Input to model of shape [Frames x Height x Width x Channels]
            :is_training:  Boolean variable indicating phase (TRAIN OR TEST)
            :input_dims:   Length of input sequence
            :output_dims:  Integer indicating total number of classes in final prediction
            :seq_length:   Length of output sequence from LSTM
            :scope:        Scope name for current model instance
            :dropout_rate: Value indicating proability of keep inputs
            :return_layer: String matching name of a layer in current model
            :weight_decay: Double value of weight decay

        Return:
            :layers[return_layer]: The requested layer's output tensor
        """

        ############################################################################
        #                       Creating C3D Network Layers                        #
        ############################################################################

        if self.verbose:
            print('Generating C3D network layers')

        # END IF

        with tf.name_scope(scope, 'c3d', [inputs]):
            layers = {}

            layers['conv1'] = conv3d_layer(input_tensor=inputs,
                    filter_dims=[3, 3, 3, 64],
                    name='c1',
                    weight_decay = weight_decay, non_linear_fn=tf.nn.relu)

            layers['pool1'] = max_pool3d_layer(layers['conv1'],
                                                 filter_dims=[1,2,2], stride_dims=[1,2,2],
                                                 name='pool1')

            layers['conv2'] = conv3d_layer(input_tensor=layers['pool1'],
                    filter_dims=[3, 3, 3, 128],
                    name='c2',
                    weight_decay = weight_decay, non_linear_fn=tf.nn.relu)

            layers['pool2'] = max_pool3d_layer(layers['conv2'],
                                                 filter_dims=[2,2,2], stride_dims=[2,2,2],
                                                 name='pool2')

            layers['conv3a'] = conv3d_layer(input_tensor=layers['pool2'],
                    filter_dims=[3, 3, 3, 256],
                    name='c3a',
                    weight_decay = weight_decay, non_linear_fn=tf.nn.relu)

            layers['conv3b'] = conv3d_layer(input_tensor=layers['conv3a'],
                    filter_dims=[3, 3, 3, 256],
                    name='c3b',
                    weight_decay = weight_decay, non_linear_fn=tf.nn.relu)

            layers['pool3'] = max_pool3d_layer(layers['conv3b'],
                                                 filter_dims=[2,2,2], stride_dims=[2,2,2],
                                                 name='pool3')

            layers['conv4a'] = conv3d_layer(input_tensor=layers['pool3'],
                    filter_dims=[3, 3, 3, 512],
                    name='c4a',
                    weight_decay = weight_decay, non_linear_fn=tf.nn.relu)

            layers['conv4b'] = conv3d_layer(input_tensor=layers['conv4a'],
                    filter_dims=[3, 3, 3, 512],
                    name='c4b',
                    weight_decay = weight_decay, non_linear_fn=tf.nn.relu)

            layers['pool4'] = max_pool3d_layer(layers['conv4b'],
                                                 filter_dims=[2,2,2], stride_dims=[2,2,2],
                                                 name='pool4')

            layers['conv5a'] = conv3d_layer(input_tensor=layers['pool4'],
                    filter_dims=[3, 3, 3, 512],
                    name='c5a',
                    weight_decay = weight_decay, non_linear_fn=tf.nn.relu)

            layers['conv5b'] = conv3d_layer(input_tensor=layers['conv5a'],
                    filter_dims=[3, 3, 3, 512],
                    name='c5b',
                    weight_decay = weight_decay, non_linear_fn=tf.nn.relu)

            layers['pool5'] = max_pool3d_layer(layers['conv5b'],
                                                 filter_dims=[2,2,2], stride_dims=[2,2,2],
                                                 name='pool5')

            if self.load_weights == 'Sports1M_finetune_UCF101':
                # Uncomment to use sports1m_finetuned_ucf101.model (aka c3d_Sports1M_finetune_UCF101.npy)
                layers['pool5'] = tf.transpose(layers['pool5'], perm=[0,1,4,2,3], name='transpose')

            layers['reshape'] = tf.reshape(layers['pool5'], shape=[tf.shape(inputs)[0], 8192], name='reshape')

            layers['dense1'] = fully_connected_layer(input_tensor=layers['reshape'],
                                                     out_dim=4096, non_linear_fn=tf.nn.relu,
                                                     name='d1', weight_decay=weight_decay)

            layers['dropout1'] = dropout(layers['dense1'], training=is_training, rate=dropout_rate)

            layers['dense2'] = fully_connected_layer(input_tensor=layers['dropout1'],
                                                     out_dim=4096, non_linear_fn=tf.nn.relu,
                                                     name='d2', weight_decay=weight_decay)

            layers['dropout2'] = dropout(layers['dense2'], training=is_training, rate=dropout_rate)

            layers['logits'] = tf.expand_dims(fully_connected_layer(input_tensor=layers['dropout2'],
                                                     out_dim=output_dims, non_linear_fn=None,
                                                     name='out', weight_decay=weight_decay), 1)

        return [layers[x] for x in return_layer]

    def load_default_weights(self):
        """
        return: Numpy dictionary containing the names and values of the weight tensors used to initialize this model
        """
        if self.load_weights == 'Sports1M_finetune_UCF101':
            return np.load('models/weights/c3d_Sports1M_finetune_UCF101.npy')

        else:
            return np.load('models/weights/c3d_Sports1M.npy')
            # REMOVE pool5 TRANSPOSE FOR SPORTS1M!!!

        # END IF

    def preprocess_tfrecords(self, input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining, video_step):
        """
        Args:
            :input_data_tensor:     Data loaded from tfrecords containing either video or clips
            :frames:                Number of frames in loaded video or clip
            :height:                Pixel height of loaded video or clip
            :width:                 Pixel width of loaded video or clip
            :channel:               Number of channels in video or clip, usually 3 (RGB)
            :input_dims:            Number of frames used in input
            :output_dims:           Integer number of classes in current dataset
            :seq_length:            Length of output sequence
            :size:                  List detailing values of height and width for final frames
            :label:                 Label for loaded data
            :is_training:           Boolean value indication phase (TRAIN OR TEST)
            :video_step:            Tensorflow variable indicating the total number of videos (not clips) that have been loaded
        """

        return preprocess(input_data_tensor, frames, height, width, channel, input_dims, output_dims, seq_length, size, label, istraining, self.input_alpha)

        # END IF


    """ Function to return loss calculated on given network """
    def loss(self, logits, labels, loss_type):
        """
        Args:
            :logits: Unscaled logits returned from final layer in model
            :labels: True labels corresponding to loaded data
        """
        labels = tf.cast(labels, tf.int64)

        cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                                    logits=logits)
        return cross_entropy_loss
