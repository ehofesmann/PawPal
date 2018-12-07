from darkflow.net.build import TFNet
import cv2
import os
import numpy as np
import argparse
import tensorflow as tf
from c3d.checkpoint_utils import load_checkpoint, initialize_from_dict
from c3d.c3d_model import C3D


#os.environ['CUDA_VISIBLE_DEVICES'] = ''
parser = argparse.ArgumentParser()
parser.add_argument('--vidnum', action='store', type=int, required=True)
args = parser.parse_args()
vidnum = args.vidnum


use_softmax = True


ckpt = load_checkpoint(532)

model = C3D(modelName='c3d', inputDims=16, outputDims=2, expName='c3d_dog_frozen_3_newdat', numVids=1, clipLength=16)


input_data_tensor = tf.placeholder(tf.float32, shape=(1,16,112,112,3))
istraining = False
input_dims = 16
output_dims = 2
seq_length = 1
scope = "my_scope"
batch_size=1
logits = model.inference(input_data_tensor, istraining, input_dims, output_dims, seq_length, scope)[0]
logits_shape = logits.get_shape().as_list()
if (logits_shape[0] != batch_size or logits_shape[1] != seq_length or logits_shape[2] != output_dims) and return_layer[0] == 'logits':
    logits = tf.reshape(logits, [batch_size, seq_length, output_dims])

# END IF

if use_softmax:
    logits = tf.nn.softmax(logits)


# TF session setup
config  = tf.ConfigProto(allow_soft_placement=True)
sess    = tf.Session(config=config)
init    = (tf.global_variables_initializer(), tf.local_variables_initializer())


# Variables get randomly initialized into tf graph
sess.run(init)

initialize_from_dict(sess, ckpt)
del ckpt


for video in [np.zeros((1,16,112,112,3))]:
    output_preds = sess.run([logits], feed_dict={input_data_tensor: video})
    import pdb; pdb.set_trace()
	





def expand_bbox(shape, minx, miny, w, h, expansion_rate=1.5):

    h2 = min(h*expansion_rate, shape[0])
    w2 = min(w*expansion_rate, shape[1]) 

    minx2 = max(minx - w/2, 0)
    miny2 = max(miny - h/2, 0)

    return int(minx2), int(miny2), int(w2), int(h2)





def process_video(video, class_path, act):
    count = 0
    video_path = class_path + '/' + video
    if os.path.isfile(video_path):
        video_result =[]
        video_as_array = []
        # Capture each Video 
        current = cv2.VideoCapture(video_path)
        framecount = int(current.get(cv2.CAP_PROP_FRAME_COUNT))
        batches = int(framecount/16)
        to_process = []
        # To interpolate for missing dog frames
        buffer_store = []
        print('framecount: ',framecount)
        for i in range(framecount):
            _, frame = current.read()
            # print(frame.shape)
            # Appends the frames
            video_as_array.append(frame)
            result = tfnet.return_predict(frame)
            # Appends the bounding box results
            video_result.append(result)

            found_one_dog = False
            dog_process = []
            dob_buffer = []
            max_patch = 0
            max_buff = [-1, i]
            max_conf = 0

            for cl in result:
                if cl['label'] == 'dog':
                    minx = cl['topleft']['x']
                    miny = cl['topleft']['y']
                    maxx = cl['bottomright']['x']
                    maxy = cl['bottomright']['y']
                    w = maxx - minx
                    h = maxy - miny
                    # Store these for interpolation
                    minx, miny, w, h = expand_bbox(video_as_array[i].shape, minx, miny, w, h)
                    buff = [i, minx, miny, w, h]
                    patch = cv2.resize(video_as_array[i][miny:(miny+h), minx:(minx+w), :], (112,112), interpolation = cv2.INTER_CUBIC)
                    if max_conf < cl['confidence']:
                        max_conf = cl['confidence']
                        max_patch = patch
                        max_buff = buff

            to_process.append(max_patch)
            # Also store the video frame where dog label doesn't exists
            buffer_store.append(max_buff)
            

            if (i%16 == 15):
                # Process all 16 frames and fill in missing frames to get outputs for C3D
                for k in range(16):
                    flag = 0 
                    if buffer_store[k][0] == -1 and k!=0:
                        prev = buffer_store[k-1]
                        fr = buffer_store[k][1]
                        while buffer_store[k+flag][0] == -1 and k+flag<15:
                            flag += 1 
                        if buffer_store[k+flag][0] !=-1 and k+flag < 16: 
                            nex = buffer_store[k+flag]
                            # Interpolate the start coordinates and dimensions
                            minx = int(np.average([nex[1], prev[1]], weights=[1, flag+1]))
                            miny = int(np.average([nex[2], prev[2]], weights=[1, flag+1]))
                            w =    int(np.average([nex[3], prev[3]], weights=[1, flag+1]))
                            h =    int(np.average([nex[4], prev[4]], weights=[1, flag+1]))
                            # Get a frame patch which has a location close to its neighboring patches                                                            
                            patch = cv2.resize(video_as_array[fr][miny:(miny+h), minx:(minx+w), :], (112,112), interpolation = cv2.INTER_CUBIC)
                            to_process[k] = patch
                            buffer_store[k] = [fr, minx, miny, w, h]
                        else:
                            minx = prev[1]
                            miny = prev[2]
                            w = prev[3]
                            h = prev[4]
                            patch = cv2.resize(video_as_array[fr][miny:(miny+h), minx:(minx+w), :], (112,112), interpolation = cv2.INTER_CUBIC)
                            to_process[k] = patch
                            buffer_store[k] = [fr, minx, miny, w, h]
                    
                    elif buffer_store[k][0] == -1 and k==0:
                        fr = buffer_store[k][1]
                        while buffer_store[k+flag][0] == -1 and k+flag<15:
                            flag += 1 
                        if buffer_store[k+flag][0] !=-1 and k+flag < 16:
                            nex = buffer_store[k+flag]
                            # Interpolate the start coordinates and dimensions
                            minx = int(nex[1])
                            miny = int(nex[2])
                            w = int(nex[3])
                            h = int(nex[4])
                            # Get a frame patch which has a location close to its neighboring patches
                            patch = cv2.resize(video_as_array[fr][miny:(miny+h), minx:(minx+w), :], (112,112), interpolation = cv2.INTER_CUBIC)
                            to_process[k] = patch
                            buffer_store[k] = [fr, minx, miny, w, h]
                        else:
                            to_process = []
                            buffer_store = []
                            break

                # Each input to activity recognition architecture will be a batch of 16 video frames
                count += 1
                file_name = os.path.join('process_data', act, act + '_' + video.split('.')[0] + '_' + str(count) + '.npy' )
                file_dir = os.path.join('process_data', act)
                try:
                    os.mkdir(file_dir)
                except:
                    pass
                print(count)
#                np.save(file_name, np.array(to_process, dtype=object))
                # to_cccd.append(to_process)
                to_process = []
                buffer_store = []







data_path = 'data_set/'

options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}

tfnet = TFNet(options)

(root, subdir, files) = os.walk('data_set')

to_cccd = []


# Root[1] contains the list of all the subdirectories in the data_set folder 
# Parse these folders to get the videos
act1 = root[1][0]
act2 = root[1][1]

class_path1 = data_path + act1
class_path2 = data_path + act2

c1vids = os.listdir(class_path1)
c1vids.sort()
c2vids = os.listdir(class_path2)
c2vids.sort()

vidnum2 = vidnum%len(c1vids)
if vidnum < len(c1vids):
    video = c1vids[vidnum]
    process_video(video, class_path1, act1)
    print(video, act1, 'act1')

elif vidnum2 < len(c2vids):
    video = c2vids[vidnum2]
    process_video(video, class_path2, act2)
    print(video, act2, 'act2')

else:
    print('No videos remaining')



