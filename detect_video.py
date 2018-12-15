import sys



##############  IMPORTANT !!!!!! ###############
# Enter your darkflow install directory below  #
################################################

darkflow_dir = '../darkflow'

#darkflow_dir = '/path/to/darkflow'

assert(darkflow_dir!='/path/to/darkflow')

sys.path.insert(0,darkflow_dir)

from darkflow.net.build import TFNet
import cv2 
import os
import numpy as np
import argparse
import tensorflow as tf
import argparse
from c3d.checkpoint_utils import load_checkpoint, initialize_from_dict
from c3d.c3d_model import C3D 
import time


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




def expand_bbox(shape, minx, miny, w, h, expansion_rate=1.5):
    '''
    Expand the region of the bounding box to ensure that the entirety of the dog is present in the clips passed to C3D
    '''
    h2 = min(h*expansion_rate, shape[0])
    w2 = min(w*expansion_rate, shape[1]) 

    minx2 = max(minx - w/2, 0)
    miny2 = max(miny - h/2, 0)

    return int(minx2), int(miny2), int(w2), int(h2)






def is_dog_on_couch(dxmin,dxmax,dymin,dymax,cxmin,cxmax,cymin,cymax):
    '''
    Compare the location of the dog and furniture bounding boxes to detect when the dog is on furniture.
    '''
    dwidth=dxmax-dxmin
    if(dxmin>=cxmin-0.4*dwidth and dxmax<=cxmax+0.4*dwidth):
        if(dymax>=cymin and dymax<=0.9*cymax):
            return 1
        else:
            return 0
    else:
        return 0


font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 2
fontColor              = (0,0,255)
fontColor2              = (0,75,255)
lineType               = 5


def process_video(video_path):

    if os.path.isfile(video_path):
        video_result =[]
        video_as_array = []
        print(video_path)
        current = cv2.VideoCapture(video_path)
        framecount = int(current.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_ps=int(current.get(cv2.CAP_PROP_FPS))
        width = int(current.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
        height = int(current.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 'x264' doesn't work
        writer = cv2.VideoWriter(os.path.join('bbox_outputs',video_path.split('/')[-1]),fourcc, frames_ps, (width,height))
        batches = int(framecount/16)
        to_process = []
        buffer_store = []
        print(framecount)
        print(width)
        print(height)
        buffer_frames = []
        prev_minx = -1
        prev_dog_on_couch=0
        prev_diff2 = 100000
        for i in range(framecount):
            ft = time.time()
            _, frame = current.read()
            # Appends the frames
            video_as_array.append(frame)
            yt = time.time()
            result = tfnet.return_predict(frame)
            print("yolo: ", time.time()-yt)
            # Appends the bounding box results
            video_result.append(result)
            dog_found=0
            couches_found=0
            cminx=[]
            cminy=[]
            cmaxx=[]
            cmaxy=[]
            max_patch = 0
            max_buff = [-1,i]
            max_conf = 0
            min_diff = 100000
            for cl in result:
                if (cl['label'] == 'dog'):
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
                    dog_found=1
                    if len(buffer_store) > 0:
                        if abs(buffer_store[-1][1]-minx) < min_diff:
                            if prev_diff2*1.5 > abs(min_diff - abs(buffer_store[-1][1]-minx)):
                                max_patch = patch
                                max_buff = buff
                                best_minx=minx
                                best_miny=miny
                                best_maxx=maxx
                                best_maxy=maxy
                                prev_diff = abs(min_diff - abs(buffer_store[-1][1]-minx))
                                min_diff = abs(buffer_store[-1][1]-minx)
                    else:
                        max_patch = patch
                        max_buff = buff
                        best_minx=minx
                        best_miny=miny
                        best_maxx=maxx
                        best_maxy=maxy

            
                if(cl['label']=='sofa' or cl['label']=='chair' or cl['label']=='diningtable' or cl['label']=='couch'):
                    cminx.append(cl['topleft']['x'])
                    cminy.append(cl['topleft']['y'])
                    cmaxx.append(cl['bottomright']['x'])
                    cmaxy.append(cl['bottomright']['y'])
                    couches_found+=1
                
            to_process.append(max_patch)
                    # Also store the video frame where dog label doesn't exists
            buffer_store.append(max_buff)

            

            if(dog_found==0):
                if prev_minx!=-1:
                    if prev_dog_on_couch==1:
                        cv2.rectangle(frame, (prev_minx, prev_miny), (prev_maxx, prev_maxy), (0,0,255), 2)
                        cv2.putText(frame,'Dog on furniture!',bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
                    else:
                        cv2.rectangle(frame, (prev_minx, prev_miny), (prev_maxx, prev_maxy), (0,255,0), 2)
            
            
            print(i)
            flaggg=0
            if(dog_found==1 and couches_found>0):
                cv2.rectangle(frame, (best_minx, best_miny), (best_maxx, best_maxy), (0,255,0), 2)
                prev_minx=best_minx
                prev_miny=best_miny
                prev_maxx=best_maxx
                prev_maxy=best_maxy
                for q in range(couches_found):
                    w=0
                    res=is_dog_on_couch(minx,maxx,miny,maxy,cminx[q],cmaxx[q],cminy[q],cmaxy[q])
                    if(res==1):
                        flaggg=1
                        w+=1
                        if(w==1):
                            cv2.rectangle(frame, (minx, miny), (maxx, maxy), (0,0,255), 2)
                            prev_dog_on_couch=1
                            bottomLeftCornerOfText = (int(np.floor(width/2))-100,100)
                            cv2.putText(frame,'Dog on furniture!',bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
            
                if(flaggg==1):
                    print('dog on couch!')
                else:
                    print('good doggy!')
                    prev_dog_on_couch=0
            
            
            buffer_frames.append(frame)

            
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

                output_pred = 0
                if to_process!=[]:
                    clip = np.array(to_process)
                    ct = time.time()
                    output_pred = sess.run([logits], feed_dict={input_data_tensor: [clip]})[0].argmax()
                    print('c3d time: ', time.time()-ct)

                if output_pred:
                    for bf in buffer_frames:
                        cv2.putText(bf,'Biting!',(30,30),font,1,fontColor2,lineType)
                        writer.write(bf)

                else:
                    for bf in buffer_frames:
                        writer.write(bf)
                buffer_frames = []
                
                    
            
                # Each input to activity recognition architecture will be a batch of 16 video frames
                count += 1
                to_process = []
                buffer_store = []
            print("frame time: ", time.time()-ft)
        for bf in buffer_frames:
            if output_pred ==1:
                cv2.putText(bf,'Biting!',(30,30),font,1,fontColor2,lineType)
            writer.write(bf)
        buffer_frames = []

        writer.release()


parser = argparse.ArgumentParser()
parser.add_argument('--vidpath', action='store', type=str, required=True, help='/path/to/video.avi')

args = parser.parse_args()


options = {"model": os.path.join(darkflow_dir, "cfg/yolo.cfg"), "load": os.path.join(darkflow_dir,"bin/yolo.weights"), "threshold": 0.1}

tfnet = TFNet(options)

vidpath = args.vidpath

process_video(vidpath)

