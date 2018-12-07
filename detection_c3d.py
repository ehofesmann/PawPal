from darkflow.net.build import TFNet
import cv2 
import os
import numpy as np
import argparse
import tensorflow as tf
import argparse
from c3d.checkpoint_utils import load_checkpoint, initialize_from_dict
from c3d.c3d_model import C3D 


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

    h2 = min(h*expansion_rate, shape[0])
    w2 = min(w*expansion_rate, shape[1]) 

    minx2 = max(minx - w/2, 0)
    miny2 = max(miny - h/2, 0)

    return int(minx2), int(miny2), int(w2), int(h2)







#data_path = 'data_set/'

#options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}

#tfnet = TFNet(options)

#(root, subdir, files) = os.walk('data_set')
#print(os.walk('data_set'))
#print(root)
#to_cccd = []

def is_dog_on_couch(dxmin,dxmax,dymin,dymax,cxmin,cxmax,cymin,cymax):
    dwidth=dxmax-dxmin
    #    cwidth=cxmax-cxmin
    if(dxmin>=cxmin-0.4*dwidth and dxmax<=cxmax+0.4*dwidth):
        if(dymax>=cymin and dymax<=0.9*cymax):
            return 1
        else:
            return 0
    else:
        return 0

# Root[1] contains the list of all the subdirectories in the data_set folder
# Parse these folders to get the videos

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 2
fontColor              = (0,0,255)
fontColor2              = (0,75,255)
lineType               = 5


#or act in root[1]:
def process_video(video, class_path, act, vidnum):
 #   class_path = data_path + act

#    for video in os.listdir(class_path):
    count = 0
    video_path = class_path + '/' + video
    if os.path.isfile(video_path):
        video_result =[]
        video_as_array = []
        # Capture each Video
        current = cv2.VideoCapture(video_path)
        framecount = int(current.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_ps=int(current.get(cv2.CAP_PROP_FPS))
        width = int(current.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
        height = int(current.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 'x264' doesn't work
        writer = cv2.VideoWriter(os.path.join('bbox_outputs','trial'+str(vidnum)+'.avi'),fourcc, frames_ps, (width,height))
      #   writer = cv2.VideoWriter('trial1.mp4',-1,1,(width,height))
      #   print(framecount)
        batches = int(framecount/16)
        to_process = []
        # To interpolate for missing dog frames
        buffer_store = []
        print(framecount)
        print(width)
        print(height)
        buffer_frames = []
        prev_minx = -1
        prev_dog_on_couch=0
        for i in range(framecount):
            _, frame = current.read()
            # print(frame.shape)
            # Appends the frames
            video_as_array.append(frame)
            result = tfnet.return_predict(frame)
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
                    if max_conf < cl['confidence']:
                        max_conf = cl['confidence']
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
             #        cv2.rectangle(frame, (cl['topleft']['x'], cl['topleft']['y']), (cl['bottomright']['x'], cl['bottomright']['y']), (255,0,0), 2)
                    couches_found+=1
                
            to_process.append(max_patch)
                    # Also store the video frame where dog label doesn't exists
            buffer_store.append(max_buff)

            

            if(dog_found==0):
         #        writer.write(frame)
                if prev_minx!=-1:
               #     buffer_frames.append(frame)
               # else:
                    if prev_dog_on_couch==1:
                        cv2.rectangle(frame, (prev_minx, prev_miny), (prev_maxx, prev_maxy), (0,0,255), 2)
                        cv2.putText(frame,'Dog on furniture!',bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
                        #writer.write(frame)
                        #buffer_frames.append(frame)
                    else:
                        cv2.rectangle(frame, (prev_minx, prev_miny), (prev_maxx, prev_maxy), (0,255,0), 2)
                        #writer.write(frame)
                        #buffer_frames.append(frame)
            
            
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
           #                  cv2.rectangle(frame, (cminx[q], cminy[q]), (cmaxx[q], cmaxy[q]), (0,0,255), 2)
                            cv2.rectangle(frame, (minx, miny), (maxx, maxy), (0,0,255), 2)
            #                 prev_minx=minx
             #                prev_miny=miny
              #               prev_maxx=maxx
               #              prev_maxy=maxy
                            prev_dog_on_couch=1
                            bottomLeftCornerOfText = (int(np.floor(width/2))-100,100)
                            cv2.putText(frame,'Dog on furniture!',bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
                ##writer.write(frame)
            
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
                    output_pred = sess.run([logits], feed_dict={input_data_tensor: [clip]})[0].argmax()

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
             #    file_name = 'process_data/' + act + '_' + video.split('.')[0] + '_' + str(count) + '.npy'
             #    print(count)
             #    np.save(file_name, np.array(to_process, dtype=object))
             #    # to_cccd.append(to_process)
                to_process = []
                buffer_store = []
        for bf in buffer_frames:
            if output_pred ==1:
                cv2.putText(bf,'Biting!',(30,30),font,1,fontColor2,lineType)
            writer.write(bf)
        buffer_frames = []

        writer.release()


#os.environ['CUDA_VISIBLE_DEVICES'] = ''
parser = argparse.ArgumentParser()
parser.add_argument('--vidnum', action='store', type=int, required=True)
parser.add_argument('--vidname', action='store', type=str, default='none')
parser.add_argument('--classname', action='store', type=str, default='none')

args = parser.parse_args()
vidnum = args.vidnum

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

vidname = args.vidname
classname = args.classname

if vidname =='none':
    c1vids = os.listdir(class_path1)
    c1vids.sort()
    c2vids = os.listdir(class_path2)
    c2vids.sort()
    
    vidnum2 = vidnum%len(c1vids)
    if vidnum < len(c1vids):
        video = c1vids[vidnum]
        process_video(video, class_path1, act1, vidnum)
        print(video, act1, 'act1')
    
    elif vidnum2 < len(c2vids):
        video = c2vids[vidnum2]
        process_video(video, class_path2, act2, vidnum)
        print(video, act2, 'act2')
    
    else:
        print('No videos remaining')

else:
    if classname == 'biting':
        process_video(vidname, class_path2, act2, int(vidname.split('_')[0]))
    else:
        process_video(vidname, class_path, act, int(vidname.split('_')[0]))
