

input_clip = batchx16x112x112x3
label = 0/1

model = c3d 

output = model.inference(input_clip)

sess.run(output)


