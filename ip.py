from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import os
checkpoint_path = os.path.join('./saved_model', "model.ckpt")

# List ALL tensors example output: v0/Adam (DT_FLOAT) [3,3,1,80]
print_tensors_in_checkpoint_file(file_name='./saved_model/model.ckpt', tensor_name='detector/darknet-53/Conv_50/weights', all_tensors = True)
