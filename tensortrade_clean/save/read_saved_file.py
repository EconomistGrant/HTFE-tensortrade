'''
meta:模型图结构
index：模型参数索引
data:模型参数值
'''

#%%查看模型参数值
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file 
printer = print_tensors_in_checkpoint_file 
printer('test', tensor_name = None, all_tensors = True, all_tensor_names = True)
#printer('test', tensor_name = 'agent/policy/policy-network/dense2/weights', all_tensors = False, all_tensor_names = False)

#%%通过meta文件生成log 供tensorboard 可视化
import tensorflow as tf
g = tf.Graph()
with g.as_default() as g:
    tf.train.import_meta_graph('test.meta')

with tf.Session(graph = g) as sess:
    file_writer = tf.summary.FileWriter(logdir = './', graph = g)