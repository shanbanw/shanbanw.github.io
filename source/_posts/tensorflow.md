---
title: Tensorflow基础
date: 2017-01-11 20:51:03
tags: deep learning
---
Tensorflow是一个通过数据流图(data flow graph)进行数值运算的库。结点表示运算，边表示数据。所有的数据都是通过tensor来表示的。Tensor是向量和矩阵的一种推广，在tensorflow里面表示多维数组。每个tensor都有固定的数据类型（data type）和动态的维度，维度用3个数字表示，分别为rank, shape, 和dimension number。<font color="blue">Tensor rank</font>表示维度的个数，rank 0表示一个标量(Scalar)，rank 1等价于向量(Vector)，rank 2等价于矩阵(Matrix)，等等；对于tensor T，可以用T[i, j, k]这种方式访问里面的数值。<font color="blue">Tensor shape</font>表示维度的大小，比如2x2的矩阵，5x5x7的3维数组。<font color="blue">Tensor data type</font>包括tf.float32, tf.float64, tf.int8, tf.int16, tf.int32, tf.int64, tf.uint8(8 bits unsigned integer), tf.string(Variable length byte arrays. Each element of a tensor is a byte array.), tf.bool.<!-- more -->
<font color="blue">Feed</font>机制可以在图上任意位置临时替换tensor的值。在run()里面使用feed_dict。
多数情况下，一个图被执行很多次，大多数tensor在一次执行后就会消失。<font color="blue">Tensor Variable</font>返回一个在图执行过程中永久存在，并且可变的tensor，通常用来表示模型中的参数。使用之前需要初始化，用tf.global_variables_initializer()初始化全部Variable，用tf.initialize_variables([])初始化某些变量。用Variable.assign()操作改变变量的值。
<pre class="brush: py;">
import tensorflow as tf
import numpy as np
T = tf.constant([[[1,2],[2,3]],[[3,4],[5,6]]])
print(T)
#Tensor("Const_1:0", shape=(2, 2, 2), dtype=int32)
with tf.Session() as sess: #结束后session自动关闭
    print(sess.run(T[1,1,0]))
#5

#交互式
sess=tf.InteractiveSession()
print(T[1,1,0].eval())
sess.close()

data = np.array([[1,2,3],[4,5,6]])
#convert_to_tensor可以把tensor对象，numpy数组，列表，标量转换为tensor对象
T_data = tf.convert_to_tensor(data, dtype=tf.float32)

# feed, 临时替换某个tensor的值
with tf.Session() as sess:
    input1 = tf.placefolder(tf.int32) #需要传递dtype参数
    input2 = tf.constant(1)
    output = tf.add(input1, input2)
    print(sess.run(output,feed_dict={input1:1})) # 2
    print(sess.run(output,feed_dict={input1:1,input2:10})) # 11

# Variable，接收一个tensor用来初始化
state = tf.Variable(0, name="counter")
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

b = tf.Variable(tf.zeros([1000]), name="bias")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b))

# 保存数据流图
tf.Graph.as_graph_def(from_version=None, add_shapes=False)
</pre>

Tensor运算
<pre class="brush: py;">
# matrix operations
tf.transpose() #转置
tf.matmul() #矩阵乘法
tf.matrix_determinant() #行列式
tf.matrix_inverse() #逆
tf.matrix_solve() #求解

# Reduction
# 在tensor的某个维度上进行运算，输出维度少1的tensor
tf.reduce_prod(x, reduction_indices) # reduce prod
tf.reduce_min()
tf.reduce_max()
tf.reduce_mean()
tf.reduce_all()
tf.reduce_any()

# Segmentation
# 在某个维度上按照一个index array进行运算
sess = tf.InteractiveSession()
seg_ids = tf.constant([0,1,1,2,2]); # Group indexes : 0|1,2|3,4
tens1 = tf.constant([[2, 5, 3, -5],  
                    [0, 3,-2,  5], 
                    [4, 3, 5,  3], 
                    [6, 1, 4,  0],
                    [6, 1, 4,  0]])
tf.segment_sum(tens1, seg_ids).eval()
# array([[ 2, 5, 3, -5],
#        [ 4, 6, 3, 8],
#        [12, 2, 8, 0]], dtype=int32)

# Sequence
tf.argmin()  #输出某个维度最小值的索引
tf.argmax()
tf.listdiff()
tf.where()
tf.unique()

# shape transformations
tf.shape(data)
tf.reshape()

# slicing and joining
tf.slice(input_, begin, size, name=None) # cutting an slice
# Splits a tensor into `num_split` tensors along one dimension
tf.split(split_dim, num_split, value, name='split')
# tf.tile([1,2],[3]) array([1, 2, 1, 2, 1, 2], dtype=int32)
tf.tile(input, multiples, name=None)
# padding 填充
tf.pad(tensor, paddings, mode='CONSTANT', name=None)
tf.concat() #concatenating
tf.pack()
tf.reverse()
</pre>

读取文件
<pre class="brush: py;">
# CSV
tf.TextLineReader(ReaderBase)
# Reading image data
tf.WholeFileReader(ReaderBase)
</pre>

Graph: tensorflow默认生成一个Graph对象，把操作都加进去。
<pre class="brush: py; highlight=[2,3];">
# 创建一个新图,不影响默认的图对象
new_g = tf.Graph()
with new_g.as_default():
    # 这里面的操作都会包含进图new_g中
    a = tf.mul(2,3)
    ...
# 外面的操作放入default graph
in_default_graph = tf.add(1,2)
# 获取default graph
default_graph = tf.get_default_graph()

# 用于创建多个相互无关的图
g1 = tf.Graph()
g2 = tf.Graph()
with g1.as_default():
    ...
with g2.as_default():
    ...
</pre>

name scopes: 用区块组织图，利于tensorboard可视化。
<pre class="brush: py;">
import tensorflow as tf
#block 1
with tf.name_scope("Scope_A"):
    a = tf.add(1,2,name="A_add")
    b = tf.mul(a,3,name="A_mul")
#block 2
with tf.name_scope("Scope_B"):
    c = tf.add(4,5,name="B_add")
    d = tf.mul(c,6,name="B_mul")
e = tf.add(b,d,name="output")
# 打开SummaryWriter
writer = tf.summary.FileWriter('./name_scope_1',graph=tf.get_default_graph())
writer.close()
</pre>

打开tensorboard
tensorboard --logdir='./name_scope_1' --port=8888
浏览器输入localhost:8888，得到
![name_scope_1](/img/name_scope_1.png)

Tensorboard: 交互式查看数据流图和程序运行结果
<pre class="brush: py;">
tf.summary.FileWriter(dir, graph)
</pre>

tensorboard --logdir=dir --port=6006
默认端口为6006