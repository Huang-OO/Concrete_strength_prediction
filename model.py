"""

神经网络搭建，有四个输入一个输出，神经网络的基本结构输入层，隐藏层，输出层。

"""


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#定义神经网络层函数
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    #定义每层的名字
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        #定义权重以及权重名字
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size], stddev=1, seed=1234), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        #定义偏置以及名字
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        #正向传播
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        #激励函数
        Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs', outputs)
    #返回输出结果，权重，偏置
    return outputs,Weights,biases

keep_prob = tf.placeholder(tf.float32)

#传输离差标准化后的数据
standardized = pd.read_excel('data/standardization.xls')
data = []
for j in range(0,len(standardized.age.values)):
    data.append([standardized.cement_component.values[j],
                 standardized.furnace_slag.values[j],
                 standardized.flay_ash.values[j],
                 standardized.water_component.values[j],
                 standardized.superplasticizer.values[j],
                 standardized.coarse_aggregate.values[j],
                 standardized.fine_aggregate.values[j],
                 standardized.age.values[j],
                 standardized.concrete_strength.values[j]])#训练集的输入值X
data = np.array(data)

#输入值
x_data_train = data[0:1000,[0,3,4,7]]
x_data_test = data[1000:,[0,3,4,7]]
#输出值即标签
y_data_train = data[0:1000,-1:]
y_data_test = data[1000:,-1:]



#定义输入输出占位
with tf.name_scope('inputs'):
    X = tf.placeholder(tf.float32, [None, 4], name='x_input')
    Y = tf.placeholder(tf.float32, [None, 1], name='y_input')


#建立隐藏层
l1,W1,b1 = add_layer(X, 4, 2, n_layer=1, activation_function=tf.nn.relu)
#输出层
prediction,W2,b2 = add_layer(l1, 2, 1, n_layer=2, activation_function=tf.nn.tanh)



#定义损失函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(Y - prediction),
                                        reduction_indices=[1]))
    tf.summary.scalar('loss', loss)


#传输训练运用梯度下降
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)



#设置中文显示
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

#建立画布
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
#混泥土硬度准确值
ax.plot(y_data_train,label='原始值')
#画布循环
plt.ion()
plt.show()


with tf.Session() as sess:

    merged = tf.summary.merge_all()
    mergeds = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES,'loss')])
    #存储可视化文件
    train_writer = tf.summary.FileWriter("logs_1/train", sess.graph)
    test_writer = tf.summary.FileWriter("logs_1/test", sess.graph)
    #全局变量初始化
    init = tf.global_variables_initializer()
    sess.run(init)
    #训练模型
    for i in range(200001):
        #放入数据训练
        sess.run(train_step, feed_dict={X: x_data_train, Y: y_data_train,keep_prob:0.98})
        #每训练1000次输出一次训练结果
        if i % 1000 == 0:
            result = sess.run(merged,
                              feed_dict={X: x_data_train, Y: y_data_train,keep_prob:1})
            train_writer.add_summary(result, i)
            loss = sess.run(mergeds,
                            feed_dict={X: x_data_test, Y: y_data_test,keep_prob:1})
            test_writer.add_summary(loss, i)
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={X: x_data_train,keep_prob:1})
            lines = ax.plot(prediction_value, label='预测值', color='r')
            plt.legend(loc='best')
            #每次输出间隔时间
            plt.pause(1)
    #输出训练的权重偏置
    print(sess.run(W1),sess.run(b1),sess.run(W2),sess.run(b2))


