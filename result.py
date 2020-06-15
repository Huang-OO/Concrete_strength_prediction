import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#设置中文显示
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


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

x_data = data[:,[0,3,4,7]]
y_data = data[:,-1:]

x_data = x_data.astype(np.float32)



W1 = tf.constant([[ 0.8981481, -0.09867217],
 [-0.5568871, 0.02610966],
 [ 1.0999992, 0.14784722],
 [ 0.26011723, -2.314894  ]])
b1 = tf.constant([[-0.12030562, 0.2547304 ]])
W2 = tf.constant([[ 0.52902246],
 [-1.5807493 ]])
b2 = tf.constant([[0.47748905]])


m = tf.nn.relu(tf.add(tf.matmul(x_data,W1),b1))
n = tf.nn.tanh(tf.add(tf.matmul(m,W2),b2))

sess = tf.Session()
n = sess.run(n)
plt.plot(y_data,label='原始值')
plt.plot(n,label='预测值',color='r')
plt.legend(loc='best')
plt.show()

# plt.plot(y_data,label='原始值',color='r',linestyle=':')
# plt.plot(n,label='预测值',color='g')