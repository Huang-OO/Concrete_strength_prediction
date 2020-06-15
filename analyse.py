"""

分析数据中各个成分对混泥土强度的影响

"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = 'data/concrete.xls'


#读取文件
data = pd.read_excel(path)

# 修改列名
data.columns = ['cement_component', 'furnace_slag', 'flay_ash', 'water_component', 'superplasticizer', \
    'coarse_aggregate', 'fine_aggregate', 'age', 'concrete_strength']

data.to_excel(path,index=False)

# 查看数据基本面
data.info()

#各个影响混泥土硬度参数的因素的比较图
plot_count = 1
for feature in list(data.columns)[:-1]:
    plt.subplot(2,4, plot_count)
    plt.scatter(data[feature], data['concrete_strength'])
    plt.xlabel(feature.replace('_',' ').title())
    plt.ylabel('Concrete strength')
    plot_count +=1
plt.show()


#相关性矩阵图
plt.figure(figsize=(9,9))
corrmat = data.corr()
sns.heatmap(corrmat, vmax= 0.8, square = True)
plt.show()
