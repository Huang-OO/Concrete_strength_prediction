""""

离差标准化，[0,1]标准化

"""

import pandas as pd

filename = 'data/concrete.xls'                   #原始数据文件
standardization = 'data/standardization.xls'    #标准化后数据文件

data = pd.read_excel(filename)                    #读取文件

data = (data - data.min())/(data.max()-data.min())#离差标准化
data = data.reset_index()

data.to_excel(standardization,index=False)        #存储文件