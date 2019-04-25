# ## 作業
# ### 請根據不同的 HOUSETYPE_MODE 對 AMT_CREDIT 繪製 Histogram

# Import 需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 另一個繪圖-樣式套件

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')

# 設定 data_path
dir_data = './data1/'

f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
app_train.head()


unique_house_type = app_train['HOUSETYPE_MODE'].unique()

nrows = len(unique_house_type)
ncols = nrows // 2

plt.figure(figsize=(10,30))
for i in range(len(unique_house_type)):
    plt.subplot(nrows, ncols, i+1)
    
    app_train.loc[:, ['unique_house_type'[i], 'AMT_CREDIT']].hist()
    
    plt.title(str(unique_house_type[i]))
plt.show()    

