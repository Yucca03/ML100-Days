# ## 作業
# 1. 請用 numpy 建立一個 10 x 10, 數值分布自 -1.0 ~ 1.0 的矩陣並繪製 Heatmap
# 2. 請用 numpy 建立一個 1000 x 3, 數值分布為 -1.0 ~ 1.0 的矩陣，並繪製 PairPlot (上半部為 scatter, 對角線為 hist, 下半部為 density)
# 3. 請用 numpy 建立一個 1000 x 3, 數值分布為常態分佈的矩陣，並繪製 PairPlot (上半部為 scatter, 對角線為 hist, 下半部為 density)


# Import 需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 另一個繪圖-樣式套件
plt.style.use('ggplot')

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')



matrix = np.random.random_sample(size = (10, 10))
print(matrix)
plt.figure(figsize=(10,10))

heatmap = sns.heatmap(matrix, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)

plt.show()

nrow = 1000
ncol = 3

matrix = np.random.random_sample(size = (nrow, ncol))

indice = np.random.choice([0,1,2], size=nrow)
plot_data = pd.DataFrame(matrix, indice)


grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False)

grid.map_upper(plt.scatter, alpha = 0.2)
grid.map_diag(sns.kdeplot)
grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r)

plt.suptitle('1000*3 PairPlot', size = 32, y = 1.05)

plt.show()

nrow = 1000
ncol = 3
mu, sigma = 0, 0.1
matrix = np.random.normal(mu, sigma, size = (nrow, ncol))

indice = np.random.choice([0,1,2], size=nrow)
plot_data = pd.DataFrame(matrix, indice)


grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False)

grid.map_upper(plt.scatter, alpha = 0.2)
grid.map_diag(sns.kdeplot)
grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r)

plt.show()

