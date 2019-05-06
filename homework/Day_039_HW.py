## 作業
請閱讀相關文獻，並回答下列問題

[脊回歸 (Ridge Regression)](https://blog.csdn.net/daunxx/article/details/51578787)
[Linear, Ridge, Lasso Regression 本質區別](https://www.zhihu.com/question/38121173)

1. LASSO 回歸可以被用來作為 Feature selection 的工具，請了解 LASSO 模型為什麼可用來作 Feature selection
2. 當自變數 (X) 存在高度共線性時，Ridge Regression 可以處理這樣的問題嗎?

#### 1. LASSO會把不重要的變數變成零，以此達到特徵篩選的功能
#### 2. 可以，透過log(λ)>0校正參數來限制係數，以降低模型變異和誤差
