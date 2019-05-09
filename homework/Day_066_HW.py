
# coding: utf-8



from keras.utils import multi_gpu_model
from keras.models import Model
from keras.layers import Input, Dense


a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(inputs=a, outputs=b)

config = model.get_config()
print(config)




model.summary()


# # 作業:
#     檢查 backend
#     檢查 fuzz factor
#     設定 Keras 浮點運算為float16



import keras
from keras import backend as K

#檢查 backend
K.backend()




#檢查 fuzz factor
K.epsilon()



#設定 Keras 浮點運算為float16
K.floatx('float16')

