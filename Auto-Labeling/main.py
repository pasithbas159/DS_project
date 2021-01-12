import pandas as pd
import auto_labeling
import modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import tensorflow as tf

PATH = CONFIDENTIAL

class ManipulateData: 
    def __init__(self):
        super().__init__()
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        # self.X_val = None
        self.X_test = None
        self.y_train = None
        # self.y_val = None
        self.y_test = None
        
    def import_data(self): 
        self.df = pd.read_excel(PATH, index_col='index')
        print(self.df)
        return self.df
    
    def data_spliting(self, X, y): 
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=101)
        
        
    
    @property
    def getX_train(self): 
        self.X_train = self.X_train / 255.0
        return self.X_train
    
    @property
    def gety_train(self): 
        self.y_train = tf.keras.utils.to_categorical(self.y_train, 3)
        return self.y_train
    
    @property
    def getX_test(self): 
        self.X_test = self.X_test / 255.0
        return self.X_test
    
    @property
    def gety_test(self): 
        self.y_test = tf.keras.utils.to_categorical(self.y_test, 3)
        return self.y_test


mod = modeling.ModelingPart()
images_part = auto_labeling.ImagesPart()
manipulate_data = ManipulateData()

data = manipulate_data.import_data()
images = images_part.insert_images(data)
img_pixel = images_part.read_images(images)

manipulate_data.data_spliting(X=img_pixel, y=np.expand_dims(data['stance'], axis=1))
images_part.show_example_images(imgs=img_pixel, data=data)
model = mod.CNN_3()
X_train = manipulate_data.getX_train
y_train = manipulate_data.gety_train
X_test = manipulate_data.getX_test
y_test = manipulate_data.gety_test

history = mod.model_train(model=model, 
                          X_train=np.asarray(X_train).astype('float32'), 
                          y_train=np.asarray(y_train).astype('float32'), 
                          X_test=np.asarray(X_test).astype('float32'), 
                          y_test=np.asarray(y_test).astype('float32'))

mod.model_acc_plot(history)
mod.model_loss_plot(history)

pred = []
pred = mod.prediction(X_test, y_test)
Y_pred = pred[0]
Y_true = pred[1]
print(classification_report(Y_true, Y_pred))

images_part.plot_results(X_test, Y_pred, Y_true)

# print(history)

# =============================================================================
# if __name__ == '__main__': 
#     main()
# =============================================================================
