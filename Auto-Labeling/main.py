import pandas as pd
import auto_labeling
import modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import tensorflow as tf

PATH = CONFIDENTIAL
CLASS = 3
TEST_SIZE = 0.1
RND_STATE = 101

class ManipulateData: 
    def __init__(self):
        super().__init__()
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
    def import_data(self): 
        self.df = pd.read_excel(PATH, index_col='index')
        print(self.df)
        return self.df
    
    def data_spliting(self, X, y): 
        self.X = X
        self.y = y
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=TEST_SIZE, random_state=RND_STATE)
        
    @property
    def getX_train(self): 
        self.X_train = self.X_train / 255.0
        return self.X_train
    
    @property
    def gety_train(self): 
        self.y_train = tf.keras.utils.to_categorical(self.y_train, CLASS)
        return self.y_train
    
    @property
    def getX_val(self): 
        self.X_val = self.X_val / 255.0
        return self.X_val
    
    @property
    def gety_val(self): 
        self.y_val = tf.keras.utils.to_categorical(self.y_val, CLASS)
        return self.y_val


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
X_val = manipulate_data.getX_val
y_val = manipulate_data.gety_val

history = mod.model_train(model=model, 
                          X_train=np.asarray(X_train).astype('float32'), 
                          y_train=np.asarray(y_train).astype('float32'), 
                          X_val=np.asarray(X_val).astype('float32'), 
                          y_val=np.asarray(y_val).astype('float32'))

mod.model_acc_plot(history)
mod.model_loss_plot(history)

pred = []
pred = mod.prediction(X_val, y_val)
Y_pred = pred[0]
Y_true = pred[1]
print(classification_report(Y_true, Y_pred))

images_part.plot_results(X_val, Y_pred, Y_true)

# =============================================================================
# if __name__ == '__main__': 
#     main()
# =============================================================================
