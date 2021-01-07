import pandas as pd
import auto_labeling
import modeling
from sklearn.model_selection import train_test_split
import tensorflow as tf

PATH = 'CONFIDENTIAL'

class ManipulateData: 
    def __init__(self):
        super().__init__()
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def import_data(self): 
        self.df = pd.read_excel(PATH, index_col='index')
        print(self.df)
        return self.df
    
    def data_spliting(self, X, y): 
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=101)
    
    @property
    def getX_train(self): 
        self.X_train = tf.convert_to_tensor(self.X_train)
        # print(self.X_train[0])
        return self.X_train
    
    @property
    def gety_train(self): 
        self.y_train = tf.convert_to_tensor(self.y_train)
        # print(self.y_train[0])
        return self.y_train


mod = modeling.ModelingPart()
images_part = auto_labeling.ImagesPart()
manipulate_data = ManipulateData()

data = manipulate_data.import_data()
images = images_part.insert_images(data)
img_pixel = images_part.read_images(images)

manipulate_data.data_spliting(X=img_pixel, y=data['stance'])
images_part.show_example_images(images)
model = mod.CNN_3()
X_train = manipulate_data.getX_train
y_train = manipulate_data.gety_train
history = mod.model_train(model, X_train, y_train)
print(history)

# =============================================================================
# if __name__ == '__main__': 
#     main()
# =============================================================================
