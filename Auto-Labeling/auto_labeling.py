import pandas as pd
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np

R = 5
C = 5
IMG_WIDTH = 50
IMG_HEIGHT = 50
PATH = CONFIDENTIAL

class ImagesPart(): 
    
    def __init__(self):
        super().__init__()
        
        self.list_image = []
        self.images = []
        self.rnd_index = []
        self.rnd_test_index = []
        self.img_pixel_df = None

    def insert_images (self, dataframe: pd.DataFrame()): 
        for i in range(dataframe.shape[0]): 
            self.list_image.append(PATH + str(int(dataframe.index[i])) + '.jpg')
        return self.list_image
            
    
    def read_images (self, lst_img: list()): 
        for i in range(len(lst_img)): 
            img = cv2.imread(lst_img[i])                     
            img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
            # img = np.expand_dims(img, axis=0)
            self.images.append(img)
            # self.images = np.append(img)
        
        self.images = np.asarray(self.images)
        # self.img_pixel_df = pd.DataFrame(list(map(np.ravel, self.images)))
        # return self.img_pixel_df
        return self.images
    
    
    def show_example_images (self, imgs, data): 
        # Randomizing step
        for i in range(0, 25):     
            self.rnd_index.append(random.choice(range(0, data.shape[0])))
            
        print(self.rnd_index)
            
        fig, axes = plt.subplots(R, C, figsize=(12,12))
        axes = axes.ravel()
        
        for i in np.arange(0, R*C):
            axes[i].imshow(imgs[self.rnd_index[i]])
            plt.subplots_adjust(wspace=1)
            
        plt.show()
            
    def plot_results(self, X_test, Y_pred_classes, Y_true): 
        for i in range(0, 25):     
            self.rnd_test_index.append(random.choice(range(0, X_test.shape[0])))
        print(self.rnd_test_index)
            
        fig, axes = plt.subplots(R, C, figsize=(12,12))
        axes = axes.ravel()
        
        for i in np.arange(0, R*C):
            axes[i].imshow(X_test[self.rnd_test_index[i]])
            axes[i].set_title("True: %s \nPredict: %s" % (Y_true[self.rnd_test_index[i]], Y_pred_classes[self.rnd_test_index[i]]))
            axes[i].axis('off')
            plt.subplots_adjust(wspace=1)
            
        self.rnd_test_index.clear()
    

# main_images = ImagesPart()

# list_image = main_images.insert_image()
# images = main_images.read_images(list_image)
# main_images.show_example_images(images)
    
