import pandas as pd
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np

R = 5
C = 5

class ImagesPart(): 
    
    def __init__(self):
        super().__init__()
        
        self.list_image = []
        self.images = []
        self.rnd_index = []

    def insert_images (self, dataframe: pd.DataFrame()): 
        for i in range(dataframe.shape[0]): 
            self.list_image.append(SOURCES(CONFIDENTIAL) + str(int(dataframe.index[i])) + '.jpg')
        return self.list_image
            
    
    def read_images (self, lst_img: list()): 
        for i in range(len(lst_img)): 
            img = cv2.imread(lst_img[i])                     
            img = cv2.resize(img,(200,200))
            self.images.append(img)
        img_converted = np.array(self.images)
        row =  img_converted.ravel()
        row_as_list = row.tolist()
        print(row_as_list)
        return row_as_list
    
    
    def show_example_images (self, imgs): 
        # Randomizing step
        for i in range(0, 25):     
            self.rnd_index.append(random.randint(0, 1000))
            
        print(self.rnd_index)
            
        fig, axes = plt.subplots(R, C, figsize=(12,12))
        axes = axes.ravel()
        
        for i in np.arange(0, R*C):
            axes[i].imshow(imgs[self.rnd_index[i]])
            plt.subplots_adjust(wspace=1)
    

# main_images = ImagesPart()

# list_image = main_images.insert_image()
# images = main_images.read_images(list_image)
# main_images.show_example_images(images)
    
