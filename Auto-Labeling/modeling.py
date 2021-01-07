import tensorflow as tf


keras = tf.keras

class ModelingPart(): 
    def __init__(self):
        super().__init__()
        self.model = None
        
        
    def CNN_3(self): 
        self.model = keras.Sequential([
            keras.layers.Conv2D(32, (3,3), activation='relu', input_shape = (200,200,3)), 
            keras.layers.MaxPooling2D(2,2), 
            keras.layers.Conv2D(64, (3,3), activation='relu'), 
            keras.layers.MaxPooling2D(2,2), 
            keras.layers.Conv2D(128, (3,3), activation='relu'), 
            keras.layers.MaxPooling2D(2,2),
            keras.layers.Flatten(), 
            keras.layers.Dense(256, activation='relu'), 
            keras.layers.Dense(2, activation='sigmoid')
            ])
        
        # print(self.model.summary())
        
        return self.model
    
    def model_train(self, model, X_train, y_train): 
        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc'])
        history = model.fit(X_train, y_train, epochs=10)
        
        return history
    
    # def model_train(self, model, X_train, y_train, X_test, y_test): 
    
