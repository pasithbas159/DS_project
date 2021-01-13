import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


keras = tf.keras
INITIAL_LR = 1e-4
CLASS = 3

class ModelingPart(): 
    def __init__(self):
        super().__init__()
        self.model = None
        self.history = None
        
        
    def CNN_3(self): 
        self.model = keras.Sequential([
            keras.layers.Conv2D(16, (3,3), activation='relu', input_shape = (50,50,3)), 
            keras.layers.MaxPooling2D(2,2), 
            keras.layers.Conv2D(32, (3,3), activation='relu'), 
            keras.layers.MaxPooling2D(2,2), 
            keras.layers.Conv2D(64, (3,3), activation='relu'), 
            keras.layers.MaxPooling2D(2,2),
            keras.layers.Flatten(), 
            keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)), 
            # keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001),
            keras.layers.Dropout(.3),
            keras.layers.Dense(CLASS, activation='softmax')
            ])
        
        # print(self.model.summary())
        
        return self.model

    def model_train(self, model, X_train, y_train, X_val, y_val): 
        self.lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                        INITIAL_LR,
                        decay_steps=1000,
                        decay_rate=0.9,
                        staircase=False)
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr_schedule), loss='categorical_crossentropy', metrics=['acc'])
        checkpoint = keras.callbacks.ModelCheckpoint('model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', save_best_only=True, monitor='val_acc', mode='max')
        self.history = model.fit(X_train, y_train, epochs=250, validation_data=(X_val, y_val), callbacks=[checkpoint])
        
        model.evaluate(X_val, y_val)
        
        return self.history
    
    def model_acc_plot(self, hist): 
        acc = hist.history['acc']
        # loss = hist.history['loss']
        val_acc = hist.history['val_acc']
        
        epochs = range(len(acc))
        
        plt.plot(epochs, acc, 'b', label='Training accuracy')
        plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
        # plt.plot(epochs, loss, 'r', label='Training loss')
        plt.title('Training and validation accuracy')
        plt.legend(loc=0)
        plt.figure()
        

        plt.show()
        
    def model_loss_plot(self, hist): 
        loss = hist.history['loss']
        val_loss = hist.history['val_loss']
        
        epochs = range(len(loss))
        
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'g', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend(loc=0)
        plt.figure()
        

        plt.show()
        
    def prediction(self, X_val, y_val): 
        # after run
        # self.model = keras.models.load_model("Model/model-140-0.707832-0.689840.h5")
        pred = self.model.predict(X_val)
        self.Y_pred_classes = np.argmax(pred, axis=1) 
        self.Y_true = np.argmax(y_val, axis=1)
        
        print(self.Y_pred_classes)
        print(self.Y_true)
        
        return self.Y_pred_classes, self.Y_true
    
    def new_data_prediction(self, X_test): 
        # after run
        self.model = keras.models.load_model("Model/model-140-0.707832-0.689840.h5")
        pred = self.model.predict(X_test)
        self.Y_pred_classes = np.argmax(pred, axis=1) 
        
        return self.Y_pred_classes
    