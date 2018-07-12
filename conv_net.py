from import_data import training_images, training_label
from keras.layers import Conv2D, MaxPool2D, Activation, Dense, Input, Flatten
from keras.models import Model, Sequential
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import os

dir = os.path.abspath(os.path.dirname(__file__))

training_label = OneHotEncoder().fit_transform(training_label.values.reshape(-1, 1))
print(training_label.shape)
training_images  = training_images.reshape((*training_images.shape, 1))
print(training_images.shape)

early_stop = EarlyStopping(patience=10)

model = Sequential()
model.add(Conv2D(filters=1, kernel_size=2, activation='relu', input_shape=training_images.shape[1:]))
model.add(Conv2D(filters=1, kernel_size=2, activation='relu', input_shape=training_images.shape[1:]))
model.add(MaxPool2D())
model.add(Conv2D(filters=1, kernel_size=2, activation='relu', input_shape=training_images.shape[1:]))
model.add(Conv2D(filters=1, kernel_size=2, activation='relu', input_shape=training_images.shape[1:]))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(20, activation='tanh'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x=training_images, y=training_label, epochs=100, batch_size=100, validation_split=.2, verbose=1,
                    callbacks=[early_stop])
model.save(dir + '\\convnet1.h5')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(dir + '\\accuracy.png')
# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(dir + '\\loss.png')
