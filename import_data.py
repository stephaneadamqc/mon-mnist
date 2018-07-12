import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

data_directory = u'%s\MNIST' % (os.path.abspath(os.path.dirname(__file__)))

training_data = pd.read_csv(data_directory + '\\train.csv')
submission_data = pd.read_csv(data_directory + '\\test.csv')
images_tmp1 = training_data.values[:, 1:]
images_tmp2 = submission_data.values

print("Submission data structure:")
print(submission_data.head())
print(submission_data.shape)
print("Training data structure")
print(training_data.head())
print(training_data.shape)

dimension = 28
training_images = np.zeros((training_data.shape[0], dimension, dimension))
submission_images = np.zeros((submission_data.shape[0], dimension, dimension))
for dim in range(dimension):
    training_images[:, dim, :] += images_tmp1[:, dim * dimension: dim * dimension + dimension]  # Reconstruct the images
    submission_images[:, dim, :] += images_tmp2[:, dim * dimension: dim * dimension + dimension]

# plt.imshow(training_images[3, :, :], cmap=matplotlib.cm.binary, interpolation='nearest')
# plt.axis("off")
# plt.show()

training_images /= 255
submission_images /= 255

training_label = training_data['label']
print("training_image shape:")
print(training_images.shape)
print("Submission_images shape:")
print(submission_images.shape)
del images_tmp2
del images_tmp1
