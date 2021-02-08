import cv2
import numpy as np
import tensorflow as tf

Image_size = 100
CatDogModel = tf.keras.models.load_model('waste.model')
ImagePath = "D://PROGRAMOZAS/PROJEKT/Okos_kuka/yolo_test/.vscode/img/paper.jpg"
Image = cv2.imread(ImagePath, cv2.IMREAD_GRAYSCALE)
NewImage = cv2.resize(Image, (Image_size, Image_size))
NewImage = np.array(NewImage).reshape(-1, Image_size, Image_size, 1)

prediction = CatDogModel.predict([NewImage])
a = prediction[0]
print(a)
