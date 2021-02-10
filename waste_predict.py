import cv2
import numpy as np
import tensorflow as tf

Image_size = 100
waste_model = tf.keras.models.load_model('waste.model')
out = []
files = ["can","can_2","opener","pet","pet_2","kupak","paper","galacsin"]

for i in range(len(files)):
    ImagePath = "img/"+files[i]+".jpg"
    Image = cv2.imread(ImagePath, cv2.IMREAD_GRAYSCALE)
    NewImage = cv2.resize(Image, (Image_size, Image_size))
    NewImage = np.array(NewImage).reshape(-1, Image_size, Image_size, 1)

    prediction = waste_model.predict([NewImage])
    a = list(prediction[0])
    out.append(a)

print(out)