from PIL import Image,ImageDraw
import json
import cv2

annotation_file = open('taco_dataset/annotations.json','r')

annotation_json = json.loads(annotation_file.read())

annotation_images = annotation_json["images"]
annotation_annot = annotation_json["annotations"]
imgs = []
annotations = []


for i in range(len(annotation_images)):
    imgs.append(annotation_images[i]["file_name"])


for a in range(len(annotation_annot)):
    segments = annotation_annot[a]["segmentation"][0]
    segment_x = []
    segment_y = []
    for i in range(len(segments)):
        if i%2 == 0:
            segment_x.append(int(segments[i]))
        else:
            segment_y.append(int(segments[i]))

    x_1 = min(segment_x)
    y_1 = min(segment_y)
    x_2 = max(segment_x)
    y_2 = max(segment_y)


    box = []
    box.append(annotation_annot[a]["image_id"])
    box.append(x_1)
    box.append(y_1)
    box.append(x_2)
    box.append(y_2)
    annotations.append(box)





"""
for i in range(len(annotation_annot)):
    box = []
    box.append(annotation_annot[i]["image_id"])

    x_1 = int(annotation_annot[i]["bbox"][0])
    x_2 = int(annotation_annot[i]["bbox"][2])
    y_1 = int(annotation_annot[i]["bbox"][1])
    y_2 = int(annotation_annot[i]["bbox"][3])

    if x_1 < x_2:
        if y_1 < y_2:
            box.append(x_1)
            box.append(y_1)
            box.append(x_2)
            box.append(y_2)
        else:
            box.append(x_1)
            box.append(y_2)
            box.append(x_2)
            box.append(y_1)

    else:
        if y_1 < y_2:
            box.append(x_2)
            box.append(y_1)
            box.append(x_1)
            box.append(y_2)
        else:
            box.append(x_2)
            box.append(y_2)
            box.append(x_1)
            box.append(y_1)

    annotations.append(box)
"""

for c in range(len(annotations)):
    print(imgs[annotations[c][0]])
    img = Image.open("taco_dataset/"+imgs[annotations[c][0]])
    img2 = img.crop((annotations[c][1],annotations[c][2],annotations[c][3],annotations[c][4]))
    img2.save("taco_annotated/img_"+str(c)+".jpg")