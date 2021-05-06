from PIL import Image
import os


cat_counter = 0
dog_counter = 0
directory = 'train/'
cropped = 'TRAINbounding-box/'
try:
    os.makedirs('TRAINbounding-box')
except Exception as e:
    print('Folder for bounding boxes already created.')

for filename in os.listdir(directory):
    if filename.endswith('.jpg'):
        img = Image.open('train/' + filename)
        name = filename.split('.')[0]
        f = open('train/' + name + '.txt')
        bounding_box = f.readline().split(' ')
        area = (int(bounding_box[1]),
                int(bounding_box[2]),
                int(bounding_box[3]),
                int(bounding_box[4]))
        cropped_img = img.crop(area)
        label = ''
        if int(bounding_box[0]) == 1:
            label = 'cat.'
            cropped_img.save(cropped + label + str(cat_counter) + '.jpg')
            cat_counter += 1
        else:
            label = 'dog.'
            cropped_img.save(cropped + label + str(dog_counter) + '.jpg')
            dog_counter += 1

