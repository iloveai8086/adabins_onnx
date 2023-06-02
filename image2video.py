import os
import cv2
import time

img_path = "video/out4/"

img = cv2.imread("video/out4/0001.png")
img_info = img.shape
print(img_info)
size = (img_info[1], img_info[0])
img_nums = len(os.listdir(img_path))
print(img_nums)

# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# fourcc = cv2.VideoWriter_fourcc('X', '2', '6', '4')

videowrite = cv2.VideoWriter(os.path.join(img_path, 'videos.mp4'), 0x00000021, 20, size)
img_name = os.listdir(img_path)
img_name.sort()

for i in img_name:
    print("processing ", i)
    filepath = os.path.join(img_path, i)
    print(filepath)
    image = cv2.imread(filepath)
    videowrite.write(image)
