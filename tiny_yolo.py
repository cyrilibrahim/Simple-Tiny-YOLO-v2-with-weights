
import torch
import torch.nn as nn
from torch.autograd import Variable
import PIL.Image as Image
from torchvision import transforms
import yolo_transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# 5 anchors given by the cfg file
anchors = np.array([1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52])* 1
classes = np.array(['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
           'sofa', 'train', 'tvmonitor'])

treshold = 0.4
input_size = 416

### IMPORTANT: Tiny YOLO v2 in original cfg doesn't have passthrough layer !

class TinyYOLO(nn.Module):

    def __init__(self):
        super(TinyYOLO, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ReflectionPad2d([0, 1, 0, 1]),
            nn.MaxPool2d(2, 1)
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.layer9 = nn.Sequential(
            nn.Conv2d(1024, 125, kernel_size=1, padding=0, stride=1, bias=False)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        return out

yolo = TinyYOLO()
yolo.load_state_dict(torch.load("tiny-yolo.pth"))
yolo.eval()

img = "data/Capture.jpg"

image = Image.open(img)

transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor()])
reimg = Variable(transform(image).unsqueeze(0))
output = yolo.forward(reimg)

# Shape [1, 125, 13, 13] to [ 5, 25,13, 13)
# For 13 x 13 are grid dimension
# That represent   x_offset   y_offset   relative width  relative height, objectness  class appartenance
output  = yolo_transform.reshape_output(output, len(anchors)// 2)
output = (output.data).cpu().numpy()
[offset_x, offset_y, relative_width, relative_height, objectness, img_classes] = yolo_transform.separate_output(output, class_number=len(classes))

x_pos = (input_size/13)*sigmoid(offset_x) + yolo_transform.get_x_grid(13,input_size)
y_pos = (input_size/13)*sigmoid(offset_y) + yolo_transform.get_y_grid(13,input_size)

reshaped_anchors = anchors.reshape(5, 2)

width_anchors = reshaped_anchors[:,0]
width_anchors = np.repeat(width_anchors,13*13,axis=0).reshape(5,13,13)
width = (416/13)*width_anchors * np.exp(relative_width)

height_anchors = reshaped_anchors[:,1]
height_anchors = np.repeat(height_anchors,13*13,axis=0).reshape(5,13,13)
height = (416/13)*height_anchors * np.exp(relative_height)

objectness = sigmoid(objectness)
fitlered_index_treshold = (objectness>treshold)

img_classes = np.transpose(img_classes, (0, 2, 3, 1))
img_classes = np.argmax(img_classes,axis=3)
final_classes = classes[img_classes]

x_pos = x_pos[fitlered_index_treshold]
y_pos = y_pos[fitlered_index_treshold]
width = width[fitlered_index_treshold]
height = height[fitlered_index_treshold]
final_classes = final_classes[fitlered_index_treshold]
image = image.resize((input_size, input_size), Image.ANTIALIAS)
objectness = objectness[fitlered_index_treshold]
print(width)

# Create figure and axes
fig,ax = plt.subplots(1)

ax.imshow(image)

little_box_w = 25
little_box_h = 14

for i in range(0, len(x_pos)):
    ax.add_patch(

        patches.Rectangle(
            (x_pos[i]- width[i]/2, y_pos[i]-height[i]/2),   # (x,y)
            width[i],          # width
            height[i],          # height
            fill=False  # remove background
            ,linewidth=math.exp(objectness[i])
            ,edgecolor="red"
        )
    )
    ax.add_patch(

        patches.Rectangle(
            (x_pos[i] - width[i]/2, y_pos[i] - height[i]/2 - little_box_h),  # (x,y)
            little_box_w,  # width
            little_box_h,  # height
            facecolor="red"
        )
    )

    ax.text(x_pos[i] - width[i]/2 , y_pos[i] - height[i]/2 - little_box_h/3, final_classes[i] + " " + str(objectness[i]), fontsize=8,color="white")

plt.show()