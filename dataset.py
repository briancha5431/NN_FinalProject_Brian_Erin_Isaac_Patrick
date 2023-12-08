import torch
import os
import pandas as pd
from PIL import Image

# torch.utils.data.Dataset is an abstract class in PyTorch that represents a dataset
# create a custom subclass that overrides two methods: __len__ and __getitem__.
class VOCDataset(torch.utils.data.Dataset):
    '''
        This class initializes and represents the VOC data of our YOLO model.
    '''

    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        '''
            csv_file:   the path to the trainingd/test datasets containing the image and label files.
            img_dir:    the image directory
            label_dir:  the labels directory
            S:          the split size
            B:          the number of bounding boxes
            C:          the number of classes
            transform:  true or false whether we want to apply tranformations to the images
        '''

        self.annotations = pd.read_csv(csv_file)        # contents of training/testing .csv file(s) are the annotations
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        '''
            Define the length function for our VOC Dataset class.
        '''

        return len(self.annotations)

    def __getitem__(self, index):
        '''
            Define the indexing ('get') function for our VOC Dataset class.
        '''
        # Obtain the text file within the training/test .csv files
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []

        with open(label_path) as f:
            for label in f.readlines():
                # class is an integer
                # midpoint + width + height are floats
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                # convert to floats and integers, then append
                boxes.append([class_label, x, y, width, height])
        
        # Obtain the image file within the training/test .csv files
        image_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(image_path)
        boxes = torch.tensor(boxes)                                 # tranformations must be made as a Pytorch Tensor

        if self.transform:
            # If we apply transformations, the bounding boxes + image must be transformed together
            image, boxes = self.transform(image, boxes)
        
        # If this line gives a bug, we need to intialize the last dimension to 30: use self.C + 5 * self.B
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))    # (S, S, 30) 

        for box in boxes:
            class_label, x, y, width, height = box.tolist()         # after transformation, convert back to list
            class_label = int(class_label)

            # Obtain the cell row and column that the class label belongs to (relative to the whole image)
            i, j = int(self.S * y), int(self.S * x)

            # Obtain the cell row and column of the class label w/ respect to a specific cell
            x_cell, y_cell = self.S * x - j, self.S * y - i

            # Obtain the width and height of the class label w/ respect to a specific cell
            width_cell, height_cell = width * self.S, height * self.S

            # Check the case when there is no object in the cell
            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix

    

