import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm       # Progress Bar
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (intersection_over_union, non_max_suppression, 
                   mean_average_precision, cellboxes_to_boxes, 
                   get_bboxes, plot_image, save_checkpoint,
                   load_checkpoint)
from loss import YoloLoss


# Reproduce same results for consistency
seed = 123
torch.manual_seed(seed)

# Set hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16     # The paper uses 64, but we will 16 for now
WEIGHT_DECAY = 0    # Don't train weight decay, just in case GPU cannot handle it
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "model.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"


class Compose(object):
    """
        This class better supports data augmentation+transforms if we require it.
    """

    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes    # Transformations are applied directly to images
    
        return img, bboxes


# Resizing our image to the same shape as presented in the paper
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])


def train_fn(train_loader, model, optimizer, loss_fn): 
    """
        Our custom training function for our YOLO Object Detection model.
        We will use the VOC dataset as the default data to train our model on.
    """

    loop = tqdm(train_loader, leave=True)       # Set up the progress bar
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)       # Set the predicted and true coords to GPU/CPU settings

        # Make each prediction
        pred = model(x)
        loss = loss_fn(pred, y)
        mean_loss.append(loss.item())

        # Update our parameters after each prediction
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the progress bar
        loop.set_postfix(loss = loss.item())
    
    print(f"Training mean loss was {sum(mean_loss) / len(mean_loss)}")
    return sum(mean_loss) / len(mean_loss)

def val_fn(val_loader, model, loss_fn):
    model.eval()  # Set the model to evaluation mode
    mean_val_loss = []
    with torch.no_grad():  # Disable gradient calculation
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = loss_fn(pred, y)
            mean_val_loss.append(loss.item())
    model.train()  # Set the model back to training mode
    print(f"Validation mean loss was {sum(mean_val_loss) / len(mean_val_loss)}")
    return sum(mean_val_loss) / len(mean_val_loss)

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        # Load the progress from previous iterations if needed
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
    
    # Obtain our training + testing datasets from our given data
    train_dataset = VOCDataset("data/train.csv",  # The larger dataset(s) are given by "data/train.csv" or "data/100examples.csv"
                               transform=transform,
                               img_dir=IMG_DIR,
                               label_dir=LABEL_DIR)  
    
    test_dataset = VOCDataset("data/test.csv",
                              transform=transform,
                              img_dir=IMG_DIR,
                              label_dir=LABEL_DIR)
    
    # Obtain our training and testing dataloaders using our hyperparameters
    train_loader = DataLoader(dataset = train_dataset, 
                              batch_size = BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY,
                              shuffle=True,
                              drop_last=False)              # Do not perform gradient descent on only 1 or 2 examples
        
    test_loader = DataLoader(dataset = test_dataset, 
                              batch_size = BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY,
                              shuffle=True,
                              drop_last=True)              # Set this to true if we use larger datasets (not 8examples)
    
    # Training Procedure
    with open('train_loss.txt', 'a') as f_train, open('val_loss.txt', 'a') as f_val:
        for epoch in range(EPOCHS):

            # Show the predictions for our bounding boxes for each image
            # We can comment this code out, if necessary
            # for x, y in train_loader:
            #     x = x.to(DEVICE)
            #     for idx in range(8):
            #         bboxes = cellboxes_to_boxes(model(x))
            #         bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format='midpoint')
            
                # class_ids = []
                # for box in bboxes:
                #     class_idx = int(box[0])
                #     class_ids.append(class_idx)
                #     print(f'Detected class: {class_idx}')
                
            #         plot_image(x[idx].permute(1,2,0).to('cpu'), bboxes, f'/research/cwloka/projects/tool_building/YOLO_project/bounding_box_image/image{idx}', class_ids)
                
            #     import sys
            #     sys.exit()

            # Obtain our bounding boxes for training
            pred_boxes, target_boxes = get_bboxes(
                train_loader, model, iou_threshold=0.5, threshold=0.4   # These parameters work best for mean average precision
            )

            # Calculate mean average precision on the derived bounding boxes
            mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format='midpoint'
            )

            print(f"Train mAP: {mean_avg_prec}")    # Display our statistic

            if mean_avg_prec > 0.9:
                checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
                import time
                time.sleep(10)
    
            train_loss = train_fn(train_loader, model, optimizer, loss_fn)
            val_loss = val_fn(test_loader, model, loss_fn)

            f_train.write(f"{epoch},{train_loss}\n")
            f_val.write(f"{epoch},{val_loss}\n")



if __name__== "__main__":
    main()

'''
Training Data Text File Format
Class, X_midpoint (relative to cell, 0 to 1), Y_midpoint (relative to cell, 0 to 1), Width (relative to entire image), Height (relative to entire image)
'''