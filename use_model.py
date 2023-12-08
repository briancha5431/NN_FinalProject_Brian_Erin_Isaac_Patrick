import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import ( non_max_suppression, cellboxes_to_boxes, 
                   plot_image)


seed = 123
torch.manual_seed(seed)

# Set hyperparameters
NUM_IMAGES = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16     # The paper uses 64, but we will 16 for now
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "our_data/images"
LABEL_DIR = "our_data/labels"

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

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    model.load_state_dict(torch.load(LOAD_MODEL_FILE)["state_dict"])
    train_dataset = VOCDataset("our_data/test.csv",  # The larger dataset(s) are given by "data/train.csv" or "data/100examples.csv"
                               transform=transform,
                               img_dir=IMG_DIR,
                               label_dir=LABEL_DIR)  
    
    train_loader = DataLoader(dataset = train_dataset, 
                              batch_size = BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY,
                              shuffle=False,
                              drop_last=False)   

    for x, y in train_loader:
        x = x.to(DEVICE)
        print(x, y)
        print(x.shape, y.shape)
        for idx in range(x.shape[0]):
            with torch.no_grad():
                predictions = model(x)
            bboxes = cellboxes_to_boxes(predictions)
            bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.3, box_format='midpoint')
            
            class_ids = []
            for box in bboxes:
                print(box)
                class_idx = int(box[0])
                class_ids.append(class_idx)
                
            print(f'{class_ids=}')
                
            plot_image(x[idx].permute(1,2,0).to('cpu'), bboxes, f'/research/cwloka/projects/tool_building/YOLO_project/our_images_output/image{idx}', class_ids)
            


if __name__== "__main__":
    main()

