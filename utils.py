import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches    
from collections import Counter

id_object = {
    0 : 'airplane',
    1 : 'bicycle',
    2 : 'bird',
    3 : 'boat',
    4 : 'bottle',
    5 : 'bus',
    6 : 'car',
    7 : 'cat',
    8 : 'chair',
    9 : 'cow',
    10 : 'dining table',
    11 : 'dog',
    12 : 'horse',
    13 : 'motorbike',
    14 : 'person',
    15 : 'plant',
    16 : 'sheep',
    17 : 'sofa',
    18 : 'train',
    19 : 'tv' 
}

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
        This function calculates the intersection over union.

        Parameters:
            boxes_preds (tensor) : predictions of bounding boxes --> shape: (batch_size, 4)
            boxes_labels (tensor): true labels of bounding boxes --> shape: (batch_size, 4)
            box_format (str)     : midpoint or corners if boxes   --> (x, y, w, h) or (x1, y1, x2, y2)
        
        Returns:
            Intersection over union for all examples (tensor)
    """

    if box_format == 'midpoint':
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    
    if box_format == 'corners':
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)

    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Find the intersection
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)              # clamp(0) is for the case where they do not intersect

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6) # epsilon, in case denominator is 0


def non_max_suppression(bboxes, iou_threshold, threshold, box_format='corners'):
    """
        This function performs non max supression given certain bboxes.

        Parameters:
            bboxes (list)        : list of lists containing all bboxes within each bbox specified as
                                   [class_pred, prob_score, x1, y1, x2, y2].
            iou_threshold (float): threshold where predicted bboxes are correct.
            threshold (float)    : threshold to remove predicted boxes (independent of iou).
            box_format (str)     : the midpoint or corners used to specify such bboxes.

        Returns:
            One or more bboxes after performing non max suppression given a specified iou threshold (list).
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]          # keep bbox if prob_score > threshold
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    bboxes_after_nms = []
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [box for box in bboxes if box[0] != chosen_box[0] or 
                  intersection_over_union(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), box_format=box_format) 
                  < iou_threshold]
        
        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20):
    """
        This function calculates the mean average precision given prediction + true bounding boxes.

        Parameters:
            pred_boxes (list)    : list of lists containing all bboxes with each bbox represented as 
                               [train_idx, class_prediction, prob_score, x1, y1, x2, y2].
            true_boxes (list)    : true versions of the pred_boxes.
            iou_threshold (float): threshold where predicted bboxes are correct.
            box_format (str)     : "midpoint" or "corners" used to specify such bboxes.
            num_classes (int)    : number of classes.

        Returns:
            The mAP value across all classes given a specified iou threshold (float).
    """

    # Store all the average precisions for the given classes
    average_precisions = []

    # Keep for checking denominators (potentially 0)
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Only add predictions+targets that belong to current class
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)
        
        # Find the number of bboxes for each training example (how many ground_truth bboxes per example)
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        
        # Sort by box probabilities
        detections.sort(key=lambda x: x[2], reverse=True)

        TP = torch.zeros((len(detections)))      # True Positives
        FP = torch.zeros((len(detections)))      # False Positives
        total_true_bboxes = len(ground_truths)

        # Move on if there are no ground truths for this class
        if total_true_bboxes == 0:
            continue
            
        for detection_idx, detection in enumerate(detections):
            # Keep ground_truths that have the same idx as the detection
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format=box_format)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
                
            if best_iou > iou_threshold:
                # Check groudnd truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # We have found a true positive
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    # Otherwise, we have found a false positive
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        # Find cumulative sums of TPs and FPs
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / (total_true_bboxes + epsilon)                         # Recall statistic
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))     # Precision
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        # Finishing touches using numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))
    
    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes, save_path, class_ids):
    """
        This function plots the predicted bounding boxes directly on the image.
    """

    img = np.array(image)
    height, width, _ = img.shape

    fix, ax = plt.subplots(1)
    
    # Display the image first
    ax.imshow(img)

    # Note: box[0] is the x midpoint, box[1] is the y midpoint
    #       box[2] is the width, box[3] is the height

    # Create a rectangular patch
    for i, box in enumerate(boxes):
        box = box[2:]

        # Before we do anything, make sure our box shape is correct
        assert len(box) == 4, "Got more values than (x, y, w, h) in a bbox!"

        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2

        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth = 1,
            edgecolor='r',
            facecolor='none'
        )

        # Add the patch onto the given axes
        ax.add_patch(rect)
        
        # Annotate with class ID
        ax.text(
            upper_left_x * width,
            upper_left_y * height,
            str(id_object[class_ids[i]]),
            color='r',
            backgroundcolor='white',
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round4')
        )
        
    plt.savefig(save_path)
    
    plt.show()


def convert_cellboxes(predictions, S=7):
    """
        This function converts bounding boxes with an image split size of S into 
        ENTIRE image ratios rather than relative cell ratios.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, 30)     # (S, S, 30)

    bboxes1 = predictions[..., 21:25]       # Obtain coords for first bbox
    bboxes2 = predictions[..., 26:30]       # Obtain coords for second bbox
    scores = torch.cat((predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0)

    # Obtain best scoring bbox for each cell
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)

    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]

    # Conversion of bbox + predictions
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)
    converted_preds = torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)

    return converted_preds


    

def cellboxes_to_boxes(out, S=7):
    """
        This function turns each bounding box in a particular cell of an image to a bounding box
        for the whole image.
    """

    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S*S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S*S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        
        all_bboxes.append(bboxes)

    return all_bboxes


def get_bboxes(loader, model, iou_threshold, threshold, pred_format='cells', box_format='midpoint', device='cuda'):
    """
        This function obtains the bounding boxes given a certain object detection model and threshold.
    """

    all_pred_boxes = []
    all_true_boxes = []

    # We want to be in evaluation mode
    model.eval()
    train_idx = 0

    # Analyze the loader
    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)
        
        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            # Obtain non max suppression bboxes
            nms_boxes = non_max_suppression(bboxes[idx], iou_threshold=iou_threshold,
                                            threshold=threshold, box_format=box_format)
            
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)
            
            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)
            
            train_idx += 1
        
        model.train()
        return all_pred_boxes, all_true_boxes


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
        Misc. function, just in case we wish to save the current checkpoint.
    """

    print("Saving checkpoint...")
    torch.save(state, filename)



def load_checkpoint(checkpoint, model, optimizer):
    """
        Misc. function, just in case we wish to load the current checkpoint.
    """

    print("Loading checkpoint...")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
