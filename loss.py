import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    '''
        Calculates the overall loss of our YOLO v1 model.
    '''
    
    # S == split size, B == bounding boxes, C == # of classes
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')      # don't average the losses!
                                                    # Sum errors instead

        self.S = S  # Split size or grid size (S x S grid)
        self.B = B  # Number of bounding boxes predicted per cell
        self.C = C  # Number of classes (originally 20)

        self.lambda_noobj = 0.5     # pre-defined constant to calculate no object prediction loss
        self.lambda_coord = 5       # pre-defined constant to calculate one or more objects prediction loss
    
    def forward(self, predictions, target):
        # Make sure predictions are reshaped to (S, S, 30)
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)        # -1 keeps the shape of examples

        # 0 to 19 inclusive: class probabilities
        # 20: class score (is there an object or not)
        # Note: '...' is a special notation that can be used in array slicing (multidimensional arrays)
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])   # 21:25 are the coords for bounding boxes
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])   # 21:25 are the coords for bounding boxes
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        
        ious_maxes, bestbox = torch.max(ious, dim=0)        # Obtain the best bounding box for a specific cell

        exists_box = target[..., 20].unsqueeze(3)           # 0 or 1 depending on if there is an object in that cell
                                                            # unsqueeze(3): add singleton dimension at the end of the tensor
                                                            # Is there an object in cell i? 


        ############################################################## 
        ###############        BOX COORDINATES        ################
        ##############################################################

        # Equation is given in the paper on page 4
        # ...s indicate that we perform operations on all dimensions, whereas we only want to
        # operate on specific classes (so we choose specific indices for C)
        box_predictions = exists_box * (
            (
                # which is the best box?
                # remember, 21:25 and 26:30 are BOX COORDINATES
                bestbox * predictions[..., 26:30] +
                (1 - bestbox) * predictions[..., 21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)         # Add epsilon in case predictions are 0
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        box_loss = self.mse(
            # end_dim=2 makes it so that (N, S, S, 4) -> (N*S*S, 4)
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )


        ##############################################################
        ###############          OBJECT LOSS          ################
        ##############################################################

        pred_box = (bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21])

        # Convert each layer to (N*S*S, 1)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )


        ##############################################################
        ###############         NO OBJECT LOSS        ################
        ##############################################################

        # (N, S, S, 1) -> (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )


        ##############################################################
        ###############          CLASS LOSS           ################
        ##############################################################

        # (N, S, S, 20) -> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )


        ##############################################################
        ###############         OVERALL LOSS          ################
        ##############################################################

        loss = (
            self.lambda_coord * box_loss            # First two rows of loss in the paper
            + object_loss                           # Third row
            + self.lambda_noobj * no_object_loss    # Fourth row
            + class_loss                            # Last row
        )

        return loss

