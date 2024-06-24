import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import *


class yoloLoss(Module):
    def __init__(self, num_class=20):
        super(yoloLoss, self).__init__()
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.S = 14
        self.B = 2
        self.C = num_class
        self.step = 1.0 / 14

    def compute_iou(self, box1, box2, index):
        """ CIoU로 변경 """
        box1 = torch.clone(box1)
        box2 = torch.clone(box2)
        box1 = self.conver_box(box1, index)
        box2 = self.conver_box(box2, index)
        x1, y1, w1, h1 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        x2, y2, w2, h2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        
        # 좌상단과 우하단 좌표 계산
        x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
        x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2
        x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
        x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2
        
        # 교집합 영역 계산
        inter_w = torch.max(torch.tensor(0.0), torch.min(x1_max, x2_max) - torch.max(x1_min, x2_min))
        inter_h = torch.max(torch.tensor(0.0), torch.min(y1_max, y2_max) - torch.max(y1_min, y2_min))
        inter = inter_w * inter_h
        
        # 합집합 영역 계산
        union = w1 * h1 + w2 * h2 - inter
        
        # 중심점 간 거리 제곱 계산
        dist_x = (x2 - x1) ** 2
        dist_y = (y2 - y1) ** 2
        dist_center = dist_x + dist_y
        
        # 대각선 길이 제곱 계산
        diag = torch.max(torch.tensor(0.0), (x1_max - x1_min) ** 2 + (y1_max - y1_min) ** 2) + \
               torch.max(torch.tensor(0.0), (x2_max - x2_min) ** 2 + (y2_max - y2_min) ** 2)
        
        # CIoU 계산
        ciou = inter / union - dist_center / diag
        
        return ciou
        

    def conver_box(self, box, index):
        i, j = index
        box[:, 0], box[:, 1] = [(box[:, 0] + i) * self.step - box[:, 2] / 2,
                                (box[:, 1] + j) * self.step - box[:, 3] / 2]
        box = torch.clamp(box, 0)
        return box

    def forward(self, pred, target):
        batch_size = pred.size(0)
        #target tensor = [batch_size,14,14,30]
        #pred tensor = [batch_size,14,14,30]       
        
        #target matirx tensor로부터 bbox 정보만 분리, [batch_size,14,14,10]
        #분리된 bbox tensor=[batch_size,14,14,10] --> [batch_size,14,14,2,5]로 변경
        #이후 예측 tensor에 관해 동일 반복
        target_boxes = target[:, :, :, :10].contiguous().reshape(
            (-1, self.S, self.S, 2, 5))
        pred_boxes = pred[:, :, :, :10].contiguous().reshape(
            (-1, self.S, self.S, 2, 5))
        
        #target matirx tensor로부터 grid cell의 class probability 분리, [batch_size,14,14,20]
        #예측 tensor에 관해 동일 반복
        target_cls = target[:, :, :, 10:]
        pred_cls = pred[:, :, :, 10:]
        
        #target tensor로부터 object가 위치하는 batch image 위치와 
        # grid cell에서 좌표를 계산
        #여기서, Obj_mask는 target tensor로부터 물체가 위치하는 
        # grid cell 위치에 true값을 갖는 mask tensor임
        #index tensor는 물체가 위치하는 batch image 위치와 grid cell에서
        # 좌표값의 index를 갖음
        obj_mask = (target_boxes[..., 4] > 0).byte() 
        sig_mask = obj_mask[..., 1].bool()
        index = torch.where(sig_mask == True)
        #object가 위치하는 grid cell마다 2개 bbox와 ground truth bbox(이하 GT)와
        # IOU값을 계산하고 예측된 2 bbox중 IOU가 최대인 bbox를 찾아 GT bbox를 예측한 
        # bbox로 선정함.
        # 그리고 obj_mask에 나머지 bbox의 confidence score 위치에 0으로 reset함
        for img_i, y, x in zip(*index):
            img_i, y, x = img_i.item(), y.item(), x.item()
            pbox = pred_boxes[img_i, y, x]
            target_box = target_boxes[img_i, y, x]
            ious = self.compute_iou(pbox[:, :4], target_box[:, :4], [x, y])
            iou, max_i = ious.max(0)
            #pred_boxes[img_i, y, x, max_i, 4] = iou.item()
            #pred_boxes[img_i, y, x, 1 - max_i, 4] = 0
            obj_mask[img_i, y, x, 1 - max_i] = 0
        
        #obj_mask를 반전시켜 물체가 위치하지 않은 mask tensor를 구성
        obj_mask = obj_mask.bool()
        noobj_mask = ~obj_mask
        
        #물체가 존재하지 않은 bbox의 confidence score 오차의 loss값 계산 
        noobj_loss = F.mse_loss(pred_boxes[noobj_mask][:, 4],
                                target_boxes[noobj_mask][:, 4],
                                reduction="sum")
        #물체가 존재하는 bbox의 confidence score 오차의 loss값 계산
        obj_loss = F.mse_loss(pred_boxes[obj_mask][:, 4],
                              target_boxes[obj_mask][:, 4],
                              reduction="sum")
        #물체가 존재하는 bbox의 중심점의 오차에 loss값 계산
        xy_loss = F.mse_loss(pred_boxes[obj_mask][:, :2],
                             target_boxes[obj_mask][:, :2],
                             reduction="sum")
        
        #물체가 존재하는 bbox의 width와 height의 오차에 loss값 계산
        wh_loss = F.mse_loss(torch.sqrt(target_boxes[obj_mask][:, 2:4]),
                             torch.sqrt(pred_boxes[obj_mask][:, 2:4]),
                             reduction="sum")
        
        #물체가 존재하는 grid cell의 class probability의 오차에 loss값 계산
        class_loss = F.mse_loss(pred_cls[sig_mask],
                                target_cls[sig_mask],
                                reduction="sum")

        #각 loss값의 합산
        loss = obj_loss + self.lambda_noobj * noobj_loss \
                    + self.lambda_coord * xy_loss + self.lambda_coord * wh_loss \
                    + class_loss
        
        return loss/batch_size