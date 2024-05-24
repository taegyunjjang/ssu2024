import cv2
import numpy as np

import torch
import torch.nn as nn

import sys
sys.path.append(r'/home/user/workspace/yolov1')
from nets.nn import resnet50
import torchvision.transforms as transforms
#from torchvision.ops import nms

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

COLORS = {'aeroplane': (0, 0, 0),
          'bicycle': (128, 0, 0),
          'bird': (0, 128, 0),
          'boat': (128, 128, 0),
          'bottle': (0, 0, 128),
          'bus': (128, 0, 128),
          'car': (0, 128, 128),
          'cat': (128, 128, 128),
          'chair': (64, 0, 0),
          'cow': (192, 0, 0),
          'diningtable': (64, 128, 0),
          'dog': (192, 128, 0),
          'horse': (64, 0, 128),
          'motorbike': (192, 0, 128),
          'person': (64, 128, 128),
          'pottedplant': (192, 128, 128),
          'sheep': (0, 64, 0),
          'sofa': (128, 64, 0),
          'train': (0, 192, 0),
          'tvmonitor': (128, 192, 0)}


def decoder(prediction):  ### prediction : 계산한 target값
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grid_num = 14
    boxes = []
    cls_indexes = []
    confidences = []
    cell_size = 1. / grid_num
    #각 image의 output tensor를 batch 축을 제거하고 [14,14,30] tensor로 변환
    # output tensor [14,14,30]로부 부터 bbox의 confidence score를 포함하는
    # tensor의 5번째와 10번재 plane 축 분리, 각각 [14,14] 크기를 갖는 tensor임
    # 두 tensor의 마지막 축을 하나 더 만들, [14,14,1]
    # 이 축을 중심으로 두 tensor들을 concatenation함, contain=[14,14,2]
    prediction = prediction.data.squeeze()  # 14x14x30, squeeze를 통해 0번 축 제거
    contain1 = prediction[:, :, 4].unsqueeze(2) #[14,14,1], bbox 1번의 confidence 분리 후 2번 축 추가
    contain2 = prediction[:, :, 9].unsqueeze(2) #[14,14,1], bbox 2번의 confidence 분리 후 2번 축 추가
    contain = torch.cat((contain1, contain2), 2) #[14,14,2]
    # contain의 값이 0.1를 갖는 성분들은 true값을 갖고 그렇지 않은 성분을 false로 set함
    mask1 = contain > 0.1
    # contain의 각 grid cell별로 confidence score의 max 값과 같은 bbox 위치에 mask2에
    # true값을 set함 그렇지 않은 bbox 위치에 false값을 set함
    mask2 = (contain == contain.max())  ### confidence 값이 높은 bbox에 대한 정보를 저장
    #mask1과 mask2의 성분별 덧셈을 수행하고 0보다 큰 성분에 대해 mask에 true, 
    #그렇지 않은 mask위치에 false값을 설정
    mask = (mask1 + mask2).gt(0)
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):  ### 2개의 bbox 확인
                if mask[i, j, b] == 1:
                    #mask tensor에서 grid cell의 bbox에서 값이 true인 경우에 다음 수행
                    #예측 output tensor로부터 bbox를 시작점과 끝점을 갖는 bbox로 변환
                    box = prediction[i, j, b * 5:b * 5 + 4]
                    contain_prob = torch.FloatTensor([prediction[i, j,b * 5 + 4]]).to(device)
                    xy = torch.FloatTensor([j, i]) * cell_size  ### normailze된 grid cell 시작위치 복원
                    box[:2] = box[:2] * cell_size + xy.to(device)  ### bbox의 중심점 복원
                    box_xy = torch.FloatTensor(box.size())
                    box_xy[:2] = box[:2] - 0.5 * box[2:]  ### bbox의 시작점 좌표 구하기
                    box_xy[2:] = box[:2] + 0.5 * box[2:]  ### bbox의 끝점 좌표 구하기
                    #bbox의 분류값과 confidence score 계산
                    max_prob, cls_index = torch.max(prediction[i, j, 10:], 0)
                    if float((contain_prob * max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1, 4))
                        cls_indexes.append(cls_index)
                        confidences.append(contain_prob * max_prob)  ### confidence score 값 계산
    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        confidences = torch.zeros(1)
        cls_indexes = torch.zeros(1)
    else:  ### 검출 하였다는 의미
        boxes = torch.cat(boxes, 0).to(device)  # (n,4)
        confidences = torch.cat(confidences, 0)  # (n,)
        cls_indexes = [item.unsqueeze(0) for item in cls_indexes]
        cls_indexes = torch.cat(cls_indexes, 0)  # (n,)
    #bbox정보와 confidence정보를 이용하여 NMS 알고리즘을 수행하여 중복된 bbox제거    
    keep = nms(boxes, confidences, threshold=0.5)
    return boxes[keep], cls_indexes[keep], confidences[keep]


def nms(b_boxes, scores, threshold=0.5):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x1 = b_boxes[:, 0].to(device)
    y1 = b_boxes[:, 1].to(device)
    x2 = b_boxes[:, 2].to(device)
    y2 = b_boxes[:, 3].to(device)
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:

        i = order.item() if (order.numel() == 1) else order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection

        over = intersection / union
        ids = (over <= threshold).nonzero().squeeze()
        # ids = torch.nonzero(ids, as_tuple=False).squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    return torch.LongTensor(keep)

def predict(model, img_name, root_path=''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    img = cv2.imread(root_path + img_name)
    h, w, _ = img.shape
    img = cv2.resize(img, (448, 448))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = (123, 117, 104)  # RGB
    img = img - np.array(mean, dtype=np.float32)

    transform = transforms.Compose([transforms.ToTensor(), ])
    img = transform(img)
    img = img[None, :, :, :]
    img = img.to(device)

    prediction = model(img).to(device)  # 1x14x14x30
    boxes, cls_indexes, confidences = decoder(prediction)

    #정규화되고 시작점과 끝점으로 표현된 bbox를 original image 상에서 크기로 변환함
    for i, box in enumerate(boxes):
        x1 = int(box[0] * w)
        x2 = int(box[2] * w)
        y1 = int(box[1] * h)
        y2 = int(box[3] * h)
        cls_index = cls_indexes[i]
        cls_index = int(cls_index)  # convert LongTensor to int
        conf = confidences[i]
        conf = float(conf)
        results.append([(x1, y1), (x2, y2), VOC_CLASSES[cls_index], img_name, conf])
    return results


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50().to(device)

    print('LOADING MODEL...')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load('./weights/yolov1_0010.pth')['state_dict'])
    model.eval()
    
    with torch.no_grad():
        image_name = './assets/person.jpg'
        image = cv2.imread(image_name)
        print('\nPREDICTING...')
        result = predict(model, image_name)

    for x1y1, x2y2, class_name, _, prob in result:
        color = COLORS[class_name]
        cv2.rectangle(image, x1y1, x2y2, color, 2)

        label = class_name + str(round(prob, 2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        p1 = (x1y1[0], x1y1[1] - text_size[1])
        cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]),
                      color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

    cv2.imwrite('./result.jpg', image)
    cv2.imshow('Prediction', image)
    cv2.waitKey(0)
