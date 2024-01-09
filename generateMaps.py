import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import time


# 区域放缩与面积判断
def resizePlusQualify(img, contours):
    # pts_x, pts_y = [],[]
    # width = img.shape[1]
    # height = img.shape[0]
    x, y, w, h = cv2.boundingRect(contours[0])
    imgCrop = img[y:y + h, x:x + w]
    # print('initial\n')
    # cv2.imshow("1", imgCrop)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    canvasRatio, convexRatio = width / height, w / h
    if canvasRatio >= convexRatio:
        newWidth = int(w * height / h)
        imgResize = cv2.resize(imgCrop, (newWidth, height), interpolation=cv2.INTER_NEAREST)
        imgResize = cv2.copyMakeBorder(imgResize, 0, 0, (width - newWidth) // 2,
                                       width - newWidth - (width - newWidth) // 2, cv2.BORDER_CONSTANT, value=0)
    else:
        newHeight = int(h * width / w)
        imgResize = cv2.resize(imgCrop, (width, newHeight), interpolation=cv2.INTER_NEAREST)
        imgResize = cv2.copyMakeBorder(imgResize, (height - newHeight) // 2,
                                       height - newHeight - (height - newHeight) // 2, 0, 0, cv2.BORDER_CONSTANT,
                                       value=0)
    # print("imgResize\n")
    # cv2.imshow("2", imgResize)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    areaRatio = (np.sum(imgResize) / 255) / (width * height)
    print(areaRatio)
    if areaRatio <= 0.5 or areaRatio >0.9:
        return imgResize, 0
    else:
        return imgResize, 1


# 增加障碍物
def addObstacles(img, trh=30):
    count = 0
    limitedNum = random.randint(0, 2)
    imgTemp = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    areaRatio = (np.sum(img) / 255) / (width * height)
    if areaRatio - 0.5 > 0.15 and limitedNum > 0:
        # spareSpace = (areaRatio - 0.5) * (width * height)
        while True:
            obs_x, obs_y = random.randint(trh, height - trh), random.randint(trh, width - trh)
            # print((np.sum(img[obs_y - trh // 2:obs_y + trh // 2,
            #            obs_x - trh // 2:obs_x + trh // 2]) / 255) / (trh * trh))
            if (np.sum(img[obs_y - trh:obs_y + trh,
                       obs_x - trh:obs_x + trh]) / 255) / (trh * trh * 4) > 0.9:
                obsPoints = []
                partL = trh // 2
                for i in range(4):
                    if i < 2:
                        orig_x, orig_y = obs_x - trh // 2, obs_y - trh // 2 + i * partL
                    else:
                        orig_x, orig_y = obs_x - trh // 2 + partL, obs_y + trh // 2 - (i % 2 + 1) * partL
                    x = random.randint(orig_x, orig_x + partL - 1)
                    y = random.randint(orig_y, orig_y + partL - 1)
                    obsPoints.append([x, y])
                obsPts = np.array(obsPoints, np.int32)
                cv2.drawContours(imgTemp, [obsPts], -1, [255], -1)
                # print(imgTemp.shape)
                # cv2.imshow("test-1",imgTemp)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                if random.random() < 0.5:
                    contours, _ = cv2.findContours(imgTemp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    print("contoursShape:",len(contours))
                    x, y, w, h = cv2.boundingRect(contours[0])
                    cv2.rectangle(imgTemp, (x, y), (x + w, y + h), [255], thickness=-1)
                    obsPts[:] = [[x, y], [x, y + h], [x + w, y + h], [x + w, y]]
                    print(obsPts)
                tempAreaRatio = (np.sum(imgTemp[obs_y - partL:obs_y + partL, obs_x - partL:obs_x + partL]) / 255) / (trh * trh)
                if tempAreaRatio < 0.5:
                    imgTemp.fill(0)
                    continue
                else:
                    imgTemp = img.copy()
                    # cv2.imshow("test0",imgTemp)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    cv2.drawContours(imgTemp, [obsPts], -1, [0], -1)
                    # cv2.imshow("test",imgTemp)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # cv2.imshow("test1",img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                areaRatio = (np.sum(imgTemp) / 255) / (width * height)
                print(areaRatio)
                if areaRatio - 0.5 < 0.15 or count == limitedNum:
                        processedImg = img
                        break
                else:
                    cv2.drawContours(img, [obsPts], -1, [0], -1)
                    count += 1
    else:
        processedImg = img
    return processedImg


# 生成地图
def generateMaps(img, num_side_list):
    flag = 0
    while not flag:
        num_vertices = 0
        while num_vertices not in num_side_list:
            # 图片初始化
            img.fill(0)
            # 随机取边
            num_vertices = random.choice(num_side_list)

            # 将背景分为八个部分
            # 随机生成多边形顶点的坐标
            # 0 7
            # 1 6
            # 2 5
            # 3 4
            partW = int(img.shape[1] / 2)
            partH = int(img.shape[0] / 4)
            areaIdx = sorted(random.sample(range(8), num_vertices))

            initialPoints = []
            for idx in areaIdx:
                # print(idx)
                if idx < 4:
                    orig_x, orig_y = 0, idx * partH
                else:
                    orig_x, orig_y = partW, height - (idx % 4 + 1) * partH
                x = random.randint(orig_x + 5, orig_x + partW - 3)
                y = random.randint(orig_y + 5, orig_y + partH - 3)
                # print(x,y)
                initialPoints.append([x, y])

            print(initialPoints)
            # print(initialPoints[0][1])
            pts = np.array(initialPoints, np.int32)
            cv2.drawContours(img, [pts], -1, [255], -1)
            # print('initial\n')
            # cv2.imshow("initial", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            retval = cv2.isContourConvex(contours[0])
            print(retval)
            if not retval:
                hull = cv2.convexHull(contours[0])
                # cv2.polylines(img, [hull], True, [255], 2)
                cv2.drawContours(img, [hull], -1, [255], -1)
                # print('convex\n')
                # cv2.imshow('convex', img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            print("contours:", len(contours))
            if len(contours) == 1:
                peri = cv2.arcLength(contours[0], True)
                vertex = cv2.approxPolyDP(contours[0], 0.01 * peri, True)
                num_vertices = len(vertex)
                # cornerPoints = np.array(vertex).reshape(-1, 2)
                print('优化后多边形的边数:', num_vertices)
                # if (np.sum(img) /255) / (width * height) > 0.9:
                #     img.fill(0)
            else:
                # img.fill(0)
                print("不连通的形状")
                num_vertices = 0
        img, flag = resizePlusQualify(img, contours)
    img = addObstacles(img, trh=25)
    return img, num_vertices


# 遴选理想地图数
def selectDesiredNumOfConvex(img, sideNum, eachNumDesired=2):
    sideTypeLen = len(sideNum)
    numList = []
    num=0
    while True:
        Map, num_ver = generateMaps(img, sideNum)
        print(num_ver)
        numList.append(num_ver)
        counter = Counter(numList)
        print(counter.values())
        if counter[num_ver] <= eachNumDesired:
            
            #cv2.imshow('img', Map)
            # cv2.waitKey(0)
            cv2.imwrite(r'generatedMaps\convex{}'.format(num)  + '.png', Map)##+ '_s{}no{}'.format(num_ver, counter[num_ver])
            num+=1
            # cv2.destroyAllWindows()
        else:
            if len(sideNum):
                sideNum.remove(num_ver)
        img.fill(0)
        Map.fill(0)
        if min(counter.values()) >= eachNumDesired and len(counter) == sideTypeLen:
            break
# 设置图片尺寸
width = 150
height = 200
# 初始化，创建一个黑色背景的栅格图片
image = np.zeros((height, width, 1), dtype=np.uint8)
image.fill(0)  # 设置背景颜色为黑色


def outmap():
    # 相关参数
    num_side = [3, 4, 6, 8]  # 需要的多边形类型
    sel_num_side=random.sample(num_side, 1)
    numDesired = 1  # 需要每种多边形的数量
    selectDesiredNumOfConvex(image, sel_num_side, numDesired)

if __name__ == "__main__":
   # outmap()
    # 设置图片尺寸
    width = 150
    height = 200
    # 初始化，创建一个黑色背景的栅格图片
    image = np.zeros((height, width, 1), dtype=np.uint8)
    image.fill(0)  # 设置背景颜色为黑色

    # 相关参数
    num_side = [3, 4, 6, 8]  # 需要的多边形类型
    numDesired = 1000  # 需要每种多边形的数量

    selectDesiredNumOfConvex(image, num_side, numDesired)
