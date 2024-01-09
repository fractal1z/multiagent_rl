from pyproj import Proj
import pyproj
import folium
import json
import cv2
import numpy as np
import math

# 打开数据文件并加载数据
with open(r'./挑战杯试题.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 访问数据
#print(data)
taskArea = data['taskArea']
vesPostion = data['vesPostion']
targets = data['targets']
changeTargets=data["changeTargets"]

# 定义UTM投影坐标系
utm_zone = 51  # UTM区域
utm_band = 'N'  # UTM带

# 定义经纬度坐标系
lon_lat_proj = Proj(proj='latlong', datum='WGS84')
# 定义UTM投影坐标系
utm_proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84',south=False)
merc_proj = Proj(proj='merc', ellps='WGS84',lat_ts=38.7 )

# 将边界点经纬度坐标转换为UTM坐标,转换为米
taskArea_utm = []
for lon, lat in taskArea:
    x, y = lon_lat_proj(lon, lat)
    x, y = pyproj.transform(lon_lat_proj, utm_proj, x, y)
    taskArea_utm.append([x, y])
#taskArea_utm=[[coord[1]*111.32*1000*math.cos(math.radians(coord[1])),coord[0]*111.32*1000]for coord in taskArea]
#print(taskArea_utm)
# 根据utm坐标画地图,地图包含范围
taskArea_utm_x = [coord[0] for coord in taskArea_utm]
taskArea_utm_y = [coord[1] for coord in taskArea_utm]
min_taskArea_utm_x, max_taskArea_utm_x = min(taskArea_utm_x), max(taskArea_utm_x)
min_taskArea_utm_y, max_taskArea_utm_y = min(taskArea_utm_y), max(taskArea_utm_y)
area_x_range = [min_taskArea_utm_x-950, max_taskArea_utm_x+950]
area_y_range = [min_taskArea_utm_y-150, max_taskArea_utm_y+150]
#print(area_x_range,area_y_range)
# 图像尺寸确定 150*200
scale=50
image_width = math.ceil((area_x_range[1]-area_x_range[0])/scale)
image_height = math.ceil((area_y_range[1]-area_y_range[0])/scale)
#print(image_width,image_height)
# 计算边界像素坐标, 向四周取舍
pixel_coordinates = []
for coordinate in taskArea_utm:
    x = (coordinate[0] - area_x_range[0]) / (area_x_range[1] - area_x_range[0]) 
    y = (coordinate[1] - area_y_range[0]) / (area_y_range[1] - area_y_range[0]) 
    if x>0.5:
        x=math.ceil(x*image_width)
    else:
        x=math.floor(x*image_width)
    if y>0.5:
        y=math.ceil(y*image_height)
    else:
        y=math.floor(y*image_height)
    pixel_coordinates.append([x, image_height-y])
#print(pixel_coordinates)
# 创建空白图像
image = np.zeros((image_height, image_width, 1), dtype=np.uint8)
# 绘制多边形
polygon_points = np.array(pixel_coordinates, np.int32)
#cv2.polylines(image, [polygon_points], isClosed=True, color=255, thickness=1)
cv2.fillPoly(image, [polygon_points], color=255)

# 显示图像
# cv2.imshow('Polygon', image)
# #cv2.imwrite("test.png", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


## 将智能体经纬度坐标转换为UTM坐标,转换为米
#print(vesPostion)
vesPostion_utm= []
for lon, lat in vesPostion.values():
    x, y = lon_lat_proj(lon, lat)
    x, y = pyproj.transform(lon_lat_proj, utm_proj, x, y)
    vesPostion_utm.append([x, y])

#print(vesPostion_utm)
# 计算智能体小船儿像素坐标
agent_pos = []
for coordinate in vesPostion_utm:
    x = (coordinate[0] - area_x_range[0]) / (area_x_range[1] - area_x_range[0]) 
    y = (coordinate[1] - area_y_range[0]) / (area_y_range[1] - area_y_range[0]) 
    x=int(x*image_width)
    y=int(y*image_height)
    agent_pos.append([x, image_height-y])
print(agent_pos)

## 将目标坐标转换为UTM坐标,转换为米
#print(targets)
targets_utm=[]
for target  in targets:
    lon, lat = target['coord']
    utm_x, utm_y = utm_proj(lon, lat)
    target['coord'] = [utm_x, utm_y]
    targets_utm.append(target)
#print(vesPostion_utm)
# for target in targets_utm:
#     print(target)
# 计算目标像素坐标,改角度
target_dict = []
for target_utm  in targets_utm:
    coordinate = target_utm['coord']
    x = (coordinate[0] - area_x_range[0]) / (area_x_range[1] - area_x_range[0]) 
    y = (coordinate[1] - area_y_range[0]) / (area_y_range[1] - area_y_range[0]) 
    x=int(x*image_width)
    y=int(y*image_height)
    target_utm['coord'] = [x, image_height-y]
    target_utm['controler_fois'] = 0
    target_utm['savoir'] = False
    target_utm['en'] = True
    target_dict.append(target_utm)
print("target:")
for target in target_dict:
    print(target)


## 将可变目标坐标转换为UTM坐标,转换为米
#print(targets)
ctargets_utm=[]
for ctarget  in changeTargets:
    lon, lat = ctarget['coord']
    utm_x, utm_y = utm_proj(lon, lat)
    ctarget['coord'] = [utm_x, utm_y]
    ctargets_utm.append(ctarget)
ctarget_dict = []
for ctarget_utm  in ctargets_utm:
    coordinate = ctarget_utm['coord']
    x = (coordinate[0] - area_x_range[0]) / (area_x_range[1] - area_x_range[0]) 
    y = (coordinate[1] - area_y_range[0]) / (area_y_range[1] - area_y_range[0]) 
    x=int(x*image_width)
    y=int(y*image_height)
    ctarget_utm['coord'] = [x, image_height-y]
    ctarget_utm['controler_fois'] = 0
    ctarget_dict.append(ctarget_utm)
print("changetarget:")
for ctarget in ctarget_dict:
    print(ctarget)
print(list(item['savoir'] for item in target_dict))

def xy2coord(x,y):
    coordinate=[0]*2
    
    utmx=x/image_width*(area_x_range[1] - area_x_range[0]) +area_x_range[0]
    utmy=(image_height-y)/image_height*(area_y_range[1] - area_y_range[0]) +area_y_range[0]
    lon, lat = pyproj.transform(utm_proj,lon_lat_proj, utmx, utmy)
    return [lon,lat]


def coord2xy(lon,lat):
    xy=[0]*2
    lon, lat = lon_lat_proj(lon, lat)
    xutm, yutm = pyproj.transform(lon_lat_proj, utm_proj, lon, lat)
    x = (xutm - area_x_range[0]) / (area_x_range[1] - area_x_range[0]) 
    y = (yutm - area_y_range[0]) / (area_y_range[1] - area_y_range[0]) 
    x=(x*image_width)#int(x*image_width)
    y=image_height-(y*image_height)#int(y*image_height)
    return [x,y]

#test
# for lon, lat in vesPostion.values():
#     print(coord2xy(lon,lat))
#     print( xy2coord(coord2xy(lon,lat)[0],coord2xy(lon,lat)[1]))

if __name__ == '__main__':
    #显示图像
    cv2.imshow('Polygon', image)
    cv2.imwrite("test.png", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    areaRatio = (np.sum(image) / 255) / (200 * 150)
    print(areaRatio)
    # print(xy2coord(16,70))
    # print(xy2coord(16,115))
    #print(xy2coord(121, 93))
