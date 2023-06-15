import numpy as np
import cv2
import json

from PIL import Image
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from shapely.geometry import Polygon

from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import RANSACRegressor

from skimage import io, img_as_float, color, draw, measure, transform
from skimage.transform import warp
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.color import label2rgb, rgba2rgb
from skimage.exposure import rescale_intensity
from skimage.morphology import (erosion, dilation, closing, opening, disk, binary_erosion, binary_dilation,
                                area_closing, area_opening, square, remove_small_objects)

from Perspective import equir2pers

standardCeilingHeigh = 2.4

ItemsName = {
    'void':  {      'color': [0,0,0]        , 'type' : 0 },
    'bed':  {       'color': [128,0,0]      , 'type' : 2 },
    'painting': {   'color': [0,128,0]      , 'type' : 1 },
    'table':  {     'color': [128,128,0]    , 'type' : 3 },
    'mirror':  {    'color': [0,0,128]      , 'type' : 1 },
    'window':  {    'color': [128,0,128]    , 'type' : 1 },
    'curtain':  {   'color': [0,128,128]    , 'type' : 1 },
    'chair':  {     'color': [128,128,128]  , 'type' : 3 },
    'light':  {     'color': [64,0,0]       , 'type' : 4 },
    'sofa':  {      'color': [192,0,0]      , 'type' : 3 },
    'door': {       'color': [64,128,0]     , 'type' : 1 },
    'cabinet': {    'color': [192,128,0]    , 'type' : 2 },
    'bedside': {    'color': [64,0,128]     , 'type' : 3 },
    'tv': {         'color': [192,0,128]    , 'type' : 4 },
    'shelf': {      'color': [64,128,128]   , 'shelf' : 2 }   
}

def detInputData():
    # load image
    global img
    img = Image.open("input/test.jpg")
    img = np.array(img)
    global pano_H
    global pano_W
    pano_H = img.shape[0]
    pano_W = img.shape[1]
    newsize = (pano_W, pano_H)

    # load corner map
    global cmap
    cmap = Image.open("tmp/test_cmap.jpg")
    cmap = cmap.resize(newsize)
    cmap = np.array(cmap)

    # load edge map
    global emap
    emap = Image.open("tmp/test_emap.jpg")
    emap = emap.resize(newsize)
    emap = np.array(emap)

    # load edge map
    global smap
    global smap_rgb
    smap = Image.open("tmp/test_segmentation_raw.png")
    smap_rgb = smap.convert("RGB")
    smap_rgb = np.array(smap_rgb)
    smap = np.array(smap)
    
    # load json bbox
    global boxes_data
    with open("tmp/test_boxes.json") as f:
        boxes_data = json.load(f)

#Get corners
def cmap2corners(cmap):
    cmap = cmap/255.
    cmap_ = np.hstack((cmap,cmap,cmap))
    cmap_prob = cmap_.copy()

    th = 0.1
    cmap_[cmap_<th] = 0
    cmap_[cmap_>th] = 1
    label_cmap = label(cmap_)
    regions = regionprops(label_cmap, cmap_prob)
    
    cor_uv = []
    for props in regions:
        y0, x0 = props.weighted_centroid 
        if x0 > (pano_W-1) and x0 < (pano_W*2+1):
            cor_uv.append([x0-pano_W,y0])

    cor_uv = np.array(cor_uv)
    sorted_cor_uv = sorted(cor_uv, key=lambda x: x[0])
    sorted_cor_uv = np.array([arr.tolist() for arr in sorted_cor_uv])

    return sorted_cor_uv

#Get wall map
def getWallMap(map):
    cmap_ = map/255.
    thresh = threshold_otsu(cmap_)

    element = np.array([[0,1,0],
                        [1,1,1],
                        [0,1,0]])

    bw = closing(cmap_ > thresh, element)
    inversbw = abs(1-bw)

    label_image  = label(inversbw, background=0)
    dilated = label_image


    for i in range(15):
        dilated = dilation(dilated, element)
    
    heigth = len(dilated)
    width = len(dilated[0])
    wallL = dilated[int(heigth/2)][0]
    wallR = dilated[int(heigth/2)][int(width - 1)]
    dilated[dilated == wallL] = wallR
    ceilC = dilated[0][int(width/2)]
    floorC = dilated[heigth-1][int(width/2)]
    uniqueColors = np.unique(dilated)    
    walls = uniqueColors[uniqueColors != 0]
    walls = walls[walls != floorC]
    walls = walls[walls != ceilC]

    wall_info = {
        'ceil' : ceilC,
        'floor' : floorC,
        'walls' : walls
    }
    return dilated, wall_info

#Get floor, ceil and walls
def uv2xyz(uv,imW,imH):
    tetha = - (uv[:,1] - imH / 2) * np.pi / imH
    phi = (uv[:,0] - imW / 2) * 2 * np.pi / imW
    xyz = np.array([np.sin(phi) * np.cos(tetha),np.cos(tetha) * np.cos(phi),np.sin(tetha)])
    return xyz.T

def alignFloorCeil(floor_xyz, ceil_xyz, f3D, c3D):
    if len(ceil_xyz) != len(floor_xyz):
        print("error")
        return None, None
    else:
        for i in range(len(f3D)):
            temp_x = (f3D[i][0] + c3D[i][0])/2
            temp_y = (f3D[i][1] + c3D[i][1])/2
            f3D[i] = [temp_x, temp_y, f3D[i][2]]
            c3D[i] = [temp_x, temp_y, c3D[i][2]]
        return f3D, c3D

def getWalls3D(floor):
    floor = np.vstack((floor, floor[0,:]))
    walls = []
    for i in range(len(floor) - 1):
        topL = [floor[i][0], standardCeilingHeigh, floor[i][2]]
        topR = [floor[i + 1][0], standardCeilingHeigh, floor[i + 1][2]]
        walls.append([floor[i], topL, topR, floor[i + 1]])
    return np.array(walls)

def getWalls3DReversed(floor, ceil):
    if len(floor) != len(ceil):
        print("error")
        return None, None
    else:
        floor = np.vstack((floor, floor[0,:]))
        ceil = np.vstack((ceil, ceil[0,:]))
        walls = []
        for i in range(len(floor) - 1):
            walls.append([floor[i], floor[i + 1], ceil[i + 1], ceil[i]])
        return np.array(walls)

#Get polygon buffer
def bufferPolygon(poly):
    polygon_points = poly[:, [0, 2]]
    scaling_factor = 0.1
    polygon = Polygon(polygon_points)
    buffered_polygon = polygon.buffer(scaling_factor, quad_segs = 0, cap_style='flat', join_style='mitre')  
    polygon_coords = np.array(buffered_polygon.exterior.coords)[::-1][:-1]
    y_array = np.reshape(poly[:, 1], (-1, 1))
    result = np.concatenate(( np.reshape(polygon_coords[:, 0], (-1, 1)), y_array,  np.reshape(polygon_coords[:, 1], (-1, 1))), axis=1)
    return np.array(result)

def getOuterWalls(floor):
    floor_out_3D = bufferPolygon(floor)
    return getWalls3D(floor_out_3D)

def getCorrectedFloorCeilWalls(cor_uv):
    cor_xyz = uv2xyz(cor_uv,pano_W,pano_H)
    ceil_xyz = cor_xyz[cor_xyz[:,2]>0,:]
    floor_xyz = cor_xyz[cor_xyz[:,2]<0,:]
    d_floor = 1.7
    #d_floor =  standardCeilingHeigh
    t_floor = -d_floor/floor_xyz[:,2]
    floor_3D = np.expand_dims(t_floor, axis=1) * floor_xyz
    t_ceil = floor_3D[:,0]/ceil_xyz[:,0]
    ceil_3D = np.expand_dims(t_ceil, axis=1) * ceil_xyz
    d_ceil = np.mean(t_ceil*ceil_xyz[:,2])
    ceil_3D[:,2] = d_ceil
    new_floor_3D, new_ceil_3D = alignFloorCeil(floor_xyz, ceil_xyz, floor_3D, ceil_3D)
    walls = getWalls3DReversed(new_floor_3D, new_ceil_3D)
    return new_floor_3D, walls

def getValue(val, limit):
    #return float(val)
    val = float(val)
    if val < 0:
        val = 0
    elif val >= limit:
        val = limit - 1
    return val

def getBboxMask(bbox):
    image = np.zeros((pano_H, pano_W), dtype=np.uint8)
    bbox = [getValue(bbox[0], pano_H),
        getValue(bbox[1], pano_W),
        getValue(bbox[2], pano_H),
        getValue(bbox[3], pano_W)]
    rr, cc = draw.rectangle((bbox[0], bbox[1]), end=(bbox[2], bbox[3]))
    image[rr, cc] = 255
    return image

def getSpecificObjectMask(type, bbox):
    object_color = ItemsName[type]['color']
    segmentation = np.uint8(np.all(smap_rgb == object_color, axis=-1) * 255)
    mask = getBboxMask(bbox)
    result = np.logical_and(segmentation, mask)
    return result

#Get objects
def getObjects():
    path_mask = "tmp/obj_mask.jpg"
    count = 0
    for obj in boxes_data.values():
        type = obj["type"]
        bbox = obj["bbox"]
        y1 = float(bbox[0])
        x1 = float(bbox[1])
        y2 = float(bbox[2])
        x2 = float(bbox[3])
        x = math.floor((x1 + x2)/2)
        y = math.floor((y1 + y2)/2)

        theta = 360 * (x / pano_W)  - 180
        phi = 180 * (y / pano_H) - 90

        width = 1200
        height = 1200

        newMask = getSpecificObjectMask(type, bbox)
        
        #Code for testing
        image = Image.fromarray(newMask)
        image.save(path_mask)

        #savemask
        FOV = 120
        input_img = './input/test.jpg'
        output_dir = './tmp/mask/perspective' + str(count) + ".jpg"

        input_mask = './tmp/obj_mask.jpg'
        output_mask = './tmp/mask/perspective' + str(count) + "_mask.jpg"

        new_img = equir2pers.equir2pers(input_img, output_dir, FOV, theta, phi, height, width)
        new_mask = equir2pers.equir2pers(input_mask, output_mask, FOV, theta, phi, height, width)
        print(new_img)
        print(new_mask)
        count = count + 1
        #get BB from mask modified
        #Generate a .json with it
        #Apply Total3D using .json, camera and the image
        #getresults in json and read it
    '''
        Iterar tots els objectes del .json
            Per cada objecte obtenir punt mig de BB, conseguir un FOV decent i generar 
            una imatge amb la foto normal i la mask

            Obtenir una nova BB amb la mask

            Amb nova foto + BB pasar-ho pel Total 3D

            Obtenir geometria de la BB-3D

            Guardarla en el 3D file amb el format adequat, type, etc
    '''
    return []

#Get room
def getRoom(floor, walls, objects):
    room = {
        'floor' : np.array(floor),
        'walls' : getWalls3D(floor),
        'walls_out' : getOuterWalls(floor),
        'objects' : np.array(objects)
    }
    return room

#Rescale room
def getNewPoint(P, S, T, R, RX, RY):
    P = RY @ RX @ R @ S @ T @ np.append(P, 1)
    return [np.round(P[0],3),np.round(P[1],3),np.round(P[2],3)]

def rescaleRoom(floor, walls, objects):
    origin_floor = walls[0][0]
    origin_ceil = walls[0][3]
    next_floor = walls[1][0]
    T = np.array([  [1, 0, 0, -origin_floor[0]],
                    [0, 1, 0, -origin_floor[1]],
                    [0, 0, 1, -origin_floor[2]],
                    [0, 0, 0, 1]])
    scale = standardCeilingHeigh/ (origin_ceil[2] -origin_floor[2])
    # Scaling matrix
    S = np.array([[scale, 0, 0, 0],
                [0, scale, 0, 0],
                [0, 0, scale, 0],
                [0, 0, 0, 1]])
    angle = math.atan(next_floor[0]/next_floor[1])

    R = np.array([[math.cos(angle), -math.sin(angle), 0, 0],
                [math.sin(angle), math.cos(angle), 0, 0],
                [0,     0, 1, 0],
                [0,     0, 0, 1]])
    
    angle1 = -np.pi / 2
    angle2 = -np.pi / 2

    RX = np.array([[1, 0, 0, 0],
                    [0, np.cos(angle1), -np.sin(angle1), 0],
                    [0, np.sin(angle1), np.cos(angle1), 0],
                    [0,     0, 0, 1]])
    '''
    RY = np.array([[np.cos(angle2), -np.sin(angle2), 0, 0],
                    [np.sin(angle2), np.cos(angle2), 0, 0],
                    [0, 0, 1, 0],
                    [0,     0, 0, 1]])
    '''
    
    RY = np.array([[np.cos(angle2), 0, np.sin(angle2), 0],
                [0, 1, 0, 0],
                [-np.sin(angle2), 0, np.cos(angle2), 0],
                
                [0,     0, 0, 1]])

    for i in range(len(floor)):
        floor[i] = getNewPoint(floor[i], S, T, R, RX, RY)
        for j in range(4):
            walls[i][j] = getNewPoint(walls[i][j], S, T, R, RX, RY)
    return getRoom(floor, walls, objects)

def getFakeRoom():
    floorDataV2 = np.array([
        [0, 0,  0],
        [3, 0,  0],
        [3, 0,  2],
        [5, 0,  2],
        [5, 0,  4],
        [0, 0,  4]                              
    ])

    '''
    ceilDataV2 = np.array([
        [0, 2.5,  0],
        [3, 2.5,  0],
        [3, 2.5,  2],
        [5, 2.5,  2],
        [5, 2.5,  4],
        [0, 2.5,  4]                           
    ])
    
    wallsDataV2 = np.array([
        [
            [0, 0,  0],
            [0, 2.5,  0],
            [3, 2.5, 0],
            [3, 0,  0]
        ],
        [
            [3, 0,  0],
            [3, 2.5,  0],
            [3, 2.5,  2],
            [3, 0,  2]
        ],
        [
            [3, 0,  2],
            [3, 2.5,  2],
            [5, 2.5,  2],
            [5, 0,  2]
        ],
        [
            [5, 0,  2],
            [5, 2.5,  2],
            [5, 2.5,  4],
            [5, 0,  4]
        ],
        [
            [5, 0,  4],
            [5, 2.5,  4],
            [0, 2.5,  4],
            [0, 0,  4]
        ],
        [
            [0, 0,  4],
            [0, 2.5,  4],
            [0, 2.5,  0],
            [0, 0,  0]
        ]
    ])'''
    
    objectsDataV2 = np.array([
        {   #bed
            'type': 1,
            'bbox': [1, 1, 2],
            'pos': [0.75, 0, 1],
            'rot': [0,0,0,1],
            'scale': [1,1,1]
        },
        {   #table  
            'type': 3,
            'bbox': [2, 1, 1],
            'pos': [4.5, 0, 3],
            'rot': [0,-0.707,0,0.707],
            'scale': [1,1,1]
        },
        {   #window
            'type': 5,
            'bbox': [1, 1, 0.12],
            'pos': [3, 1.5, 4.05],
            'rot': [0,1,0,0],
            'scale': [1,1,1]
        },
        {   #chair
            'type': 7,
            'bbox': [0.5, 0.5, 0.5],
            'pos': [3.5, 0, 3],
            'rot': [0,-0.707,0,-0.707],
            'scale': [1,1,1]
        },
        {   #sofa
            'type': 9,
            'bbox': [2, 1, 1],
            'pos': [1.5, 0, 3.5],
            'rot': [0,1,0,0],
            'scale': [1,1,1]
        },
        {   #door
            'type': 10,
            'bbox': [1, 2, 0.12],
            'pos': [3.05, 1, 0.5],
            'rot': [0, -0.707, 0, 0.707],
            'scale': [1,1,1]
        },
        {   #cabinet
            'type': 11,
            'bbox': [1.5, 2, 0.5],
            'pos': [0.25, 0, 3],
            'rot': [0, -0.707, 0, -0.707],
            'scale': [1,1,1]
        },
        {   #bedside
            'type': 12,
            'bbox': [0.5, 0.5, 0.5],
            'pos': [2, 0, 0.25],
            'rot': [0, 0, 0, 1],
            'scale': [1,1,1]
        },
        {   #shelf
            'type': 14,
            'bbox': [1, 2, 0.5],
            'pos': [2.75, 0, 1.5],
            'rot': [0, -0.707, 0, 0.707],
            'scale': [1,1,1]
        }
    ])
    
    '''
    objectsDataV2 = np.array([
        {   #bed
            'type': 1,
            'bbox': [1, 1, 2],
            'pos': [0.75, 0, 1],
            'rot': [0,0,0,1],
            'scale': [1,1,1]
        },
        {   #table  
            'type': 3,
            'bbox': [2, 1, 1],
            'pos': [4.5, 0, 3],
            'rot': [0,-0.707,0,0.707],
            'scale': [1,1,1]
        },
        {   #window
            'type': 5,
            'pos_ini': [3.5, 1, 3.99],
            'pos_end': [2.5, 2, 3.99]
        },
        {   #chair
            'type': 7,
            'bbox': [0.5, 0.5, 0.5],
            'pos': [3.5, 0, 3],
            'rot': [0,-0.707,0,-0.707],
            'scale': [1,1,1]
        },
        {   #sofa
            'type': 9,
            'bbox': [2, 1, 1],
            'pos': [1.5, 0, 3.5],
            'rot': [0,1,0,0],
            'scale': [1,1,1]
        },
        {   #door
            'type': 10,
            'pos_ini': [2.99, 0, 0.1],
            'pos_end': [2.99, 2, 0.9]
        },
        {   #cabinet
            'type': 11,
            'bbox': [1.5, 2, 0.5],
            'pos': [0.25, 0, 3],
            'rot': [0, -0.707, 0, -0.707],
            'scale': [1,1,1]
        },
        {   #bedside
            'type': 12,
            'bbox': [0.5, 0.5, 0.5],
            'pos': [2, 0, 0.25],
            'rot': [0, 0, 0, 1],
            'scale': [1,1,1]
        },
        {   #shelf
            'type': 14,
            'bbox': [1, 2, 0.5],
            'pos': [2.75, 0, 1.5],
            'rot': [0, -0.707, 0, 0.707],
            'scale': [1,1,1]
        }
    ])'''

    RoomV2 = {
        'floor' : floorDataV2,
        'walls' : getWalls3D(floorDataV2),
        'walls_out' : getOuterWalls(floorDataV2),
        'objects' : objectsDataV2
    }
    return RoomV2

#Main code
def convert():
    detInputData()
    cor_uv = cmap2corners(cmap)
    walls_image, walls_info = getWallMap(emap)
    floor, walls = getCorrectedFloorCeilWalls(cor_uv)
    objects = getObjects()
    new_room = rescaleRoom(floor, walls, objects)
    new_room = getFakeRoom()
    return new_room