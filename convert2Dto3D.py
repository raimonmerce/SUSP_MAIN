import numpy as np
import cv2
import json

from PIL import Image
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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


standardCeilingHeigh = 2.4

def test():
    print("Works")

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
    smap = Image.open("tmp/test_segmentation_raw.png")
    smap = np.array(smap)
    
    # load json bbox
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

def alignFloorCeil(ceil_xyz, floor_xyz, f3D, c3D):
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

def getWalls(floor, ceil):
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

def getCorrectedFloorCeilWalls(cor_uv):
    cor_xyz = uv2xyz(cor_uv,pano_W,pano_H)
    ceil_xyz = cor_xyz[cor_xyz[:,2]>0,:]
    floor_xyz = cor_xyz[cor_xyz[:,2]<0,:]
    d_floor = 1.7
    t_floor = -d_floor/floor_xyz[:,2]
    floor_3D = np.expand_dims(t_floor, axis=1) * floor_xyz
    t_ceil = floor_3D[:,0]/ceil_xyz[:,0]
    ceil_3D = np.expand_dims(t_ceil, axis=1) * ceil_xyz
    d_ceil = np.mean(t_ceil*ceil_xyz[:,2])
    ceil_3D[:,2] = d_ceil
    new_floor_3D, new_ceil_3D = alignFloorCeil(ceil_xyz, floor_xyz, floor_3D, ceil_3D)
    walls = getWalls(new_floor_3D, new_ceil_3D)
    return new_floor_3D, new_ceil_3D, walls

#Get objects
def getObjects():
    return []

#Get room
def getRoom(floor, ceil, walls, objects):
    room = {
        'ceil' : np.array(floor),
        'floor' : np.array(ceil),
        'walls' : np.array(walls),
        'objects' : np.array(objects)
    }
    return room

#Rescale room
def getNewPoint(P, S, T, R, RX, RY):
    P = RY @ RX @ R @ S @ T @ np.append(P, 1)
    return [np.round(P[0],3),np.round(P[1],3),np.round(P[2],3)]

def rescaleRoom(room):
    floor = room["floor"]
    ceil = room["ceil"]
    walls = room["walls"]
    objects = room["objects"]
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
        ceil[i] = getNewPoint(ceil[i], S, T, R, RX, RY)
        for j in range(4):
            walls[i][j] = getNewPoint(walls[i][j], S, T, R, RX, RY)
    return getRoom(floor, ceil, walls, objects)

def getFakeRoom():
    ceilDataV2 = np.array([
        [0, 0,  0],
        [3, 0,  0],
        [3, 0,  2],
        [5, 0,  2],
        [5, 0,  4],
        [0, 0,  4]                              
    ])


    floorDataV2 = np.array([
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
    ])

    objectsDataV2 = np.array([
        {
            'type': 1,
            'bbox': [1, 1, 2],
            'pos': [0.5, 0, 1],
            'rot': [0,0,0,1],
            'scale': [1,1,1]
        },
        
            {
            'type': 3,
            'bbox': [2, 1, 1],
            'pos': [4.5, 0, 3],
            'rot': [0,-0.707,0,0.707],
            'scale': [1,1,1]
        },
            {
            'type': 5,
            'bbox': [1, 1, 0.2],
            'pos': [3.5, 1, 4.1],
            'rot': [0,1,0,0],
            'scale': [1,1,1]
        },
            {
            'type': 7,
            'bbox': [1, 1, 1],
            'pos': [4.5, 0, 4],
            'rot': [0,-0.707,0,-0.707],
            'scale': [1,1,1]
        },
            {
            'type': 9,
            'bbox': [2, 1, 1],
            'pos': [1.5, 0, 3.5],
            'rot': [0,1,0,0],
            'scale': [1,1,1]
        },
            {
            'type': 10,
            'bbox': [1, 2, 0.2],
            'pos': [3.1, 0, 0.5],
            'rot': [0, -0.707, 0, 0.707],
            'scale': [1,1,1]
        },
            {
            'type': 11,
            'bbox': [1.5, 2, 0.5],
            'pos': [0.25, 0, 3],
            'rot': [0, -0.707, 0, -0.707],
            'scale': [1,1,1]
        },
            {
            'type': 12,
            'bbox': [0.5, 0.5, 0.5],
            'pos': [1.25, 0, 0.25],
            'rot': [0, 0, 0, 1],
            'scale': [1,1,1]
        },
            {
            'type': 14,
            'bbox': [1, 2, 0.5],
            'pos': [2.75, 0, 1.5],
            'rot': [0, -0.707, 0, 0.707],
            'scale': [1,1,1]
        }
    ])

    RoomV2 = {
        'ceil' : ceilDataV2,
        'floor' : floorDataV2,
        'walls' : wallsDataV2,
        'objects' : objectsDataV2
    }
    return RoomV2

#Main code
def convert():
    detInputData()
    cor_uv = cmap2corners(cmap)
    walls_image, walls_info = getWallMap(emap)
    floor, ceil, walls = getCorrectedFloorCeilWalls(cor_uv)
    objects = getObjects()
    room = getRoom(floor, ceil, walls, objects)
    new_room = rescaleRoom(room)
    new_room = getFakeRoom()
    return new_room