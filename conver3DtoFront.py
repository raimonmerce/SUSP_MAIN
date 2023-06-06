import numpy as np
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.datasets.init import filter_function

ItemsT = {
    #0:  {'color': [0,0,0]       ,'name': 'none'     , 'type' : 0 ,'name': 'none'},
    1:  {'color': [128,0,0]     ,'name': 'bed'      , 'type' : 2 ,'future': 'single_bed'}, #double_bed
    #2:  {'color': [0,128,0]     ,'name': 'painting' , 'type' : 1 ,'future': 'none'},
    3:  {'color': [128,128,0]   ,'name': 'table'    , 'type' : 3 ,'future': 'table'},
    #4:  {'color': [0,0,128]     ,'name': 'mirror'   , 'type' : 1 ,'future': 'none'},
    5:  {'color': [128,0,128]   ,'name': 'window'   , 'type' : 1 ,'future': 'none'},
    #6:  {'color': [0,128,128]   ,'name': 'curtain'  , 'type' : 1 ,'future': 'none'},
    7:  {'color': [128,128,128] ,'name': 'chair'    , 'type' : 3 ,'future': 'chair'},
    #8:  {'color': [64,0,0]      ,'name': 'light'    , 'type' : 4 ,'future': 'none'},
    9:  {'color': [192,0,0]     ,'name': 'sofa'     , 'type' : 3 ,'future': 'sofa'},
    10: {'color': [64,128,0]    ,'name': 'door'     , 'type' : 1 ,'future': 'none'},
    11: {'color': [192,128,0]   ,'name': 'cabinet'  , 'type' : 2 ,'future': 'cabinet'},
    12: {'color': [64,0,128]    ,'name': 'bedside'  , 'type' : 3 ,'future': 'nightstand'},
    #13: {'color': [192,0,128]   ,'name': 'tv'       , 'type' : 4 ,'future': 'none'},
    14: {'color': [64,128,128]  ,'name': 'shelf'    , 'type' : 2 ,'future': 'shelf'}   
}

countId = 1

def getDataset():
    path_to_3d_front_dataset_directory = '../../../../../media/raimon/SSD/3Dfront/3D-FRONT'
    path_to_model_info = '../../../../../media/raimon/SSD/3Dfront/3D-FUTURE-model/model_info.json'
    path_to_3d_future_dataset_directory = '../../../../../media/raimon/SSD/3Dfront/3D-FUTURE-model'
    path_to_invalid_scene_ids = "config/invalid_threed_front_rooms.txt"
    path_to_invalid_bbox_jids = "config/black_list.txt"
    path_to_pickled_3d_futute_models = "ATISS_extra/pickle/threed_future_model_bedroom.pkl"
    annotation_file = "config/bedroom_threed_front_splits.csv"

    '''
    dataset_furniture = ThreedFutureDataset.from_dataset_directory(
    dataset_directory=path_to_3d_front_dataset_directory,
    path_to_model_info=path_to_model_info,
    path_to_models=path_to_3d_future_dataset_directory)
    return dataset_furniture
    '''

    dataset_furniture_pickle = ThreedFutureDataset.from_pickled_dataset(
    path_to_pickled_dataset=path_to_pickled_3d_futute_models)
    return dataset_furniture_pickle



def get3DFrontTemplate():
    return {
        'uid' : "0a8d471a-2587-458a-9214-586e003e9cf9",
        'design_version' : "0.1",
        'code_version' : "0.4",
        'north_vector' : [0, 1, 0],
        'furniture' : [],
        'mesh' : [],
        'material' : [
        {
            "uid": "sge/2592cffe-9599-404e-9703-4f833fe1d20c/2740",
            "aid": [], 
            "jid": "57016195-8cd6-45f5-a106-89caaebf4f3c", 
            "texture": "", 
            "normaltexture": "", 
            "color": [255, 255, 255, 255], 
            "seamWidth": 0, 
            "useColor": True, 
            "normalUVTransform": [1, 0, 0, 0, 1, 0, 0, 0, 1], 
            "contentType": ["material", "paint"]}

        ],
        'extension' : {
            'door' : [],
            'outdoor' : "",
            'pano' : {},
            'mini_map' : "",
            'perspective_view' : {
                'link' : []
            },
            'area' : []
        },
        'scene' : {
            'ref' : "-1",
            'pos' : [0,0,0],
            'rot' : [0,0,0,1],
            'scale' : [1,1,1],
            'room' : [
                {
                    'type' : "Bedroom",
                    'instanceid' : "generate",
                    'pos' : [0,0,0],
                    'rot' : [0,0,0,1],
                    'scale' : [1,1,1],
                    'children' : [],
                    'empty' : 0
                }
            ],
        },
        'groups' : [],
        'materialList' : [
            '57016195-8cd6-45f5-a106-89caaebf4f3c'
        ],    
        'version' : "2.0"
    }

def is_point_in_triangle(point, vertex1, vertex2, vertex3):
    # Check if a point is inside the triangle formed by three vertices
    area = 0.5 * (-vertex2[2] * vertex3[0] + vertex1[2] * (-vertex2[0] + vertex3[0]) + vertex1[0] * (vertex2[2] - vertex3[2]) + vertex2[0] * vertex3[2])
    #print(area)
    s = 1 / (2 * area) * (vertex1[2] * vertex3[0] - vertex1[0] * vertex3[2] + (vertex3[2] - vertex1[2]) * point[0] + (vertex1[0] - vertex3[0]) * point[2])
    t = 1 / (2 * area) * (vertex1[0] * vertex2[2] - vertex1[2] * vertex2[0] + (vertex1[2] - vertex2[2]) * point[0] + (vertex2[0] - vertex1[0]) * point[2])
    return s > 0 and t > 0 and 1 - s - t > 0

def is_ear(vertex1, vertex2, vertex3, polygon):
    # Check if the vertex is an ear
    for i in range(len(polygon)):
        if polygon[i] not in [vertex1, vertex2, vertex3]:
            if is_point_in_triangle(polygon[i], vertex1, vertex2, vertex3):
                return False
    return True

def ear_clip_triangulation(polygon):
    # Perform ear clipping triangulation on the given polygon
    triangles = []
    remaining_vertices = polygon.copy()

    while len(remaining_vertices) >= 3:
        for i in range(len(remaining_vertices)):
            vertex1 = remaining_vertices[i - 1]
            vertex2 = remaining_vertices[i]
            vertex3 = remaining_vertices[(i + 1) % len(remaining_vertices)]

            if is_ear(vertex1, vertex2, vertex3, remaining_vertices):
                triangles.append([vertex1, vertex2, vertex3])
                remaining_vertices.remove(vertex2)
                break

    return triangles
    
def getMeshFC(points, type):
    #0 = floor 1 = ceil
    floorId = 0
    mesh = {}
    #mesh['uid'] = str(countId) + "/0"
    mesh['aid'] = []
    mesh['jid'] = ''
    mesh['xyz'] = []
    mesh['normal'] = [] 
    mesh['uv'] = []
    mesh['faces'] = []
    mesh['material'] = "sge/2592cffe-9599-404e-9703-4f833fe1d20c/2740"
    mesh['type'] = "Floor" #if type == 1 else "Ceiling"
    #countId += 1

    triangles = ear_clip_triangulation(points)

    normalY = 0 if type == 0 else 2.4 

    for tri in triangles:
        for p in tri:
            mesh['xyz'].append(p[0])
            mesh['xyz'].append(normalY)
            mesh['xyz'].append(p[2])
            mesh['normal'].append(1)
            mesh['normal'].append(1)
            mesh['normal'].append(1)
            mesh['uv'].append(p[0])
            mesh['uv'].append(p[2])
            mesh['faces'].append(floorId)
            floorId = floorId + 1

    if type == 0:
        for i in range(int(len(mesh['faces'])/3)):
            tmp = mesh['faces'][i*3 + 1]
            mesh['faces'][i*3 + 1] = mesh['faces'][i*3 + 2]
            mesh['faces'][i*3 + 2] = tmp

    return mesh

def getNormal(triangle):
    vertex1 = np.array(triangle[0])
    vertex2 = np.array(triangle[1])
    vertex3 = np.array(triangle[2])

    # Calculate two edges of the triangle
    edge1 = vertex2 - vertex1
    edge2 = vertex3 - vertex1

    # Compute the cross product of the edges
    normal = np.cross(edge1, edge2)

    # Normalize the normal vector (optional)
    normal = normal / np.linalg.norm(normal)
    return normal

def getMeshWall(wall, type):
    mesh = {}
    mesh['aid'] = []
    mesh['jid'] = ''
    mesh['xyz'] = [ wall[0][0], wall[0][1], wall[0][2], 
        wall[2][0], wall[2][1], wall[2][2],
        wall[1][0], wall[1][1], wall[1][2], 
        wall[0][0], wall[0][1], wall[0][2], 
        wall[3][0], wall[3][1], wall[3][2],
        wall[2][0], wall[2][1], wall[2][2]
    ]

    mesh['normal'] = [
        0, 1, 0, 
        0, 1, 0,
        0, 1, 0, 
        0, 1, 0,
        0, 1, 0,
        0, 1, 0
    ] 
    mesh['uv'] = [  
        0.0, 0.0, 1.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 0.0, 1.0, 1.0
    ]
    if type:
        mesh['faces'] = [0, 1, 2, 3, 4, 5]
        mesh['type'] = "WallInner"
    else:
        mesh['faces'] = [0, 2, 1, 3, 5, 4]
        mesh['type'] = "WallOuter"
    
    mesh['material'] = "sge/2592cffe-9599-404e-9703-4f833fe1d20c/2740"

    return mesh

def getMeshTopWall(wall_in, wall_out):
    mesh = {}
    mesh['aid'] = []
    mesh['jid'] = ''
    mesh['xyz'] = [ wall_in[1][0], wall_in[1][1], wall_in[1][2], 
        wall_out[2][0], wall_out[2][1], wall_out[2][2],
        wall_out[1][0], wall_out[1][1], wall_out[1][2], 
        wall_in[1][0], wall_in[1][1], wall_in[1][2], 
        wall_in[2][0], wall_in[2][1], wall_in[2][2],
        wall_out[2][0], wall_out[2][1], wall_out[2][2]
    ]

    mesh['normal'] = [
        1, 1, 1, 
        1, 1, 1,
        1, 1, 1, 
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    ] 
    
    mesh['uv'] = [  
        0.0, 0.0, 1.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 0.0, 1.0, 1.0
    ]

    #mesh['faces'] = [0, 2, 1, 3, 5, 4]
    mesh['faces'] = [0, 1, 2, 3, 4, 5]
    mesh['material'] = "sge/2592cffe-9599-404e-9703-4f833fe1d20c/2740"
    mesh['type'] = "WallTop"
    return mesh

def getMeshBottomWall(wall_in, wall_out):
    mesh = {}
    mesh['aid'] = []
    mesh['jid'] = ''
    mesh['xyz'] = [ wall_in[0][0], wall_in[0][1], wall_in[0][2], 
        wall_out[3][0], wall_out[3][1], wall_out[3][2],
        wall_out[0][0], wall_out[0][1], wall_out[0][2], 
        wall_in[0][0], wall_in[0][1], wall_in[0][2], 
        wall_in[3][0], wall_in[3][1], wall_in[3][2],
        wall_out[3][0], wall_out[3][1], wall_out[3][2]
    ]

    mesh['normal'] = [
        1, 1, 1, 
        1, 1, 1,
        1, 1, 1, 
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    ] 
    
    mesh['uv'] = [  
        0.0, 0.0, 1.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 0.0, 1.0, 1.0
    ]

    mesh['faces'] = [0, 2, 1, 3, 5, 4]
    #mesh['faces'] = [0, 1, 2, 3, 4, 5]
    mesh['material'] = "sge/2592cffe-9599-404e-9703-4f833fe1d20c/2740"
    mesh['type'] = "WallBottom"
    return mesh

def getNewPoint(P, S, T, R):
    P =  S @ T @ R @ np.append(P, 1)
    return [np.round(P[0],3),np.round(P[1],3),np.round(P[2],3)]

def getSimpleMesh(p0, p1, p2, p3):
    mesh = {}
    mesh['aid'] = []
    mesh['jid'] = ''
    mesh['xyz'] = [ 
        p0[0], p0[1], p0[2], 
        p2[0], p2[1], p2[2],
        p1[0], p1[1], p1[2], 
        p0[0], p0[1], p0[2], 
        p3[0], p3[1], p3[2],
        p2[0], p2[1], p2[2]
    ]
    mesh['normal'] = [
        0.5, 0, 0.5, 
        0.5, 0, 0.5,
        0.5, 0, 0.5, 
        0.5, 0, 0.5,
        0.5, 0, 0.5,
        0.5, 0, 0.5
    ] 
    mesh['uv'] = [  
        0.0, 0.0, 1.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 0.0, 1.0, 1.0
    ]
    mesh['faces'] = [0, 1, 2, 3, 4, 5]
    mesh['material'] = "sge/2592cffe-9599-404e-9703-4f833fe1d20c/2740"
    mesh['type'] = "Window" #if type == 1 else "Ceiling"
    return mesh

def getInWallMesh(obj):
    bbox = obj['bbox']
    pos = obj['pos']
    rot = obj['rot']
    scale = obj['scale']

    T = np.array([  [1, 0, 0, pos[0]],
                    [0, 1, 0, pos[1]],
                    [0, 0, 1, pos[2]],
                    [0, 0, 0, 1]])
    # Scaling matrix
    S = np.array([[scale[0], 0, 0, 0],
                    [0, scale[1], 0, 0],
                    [0, 0, scale[2], 0],
                    [0, 0, 0, 1]])
    w = rot[0]
    x = rot[1]
    y = rot[2]
    z = rot[3]
    
    xx = 1 - 2 * (y ** 2) - 2 * (z ** 2)
    xy = 2 * x * y - 2 * w * z
    xz = 2 * x * z + 2 * w * y

    yx = 2 * x * y + 2 * w * z
    yy = 1 - 2 * (x ** 2) - 2 * (z ** 2)
    yz = 2 * y * z - 2 * w * x

    zx = 2 * x * z - 2 * w * y
    zy = 2 * y * z + 2 * w * x
    zz = 1 - 2 * (x ** 2) - 2 * (y ** 2)

    R = np.array([[xx, xy, xz, 0],
                [yx, yy, yz, 0],
                [zx, zy, zz, 0],
                [0,  0, 0, 1]])

    pos_ini = [bbox[0]/2, bbox[1]/2, bbox[2]/2]
    pos_end = [-bbox[0]/2, -bbox[1]/2, -bbox[2]/2]

    #

    pos_0 = [pos_end[0], pos_end[1], pos_ini[2]]
    pos_1 = [pos_end[0], pos_ini[1], pos_ini[2]]
    pos_2 = pos_ini
    pos_3 = [pos_ini[0], pos_end[1], pos_ini[2]]

    p0 = getNewPoint(pos_0, S, T, R)
    p1 = getNewPoint(pos_1, S, T, R)
    p2 = getNewPoint(pos_2, S, T, R)
    p3 = getNewPoint(pos_3, S, T, R)

    meshFront = getSimpleMesh(p0, p1, p2, p3)

    pos_0 = [pos_end[0], pos_ini[1], pos_end[2]]
    pos_1 = pos_end
    pos_2 = [pos_ini[0], pos_end[1], pos_end[2]]
    pos_3 = [pos_ini[0], pos_ini[1], pos_end[2]]

    p0 = getNewPoint(pos_0, S, T, R)
    p1 = getNewPoint(pos_1, S, T, R)
    p2 = getNewPoint(pos_2, S, T, R)
    p3 = getNewPoint(pos_3, S, T, R)

    meshBack = getSimpleMesh(p0, p1, p2, p3)

    return meshFront, meshBack

#Main code
def convert(room, style):
    count = 1
    print(room)
    room3dfront = get3DFrontTemplate()
    dataset_furniture = getDataset()
    objects = room['objects']
    children = []
    furniture = []
    mesh = []
    #Get furniture
    for obj in objects:
        if obj['type'] in ItemsT:
            name = ItemsT[obj['type']]["name"]
            if name == "door" or name == 'window':
                front, back = getInWallMesh(obj)
                mesh.append(front)
                mesh.append(back)
            else:
                category = ItemsT[obj['type']]["future"]
                if category:
                    bbox = obj['bbox']
                    future_object = dataset_furniture.get_closest_furniture_to_box_style(category, bbox, style)
                    if future_object == None:
                        future_object = dataset_furniture.get_closest_furniture_to_box(category, bbox)
                    if future_object:
                        child = {}
                        child['ref'] = future_object.model_uid
                        child['instanceid'] = "furniture/" + str(count)
                        child['pos'] = obj['pos']
                        child['rot'] = obj['rot']
                        child['scale'] = obj['scale']
                        children.append(child)
                        f = {}
                        f['uid'] = future_object.model_uid
                        f['jid'] = future_object.model_jid
                        f['category'] = future_object.model_info.category
                        f['aid'] = []
                        f['size'] = True
                        f['bbox'] = obj['bbox']
                        f['title'] = ''
                        f['sourceCategoryId'] = "14d0d6c8-6857-4bbf-8c4a-26bf6c25c968"
                        f['valid'] = True
                        furniture.append(f)
                        count += 1
    room3dfront["furniture"] = furniture 

    countId = 1
    #Get meshes
    mesh.append(getMeshFC(room['floor'].tolist(), 0)) #Floor
    mesh.append(getMeshFC(room['floor'].tolist(), 1)) #Ceiling
    #for wall in room['walls']:
    for i in range(len(room['walls'])):
        wall_in = room['walls'][i]
        wall_out = room['walls_out'][i]
        mesh.append(getMeshWall(wall_in, True))
        mesh.append(getMeshWall(wall_out, False))
        mesh.append(getMeshTopWall(wall_in, wall_out))
        mesh.append(getMeshBottomWall(wall_in, wall_out))
    for msh in mesh:
        tmp = str(countId) + "/0"
        my_dict = {'uid': tmp}
        #msh['uid'] = 1
        msh.update(my_dict)
        child = {}
        child['ref'] = msh["uid"]
        child['pos'] = [0,0,0]
        child['rot'] = [0,0,0,1]
        child['scale'] = [1,1,1]
        child['instanceid'] = "mesh/" + str(count)
        children.append(child)
        count += 1
        countId += 1
    room3dfront["scene"]["room"][0]["children"] = children 
    room3dfront["mesh"] = mesh
    return room3dfront