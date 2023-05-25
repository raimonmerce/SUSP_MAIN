import numpy as np
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.datasets.init import filter_function

ItemsT = {
    #0:  {'color': [0,0,0]       ,'name': 'none'     , 'type' : 0 ,'name': 'none'},
    1:  {'color': [128,0,0]     ,'name': 'bed'      , 'type' : 2 ,'future': 'single_bed'}, #double_bed
    #2:  {'color': [0,128,0]     ,'name': 'painting' , 'type' : 1 ,'future': 'none'},
    3:  {'color': [128,128,0]   ,'name': 'table'    , 'type' : 3 ,'future': 'table'},
    #4:  {'color': [0,0,128]     ,'name': 'mirror'   , 'type' : 1 ,'future': 'none'},
    ###5:  {'color': [128,0,128]   ,'name': 'window'   , 'type' : 1 ,'future': 'none'},
    #6:  {'color': [0,128,128]   ,'name': 'curtain'  , 'type' : 1 ,'future': 'none'},
    7:  {'color': [128,128,128] ,'name': 'chair'    , 'type' : 3 ,'future': 'chair'},
    #8:  {'color': [64,0,0]      ,'name': 'light'    , 'type' : 4 ,'future': 'none'},
    9:  {'color': [192,0,0]     ,'name': 'sofa'     , 'type' : 3 ,'future': 'sofa'},
    ###10: {'color': [64,128,0]    ,'name': 'door'     , 'type' : 1 ,'future': 'none'},
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
                    #'instanceid' : "LivingDiningRoom-4017",
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
    #print(point)
    #print(vertex1)
    #print(vertex2)
    #print(vertex3)
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
    mesh['type'] = "Floor" if type == 0 else "Ceiling"
    #countId += 1

    triangles = ear_clip_triangulation(points)

    normalY = 1 if type == 0 else -1 

    for tri in triangles:
        for p in tri:
            mesh['xyz'].append(p[0])
            mesh['xyz'].append(p[1])
            mesh['xyz'].append(p[2])
            mesh['normal'].append(0)
            mesh['normal'].append(normalY)
            mesh['normal'].append(0)
            mesh['uv'].append(p[0])
            mesh['uv'].append(p[2])
            mesh['faces'].append(floorId)
        ++floorId
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

def getMeshWall(wall):
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
    norm1 = getNormal([wall[0], wall[2], wall[1]])
    
    mesh['normal'] = [
        norm1[0], norm1[1], norm1[2], 
        norm1[0], norm1[1], norm1[2],
        norm1[0], norm1[1], norm1[2], 
        norm1[0], norm1[1], norm1[2], 
        norm1[0], norm1[1], norm1[2],
        norm1[0], norm1[1], norm1[2]
    ] 
    
    mesh['uv'] = [  
        0.0, 0.0, 1.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 0.0, 1.0, 1.0
    ]
    
    mesh['faces'] = [0, 0, 0, 1, 1, 1]
    mesh['material'] = "sge/2592cffe-9599-404e-9703-4f833fe1d20c/2740"
    mesh['type'] = "WallInner"
    return mesh

#Main code
def convert(room, style):
    count = 1
    print(room)
    room3dfront = get3DFrontTemplate()
    dataset_furniture = getDataset()
    objects = room['objects']
    children = []
    furniture = []
    
    #Get furniture
    for obj in objects:
        if obj['type'] in ItemsT:
            category = ItemsT[obj['type']]["future"]
            if category:
                bbox = obj['bbox']
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
    
    #global countId
    countId = 1
    #Get meshes
    mesh = []
    mesh.append(getMeshFC(room['floor'].tolist(), 0))
    mesh.append(getMeshFC(room['ceil'].tolist(), 1))
    for wall in room['walls']:
        mesh.append(getMeshWall(wall))
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