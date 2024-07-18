def get_foot_positions(bbox):
    """
    Returns the feet position of the players 
    given their bounding boxes 
    """
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
   """
   Gets the closest keypoint index to a point 
   which is further used to conversion 
   """
   closest_distance = float('inf')
   key_point_ind = keypoint_indices[0]
   for keypoint_indix in keypoint_indices:
       keypoint = keypoints[keypoint_indix*2], keypoints[keypoint_indix*2+1]
       distance = abs(point[1]-keypoint[1])

       if distance<closest_distance:
           closest_distance = distance
           key_point_ind = keypoint_indix
    
   return key_point_ind

    

def get_height_of_bbox(bbox):
    """
    Returns the height of the bounding boxes 
    """
    x1 , y1 , x2 , y2 = bbox
    return y2-y1

def measure_xy_distance (object_pos , kp):
    """
    Computes the absolute distances in the x and y directions between two points.
    """
    return abs(object_pos[0]- kp[0]) , abs(object_pos[1] - kp[1])


def get_centre_of_bbox(bbox):
    """
    Computes the centre of the bounding box
    """
    return (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))

