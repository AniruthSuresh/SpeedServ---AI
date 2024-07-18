def get_centre_of_box(bbox):
    """
    Computes the center of the bounding box
    """
    xmin , ymin , xmax , ymax = bbox
    centre_x = int((xmin + xmax)/2)
    centre_y = int((ymin + ymax)/2)

    return (centre_x , centre_y)

def get_euclidean_dis(p1 , p2):
    """
    Computes the euclidean distance between two points
    """
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5