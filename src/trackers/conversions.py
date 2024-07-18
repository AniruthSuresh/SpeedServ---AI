def convert_pixel_to_meter(pixel_distance , ref_height_in_meters , ref_height_in_pixels):
    """
    Takes in a pixel distance in the mini court and 
    returns the corresponding meter distance in the actual court
    """
    return (ref_height_in_meters / ref_height_in_pixels) * pixel_distance


def convert_meter_to_pixel( meters , ref_height_in_meters , ref_height_in_pixels):
    """
    Takes in meters in the actual court and return the 
    corresponding pixel distance in the mini court 
    """
    return (meters * ref_height_in_pixels)/ref_height_in_meters




