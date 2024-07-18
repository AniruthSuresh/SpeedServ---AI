import cv2
import sys
sys.path.append('../')
import os
from utils import *
from trackers import (convert_pixel_to_meter , 
                      convert_meter_to_pixel)

from court_constants.constants import *
import numpy as np


class MiniCourtSketch():

    def __init__(self , frame):

        self.drawing_rect_width = 250
        self.drawing_rect_height = 550
        self.buffer = 50
        self.padding_inside_court = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()
    
    def set_canvas_background_box_position(self , frame):
        """

        Sets the mini court external positions 
        """
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.start_x = self.end_x - self.drawing_rect_width

        self.start_y = self.buffer
        self.end_y = self.start_y + self.drawing_rect_height 

    def set_mini_court_position(self):
        """
        Sets the mini court internal dimension (external - buffer)
        """

        self.court_start_x = self.start_x + self.padding_inside_court
        self.court_end_x = self.end_x - self.padding_inside_court

        self.court_start_y = self.start_y + self.padding_inside_court
        self.court_end_y = self.end_y - self.padding_inside_court

        self.court_width  = self.court_end_x - self.court_start_x
        self.court_height = self.court_end_y - self.court_start_y

    def convert_meters_to_pixels(self, meters):
        """

        Converts the mts to pixels using a standard reference of 
        half court length in pixels and meters 
        """
        return convert_meter_to_pixel(meters,
                                        HALF_COURT_LINE_HEIGHT,
                                        self.drawing_rect_width
                                        )

    def set_court_lines(self):
        """
        Sets the endpoints of various court lines 
        for plotting in mini court        
        """
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),
            (8,9),
            (10,11),
            (10,11),
            (2,3)
        ]

    def set_court_drawing_key_points(self):
        """
        This function sets the keypoints in the minicourt based
        on its actual detection in the main court 
        """
        drawing_key_points = [0]*28

        # point 0 
        drawing_key_points[0] , drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        
        # point 1
        drawing_key_points[2] , drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(HALF_COURT_LINE_HEIGHT*2)

        # point 3
        drawing_key_points[6] = drawing_key_points[2]
        drawing_key_points[7] = drawing_key_points[5] 

        # point 4
        drawing_key_points[8] = drawing_key_points[0] +  self.convert_meters_to_pixels(DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 

        # point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5] 

        # point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 

        # point 7
        drawing_key_points[14] = drawing_key_points[12]
        drawing_key_points[15] = drawing_key_points[7] 

        # point 8
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(NO_MANS_LAND_HEIGHT)

        # point 9
        drawing_key_points[18] = drawing_key_points[12]
        drawing_key_points[19] = drawing_key_points[17] 

        # point 10
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(NO_MANS_LAND_HEIGHT)

        # point 11
        drawing_key_points[22] = drawing_key_points[12]
        drawing_key_points[23] = drawing_key_points[21] 

        # point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 

        # point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21] 

        self.drawing_key_points =drawing_key_points

    def draw_background_court(self, frame):
        """
        Draws the minicourt on top of the video frame with 
        transparency . 
        """
        if not isinstance(frame, np.ndarray):
            raise ValueError("Input frame must be a numpy array.")

        # Create a transparent image of the same size as the frame, initialized to zeros (black)
        shape = np.zeros_like(frame, np.uint8)  # makes it transparent

        # Draw a filled white rectangle on the 'shape' image
        cv2.rectangle(shape, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)  # fills the rectangle with white color

        # Make a copy of the original frame
        out = frame.copy()
        alpha = 0.5  # Alpha blending factor

        # Create a mask where the white rectangle is
        mask = shape[:, :, 0] > 0

        # Blend the rectangle with the original frame using the mask
        blended = cv2.addWeighted(frame, alpha, shape, 1 - alpha, 0)

        # Apply the blended result where the mask is True
        out[mask] = blended[mask]

        return out

    def draw_keypoints_on_court (self , frame):

        """
        Scatters and plots the keypoints on the mini court
        which was detected earlier 
        """
        for i in range( 0 , len(self.drawing_key_points) , 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(frame , (x,y) , 5 , (0 , 0 , 255) , -1)

        return frame

    def draw_lines_in_court(self , frame):
        """
        Sketches the court lines which was given by the 
        set_court_lines function 
        
        """
        for line in self.lines:

            # (0 ,2) -> draw from point 0 (x1, y1) to point 2 (x2,y2)
            start_point = (int(self.drawing_key_points[line[0]*2]) , int(self.drawing_key_points[line[0]*2+1])) #(x1,y1)
            end_point = (int(self.drawing_key_points[line[1]*2]) , int(self.drawing_key_points[line[1]*2+1])) # (x2,y2)

            cv2.line(frame , start_point , end_point , (0,0,0) , 2)
        return frame

    def draw_net_in_court(self , frame):
        """
        This function draws a net in the mini court 
        based on its mid point position for visualisation 
        """
        # Draw net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (139, 0, 0), 2)


        return frame

    def draw_mini_court_all_frames(self , frames):
        """
        Stitches all the frames with the mini court and 
        return the final total frames which can be converted to video 
        """
        output_frames = []

        for frame in frames:
            frame = self.draw_background_court(frame)
            frame = self.draw_keypoints_on_court(frame)
            frame = self.draw_lines_in_court(frame)
            frame = self.draw_net_in_court(frame)
            
            output_frames.append(frame)

        return output_frames
            
    def get_start_point_of_mini_court(self):
        """
        Returns the start position of the minicourt 
        """
        return (self.court_start_x , self.court_start_y)

    def width_of_mini_court(self):
        """
        Returns the width of the mini-court
        """
        return self.drawing_rect_width

    def get_court_keypoints(self):
        """
        Returns the keypoint positon on the minicourt 
        """
        return self.drawing_key_points
    
    def get_mini_court_player_position(self , 
                                object_pos , 
                                closest_keypoint ,
                                closest_kp_index , 
                                player_height_in_pixels ,
                                player_height_in_meter):
        """
        Returns the current position of the players 
        on the court and maps it to mini court position 
        to depict and visualise later 
        """
        
        # dis between the closest keypoint and the foot 
        distance_x_in_pixels , distance_y_in_pixels = measure_xy_distance(object_pos , closest_keypoint)

        # convert this to meters (this is the distance the player has moved)
        distance_from_kp_x_meters = convert_pixel_to_meter(pixel_distance=distance_x_in_pixels , ref_height_in_meters=player_height_in_meter , 
                                                           ref_height_in_pixels= player_height_in_pixels)
        

        distance_from_kp_y_meters = convert_pixel_to_meter(pixel_distance=distance_y_in_pixels , ref_height_in_meters=player_height_in_meter , 
                                                    ref_height_in_pixels= player_height_in_pixels)
        
        # convert to mini court coordinates
        mini_court_x_pixel_drift = self.convert_meters_to_pixels(distance_from_kp_x_meters)
        mini_court_y_pixel_drift = self.convert_meters_to_pixels(distance_from_kp_y_meters)

        closest_mini_court_kp = (
            self.drawing_key_points[closest_kp_index*2], 
            self.drawing_key_points[closest_kp_index*2+1]
        )

        mini_court_player_position = (
            closest_mini_court_kp[0] + mini_court_x_pixel_drift , 
            closest_mini_court_kp[1] + mini_court_y_pixel_drift , 
        )

        return mini_court_player_position

    def map_player_and_ball_from_actual_to_mini_court(self  , player_boxes , ball_boxes , court_keypoints):
        """
        Maps the current player and ball positions and returns them 
        so they can be depicted on the minicourt 
        """
        player_height = {
            1: PLAYER_1_HEIGHT,
            2:PLAYER_2_HEIGHT
        }

        output_player_boxes=[]
        output_ball_boxes = []

        for frame_id , players_box in enumerate(player_boxes): # list of dict 
            
            output_player_bboxes_dict = {}

            ball_box = ball_boxes[frame_id][1]
            ball_centre = get_centre_of_bbox(ball_box)

            closest_player_id_to_ball = min(players_box.keys(), key=lambda x: get_euclidean_dis(ball_centre, get_centre_of_bbox(players_box[x])))

            
            for key , bbox in players_box.items(): # key is the player ID 
                foot_position = get_foot_positions(bbox)

                # get the closest keypoints in pixels -> search only a few keypoitd = [0 , 2, 12 , 13]
                closest_index = get_closest_keypoint_index(foot_position , court_keypoints , [0 , 2 , 12 , 13])
                closest_kp = (court_keypoints[closest_index*2] , court_keypoints[closest_index*2 +1])

                # player height in pixel -> we use the max height of the player in a window frame
                frame_index_min = max(0 , frame_id-20)
                frame_index_max = min(len(player_boxes) , frame_id+50)

                max_player_height = float('-Inf')

                bboxes_heights_in_pixels = [get_height_of_bbox(player_boxes[i][key]) for i in range (frame_index_min,frame_index_max)]
                max_player_height = max(bboxes_heights_in_pixels)
                
                mini_court_player_pos = self.get_mini_court_player_position(foot_position , 
                                                                            closest_kp ,
                                                                            closest_index, 
                                                                            max_player_height, 
                                                                            player_height[key],
                                                                            )
                
                output_player_bboxes_dict[key] = mini_court_player_pos

                if closest_player_id_to_ball == key:
                    closest_index = get_closest_keypoint_index(ball_centre , court_keypoints , [0 , 2 , 12 , 13])
                    closest_kp = (court_keypoints[closest_index*2] , court_keypoints[closest_index*2 +1])

                    mini_court_ball_pos = self.get_mini_court_player_position(ball_centre , 
                                                            closest_kp ,
                                                            closest_index, 
                                                            max_player_height, 
                                                            player_height[key],
                                                            )
                    
                    output_ball_boxes.append({1:mini_court_ball_pos}) # only 1 ball 
                
            output_player_boxes.append(output_player_bboxes_dict)

        return output_player_boxes , output_ball_boxes
                
    def draw_points_on_mini_court(self ,frames , positions , color= (0, 255 ,0) ):
        """
        This function draws points on each frame and 
        stitches multiple frames to form the final frame video 
        """
        for frame_num , frame in enumerate(frames):
            for _, position in positions[frame_num].items():
                x , y = position
                x = int(x)
                y = int(y)

                cv2.circle(frame , (x,y) , 5 , color , -1)

        return frames    


