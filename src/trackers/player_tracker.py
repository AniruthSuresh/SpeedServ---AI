from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append('../') # go one folder back
from utils import (get_centre_of_box , 
                   get_euclidean_dis)

class PlayerTracker:
    def __init__(self , model_path):
        self.model = YOLO(model_path)

    def choose_two_player(self , court_keypoints , player_detections):
        """
        Filters and returns the positions of only the two players 
        which are actually playing . 
        """
        player_detections_first_frame = player_detections[0]
        chosen_player = self.chosen_player(court_keypoints , player_detections_first_frame)
        
        # now filter the frame so that only the chosen players are present 

        filtered_player_detect_total = []
        for player in player_detections:

            # store only the chosen players 
            filtered_player_detect = {track_id : bbox for track_id , bbox in player.items() if track_id in chosen_player}
            filtered_player_detect_total.append(filtered_player_detect)

        return filtered_player_detect_total

    def chosen_player(self , court_keypoints , player_dict):
        """
        Returns the 2 closest player to list of all keypoints 
        among all the players detected by the model 
        """
        distances = []

        for track_id , bbox in player_dict.items():
        # player_dict consists like xmin , xmax , ymin, ymax -> we need to get its centre
            player_centre = get_centre_of_box(bbox)

            min_dis = float('inf')
            # get the min dis of the keypoint from the player centre 

            for i in range(0 , len(court_keypoints) , 2):
                court_points = (court_keypoints[i] , court_keypoints[i+1])
                dist = get_euclidean_dis(player_centre , court_points)

                if(dist < min_dis):
                    min_dis = dist
                
            distances.append((track_id , min_dis))

        # sort the distance in ascend order and pick the first two 
        distances.sort(key = lambda x : x[1])

        chosen_players = (distances[0][0] , distances[1][0])

        return chosen_players

    def detect_mutiple_frames(self , frames , read_from_stubs = False , stub_path = None):
        """
        Detects the position of the person in each frame 
        and stitches all of them together and returns a players_detection list 
        over the entire video 
        """
        player_detections =[]

        if read_from_stubs and stub_path is not None:
            with open(stub_path , 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None: # store this prediction in the stub (next time -> saves time)
            with open (stub_path , 'wb') as f:
                pickle.dump(player_detections , f)


        return player_detections
    
    def detect_frame(self , frame):
        """
        Detects the position of the players in each frame and 
        returns their position 
        """
        results = self.model.track(frame , persist=True)[0]  # persist basically means the next frames in the video are coming
        id_name_dict = results.names

        player_dict = {}

        for box in results.boxes:
            track_id = int(box.id.tolist()[0])

            # gets the bounding box
            result = box.xyxy.tolist()[0]
            obj_class_id = box.cls.tolist()[0]
            obj_class_name = id_name_dict[obj_class_id]

            if obj_class_name == "person":
                player_dict[track_id] = result

        return player_dict
    

    def draw_bounding_boxes(self , video_frames , player_detections):
        """
        Sketches the bounding box around the detected keypoints of the players
        """
        output_video_frames = []

        for frame , player_dict in zip(video_frames , player_detections):
            
            #Draw the bounding boxes around the keypoints
            for track_id , bbox in player_dict.items():
                xmin , ymin , xmax , ymax = bbox

                # 2 means it's not filled and its just the boundary 

                cv2.putText(frame , f"Player Id :{track_id} " , ( int(bbox[0]) , int(bbox[1]-10) ), cv2.FONT_HERSHEY_COMPLEX , 0.9 , (0 , 0 ,255) ,2 )
                frame = cv2.rectangle(frame , (int(xmin) , int(ymin)) , ((int(xmax) , int(ymax))) ,(0, 0 , 255) , 2 )

            output_video_frames.append(frame) # it adds on top of the frames

        return output_video_frames






