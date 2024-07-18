from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:

    def __init__(self , model_path):
        self.model = YOLO(model_path)

    def detect_mutiple_frames(self , frames , read_from_stubs = False , stub_path = None):
        """
        Detects the position of the ball in each frame 
        and stitches all of them together and returns a ball_detection list 
        over the entire video 
        """
        ball_detections =[]

        if read_from_stubs and stub_path is not None:
            with open(stub_path , 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        if stub_path is not None: # store this prediction in the stub (next time -> saves time)
            with open (stub_path , 'wb') as f:
                pickle.dump(ball_detections , f)


        return ball_detections
    
    def detect_frame(self , frame):
        """
        Detects the position of the ball in each frame and 
        returns its position 
        """
        # we are not tracking the ball -> so remove the persist
        results = self.model.predict(frame , conf = 0.2 )[0] 
        ball_dict = {}

        for box in results.boxes:
            # gets the bounding box
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result


        return ball_dict
    
            
    def interpolate_ball_positions(self , ball_positions):
        """
        Interpolates missing ball position data using pandas DataFrame operations.
        """
        # if key = 1 -> get the positions 
        # if no kety -> returns empty list
        ball_positions = [x.get(1, []) for x in ball_positions]

        df_ball_posi = pd.DataFrame(ball_positions , columns= ['xmin' , 'ymin' , 'xmax' , 'ymax'])  # convert to pandas
        df_ball_posi = df_ball_posi.interpolate() # fills the missing values 

        # if the initial values are missing -> then interpolate will raise an error 
        df_ball_posi = df_ball_posi.bfill() # back filling 

        ball_positions = [{1 : x} for x in df_ball_posi.to_numpy().tolist()] # return back to the original df 

        return ball_positions
    
    def interpolate_ball_positions_as_table(self , ball_positions):
        """
        Interpolates missing ball position data using pandas DataFrame operations
        and returns as pandas format which can be readily used for visualisation 
        """
        # if key = 1 -> get the positions 
        # if no kety -> returns empty list
        ball_positions = [x.get(1, []) for x in ball_positions]

        df_ball_posi = pd.DataFrame(ball_positions , columns= ['xmin' , 'ymin' , 'xmax' , 'ymax'])  # convert to pandas
        df_ball_posi = df_ball_posi.interpolate() # fills the missing values 

        # if the initial values are missing -> then interpolate will raise an error 
        df_ball_posi = df_ball_posi.bfill() # back fillfing 

        return df_ball_posi
    
    def get_ball_hit_positions(self , ball_positions):
        """
        This function gets the frame position of where the ball is in 
        contact with the racket which can be used to obtain and track the
        speed of both ball and players 
        """
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['xmin','ymin','xmax','ymax'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['ymin'] + df_ball_positions['ymax'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                
                change_count = 0 

                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_positions['ball_hit'].iloc[i] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits



    def draw_bounding_boxes(self , video_frames , ball_detections):
        """
        Sketches the bounding box around the detected keypoints 
        """
        output_video_frames = []

        for frame , ball_dict in zip(video_frames , ball_detections):
            
            #Draw the bounding boxes around the keypoints
            for track_id , bbox in ball_dict.items():
                xmin , ymin , xmax , ymax = bbox

                # 2 means it's not filled and its just the boundary 

                cv2.putText(frame, f"Ball Id: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_COMPLEX, 0.9, (73, 136, 250), 2)
                frame = cv2.rectangle(frame , (int(xmin) , int(ymin)) , ((int(xmax) , int(ymax))) ,(73, 136, 250) , 2 ) # B G R format 

            output_video_frames.append(frame) # it adds on top of the frames

        return output_video_frames
    







