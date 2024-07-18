from utils import (read_video , 
                   save_video , 
                   get_time_in_seconds,
                   get_euclidean_dis)
import os
from trackers import PlayerTracker , BallTracker , convert_pixel_to_meter

from mini_court.mini_court_sketch import MiniCourtSketch
from court_line_detector import CourtLineDetector
import cv2 as cv2
from copy import deepcopy
import pandas as pd
from player_stats_display.player_stats_project import draw_player_stats
from court_constants import constants 

KMPH_CONVERT = 3.6

def main():

    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)
    
    # Detect players 
    player_tracker = PlayerTracker(model_path= "models/yolov8x")

    # first time we run with False , once the predictions are saved -> directly access it from the pickle
    player_detections = player_tracker.detect_mutiple_frames(video_frames ,
                                                             read_from_stubs=True ,
                                                             stub_path = "tracker_stubs/player_detections.pkl"
                                                            ) 
    

    # Detect balls 
    ball_tracker = BallTracker(model_path= "models/yolo5_last.pt")
    ball_detections = ball_tracker.detect_mutiple_frames(video_frames ,
                                                            read_from_stubs=True ,
                                                            stub_path = "tracker_stubs/ball_detections.pkl"
                                                        ) 
    ball_detections = ball_tracker.interpolate_ball_positions(ball_positions= ball_detections) # interpolate to constantly track it
    
    # detect ball shot frames
    ball_shot_frames = ball_tracker.get_ball_hit_positions(ball_positions=ball_detections)
    # print(ball_shot_frames)





    # Detect the court lines 
    court_model_path = "models/keypoints_detect.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0]) # give only one frame as the camera motion is static and track points wont change 

    # filter and get the two players 
    player_detections = player_tracker.choose_two_player(court_keypoints= court_keypoints , player_detections= player_detections)
    # print(player_detections)

              

    # Draw the player  , ball , court key points bounding boxes 
    output_video_frames = player_tracker.draw_bounding_boxes(video_frames , player_detections)
    output_video_frames = ball_tracker.draw_bounding_boxes(output_video_frames , ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames , court_keypoints)

    # draw the mini court
    mini_court = MiniCourtSketch(video_frames[0])
    output_video_frames = mini_court.draw_mini_court_all_frames(output_video_frames)

    player_mini_court_pos , ball_mini_court_pos = mini_court.map_player_and_ball_from_actual_to_mini_court(player_detections , ball_detections , 
                                                                                                           court_keypoints)
    
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames , 
                                                            ball_mini_court_pos, 
                                                            color = (0,255,255))
    

    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames , 
    
                                                               player_mini_court_pos)
    

    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    } ]
    
    for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind+1]
        ball_shot_time_in_seconds = (end_frame-start_frame)/24 # 24fps

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = get_euclidean_dis(ball_mini_court_pos[start_frame][1],
                                                           ball_mini_court_pos[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_to_meter( distance_covered_by_ball_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.width_of_mini_court()
                                                                           ) 

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * KMPH_CONVERT

        # player who the ball
        player_positions = player_mini_court_pos[start_frame]
        player_shot_ball = min( player_positions.keys(), key=lambda player_id: get_euclidean_dis(player_positions[player_id],
                                                                                                 ball_mini_court_pos[start_frame][1]))

        # opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = get_euclidean_dis(player_mini_court_pos[start_frame][opponent_player_id],
                                                                player_mini_court_pos[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_pixel_to_meter( distance_covered_by_opponent_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.width_of_mini_court()
                                                                           ) 

        speed_of_opponent = distance_covered_by_opponent_meters/ball_shot_time_in_seconds * KMPH_CONVERT

        current_player_stats= deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']
    # draw the player stats 

    output_video_frames = draw_player_stats(output_video_frames= output_video_frames , player_stats=player_stats_data_df)


    directory = 'output_videos'
    if os.path.exists(directory):
        os.system(f'rm -rf {directory}')
        
    os.makedirs(directory)


    #add frame number in each frame of the video in the top left corner 
    for i , frame in enumerate(output_video_frames):
        cv2.putText(frame , f"Frame : {i} " , (10,30) , cv2.FONT_HERSHEY_COMPLEX , 1 , (0, 255 , 0) , 2)

    save_video(output_video_frames , "output_videos/output_video.avi")


if __name__ =='__main__':
    main()

