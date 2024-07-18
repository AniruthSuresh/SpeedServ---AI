# SpeedServ---AI
SpeedServAI is a machine learning project that tracks tennis players' speed, position, and ball speed in real-time video. Using YOLO for object detection and CNNs for feature extraction, it provides accurate, real-time insights into match dynamics, enhancing performance analysis . 


### Installation and Setup

1. Clone the git repo :
   ```bash
   git clone https://github.com/AniruthSuresh/SpeedServ---AI.git
   cd SpeedServ---AI

2. Create the conda environment (tennis in this case) using the provided `env.yml` file and activate it:
   
   ```bash
   conda env create -f env.yml
   conda activate tennis

3. To get the final output :
   ```bash
   cd src
   python3 main.py

## Training

To train the models used in SpeedServAI, follow these steps:

### Tennis Ball Tracking Model (YOLO)

1. Navigate to the `src/training` directory.
2. Open and run `tennis_ball_training.ipynb`.
3. Save the trained model as `yolo5_last.pt` in the `src/models` directory.

### Keypoint Detection on the Court Model

1. Navigate to the `src/training` directory.
2. Open and run `tennis_keypoint_court.ipynb`.
3. Save the trained model as `keypoints_detect.pth` in the `src/models` directory.


## SpeedServAI Project Description

1. **Custom Ball Tracking with YOLO**: SpeedServAI initially employed a pretrained YOLO (You Only Look Once) model for ball tracking. However, to achieve more accurate results, a custom YOLO model was trained specifically for tennis ball tracking. Outliers in ball positions were removed, and the remaining positions were interpolated for smoother trajectory estimation.

2. **Player Tracking with Pretrained YOLO**: Leveraging a pretrained YOLO model, SpeedServAI identifies and tracks players on the court in real-time.

3. **Keypoint Detection with ResNet50 + Updated CNN**: To enhance accuracy further, SpeedServAI employs a ResNet50 architecture combined with an updated CNN on its final layer. This configuration allows for precise detection of keypoints on the court, providing detailed insights into player positioning and movement patterns.


## SpeedServAI Demo

![SpeedServAI Demo](https://github.com/user-attachments/assets/36d41dd7-8bab-4f11-b271-2da3c388d3bf)

This GIF demonstrates the output of SpeedServAI in action, showcasing its capabilities in tracking tennis players , ball  and analyzing match dynamics in real-time.

## Future Scope of Improvement

SpeedServAI has potential for enhancement in the following areas:

1. Improved Ball Tracking

- **Larger Dataset Training**: Enhance ball tracking accuracy by training the YOLO model on a larger dataset. This would allow for smoother trajectory estimation and more robust tracking in varying conditions.

2. Enhanced Keypoint Detection

- **Homography for Keypoint Estimation**: Implement homography techniques to reconstruct shifted keypoints accurately. This method could improve the precision of player position detection on the court, especially in dynamic gameplay scenarios.





