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
   
## SpeedServAI Project Description


1. **Custom Ball Tracking with YOLO**: SpeedServAI initially employed a pretrained YOLO (You Only Look Once) model for ball tracking. However, to achieve more accurate results, a custom YOLO model was trained specifically for tennis ball tracking. Outliers in ball positions were removed, and the remaining positions were interpolated for smoother trajectory estimation.

2. **Player Tracking with Pretrained YOLO**: Leveraging a pretrained YOLO model, SpeedServAI identifies and tracks players on the court in real-time.

3. **Keypoint Detection with ResNet50 + Updated CNN**: To enhance accuracy further, SpeedServAI employs a ResNet50 architecture combined with an updated CNN on its final layer. This configuration allows for precise detection of keypoints on the court, providing detailed insights into player positioning and movement patterns.

