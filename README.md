# Face Detection & Recognition (OpenCV + Keras)

A small end-to-end demo:
1) capture faces
2) train a lightweight CNN
3) run recognition

## Requirements
- Python 3.x
- OpenCV
- TensorFlow / Keras
- Camera (Raspberry Pi or PC)

## How to run
- `capturing_faces.py` → collect samples
- `learning_faces.py` → train
- `recognise_faces.py` → inference

## Notes
- Accuracy depends on dataset size, lighting, and camera quality.
- Do not use for sensitive/production identification use-cases.

Model accuracy is almost %98


#Usage
![alt text](https://github.com/taskma/Face_detect_and_recognise/blob/master/rapberrypie.png)
![alt text](https://github.com/taskma/Face_detect_and_recognise/blob/master/face.png)
![alt text](https://github.com/taskma/Face_detect_and_recognise/blob/master/cv2_tensorflow.png)


1) Run capturing_faces.py for diffrent users (using opencv for capturing faces from camera)

2) Run learning_faces.py for learning user faces (using tensorflow-keras for cnn deep learning)

3) Run Recognise_faces.py for  recognising faces that learned (using opencv and tensorflow-keras)

![alt text](https://github.com/taskma/Face_detect_and_recognise/blob/master/accuracy.png)

Model accuracy is almost %98
