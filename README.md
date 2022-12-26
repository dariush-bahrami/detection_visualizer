# Detection Visualizer 🎨🖌️

Over engineered, satisfying and easy to use package to visualize object detection datasets and predictions.

## Dependencies
The goal is to keep this package as independent and simple as possible. There is only two dependencies:
* [NumPy](https://pypi.org/project/numpy/)
* [OpenCV](https://pypi.org/project/opencv-python/)

## How to use?

So you have an image and a bunch of annotations for that image. There is four possible annotation type:

* `ObjectDetectionGroundTruth`
* `ObjectDetectionPrediction`
* `SegmentationGroundTruth`
* `SegmentationPrediction`

Note that you can mix multiple types of annotations together for an image. you pass your image as numpy ndarray and a list of these annotations to an instance of `Visualizer` class and thats it! The `Visualizer` instance will return an image as numpy ndarray.

*Warning: There is chane that original image be mutated in the visualizer so if thats important to you make a copy of it before passing it the visualizer.*

## TODO:
* Add a proper documentation.