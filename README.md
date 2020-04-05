# Segmenting-Buildings-for-Disaster-Resilience--Open-Cities-AI-
Open Cities AI Challenge: Segmenting Buildings for Disaster Resilience
Jeetendra Gan (jgan2@buffalo.edu), Karun Parashar(karunpar@buffalo.edu), Chandana Singh(csingh@buffalo.edu)


Project description:

Our project is based on a challenge hosted on Driven Data (https://www.drivendata.org/). The goal in this challenge is to segment building footprints from aerial imagery. The data consists of drone imagery from 10 different cities and regions across Africa. Our goal is to identify the presence or absence of a building on a pixel-by-pixel basis.  

Prediction UI

![UI](/UI_sample.png)


Machine Learning Methods:

We primarily plan to use Convolutional neural networks for detecting edges of buildings. We will try implementing other techniques like building models on edge-detected images if time permits.

Data description:

Train data:

The data is around 200 GBs large, consisting primarily of images. Spatial resolution varies from region to region. All images include 4 bands: red, green, blue and alpha. The alpha band can be used to mask out NoData values.

Given that the labels vary in quality (e.g. how exhaustively an image is labeled, how accurate the building footprints are), the training data have been divided up into tier 1 and tier 2 subsets. The tier 1 images have more complete labels than tier 2.

Test Data:

The test set consists of 11,481 1024 x 1024 pixel COG "chips" derived from a number of different scenes. None of these scenes are included in the training set. Some of the test scenes are from regions that are present in the training set while others are not. The correct georeferences for the test chips have been removed. The test set labels (unavailable to participants) have a level of accuracy commensurate with the tier 1 data.

Labels:
Each image in the train set corresponds to a GeoJSON, where labels are encoded as FeatureCollections. geometry provides the outline of each building in the image. Your goal is only to classify the presence (or lack thereof) of a building on a pixel-by-pixel basis.

train_metadata.csv links the each image in the train set with its corresponding GeoJSON label file. This csv also includes the region and tier of the image. Note that region information is not provided for the test set.

Label GeoJSON files have been clipped to the extents of the non-NoData portions of the images, all building geometries will overlap with image data.


Next steps:

The next steps are:
Preprocess the data:
The data that we have is in the form of .tiff files which are very large. We plan to segment them in smaller chunks.
Initial data exploration
We plan to use the building boundary tags from the geoJSON files to draw lines on images.
Building model
CNN will be used to train the data
Building the GUI
We plan to build a desktop application to show results
i) Users will be able to upload any satellite image on our software and the software will draw boundaries around detected images.
