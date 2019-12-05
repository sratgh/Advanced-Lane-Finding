# **Advanced Lane Finding Project**

![Project video Part 4/5](https://github.com/sratgh/Advanced-Lane-Finding/blob/master/out_challenge_video_4.gif)

## Goals of the project
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Installation / Setup

There are two ways of running this project:

**1) The first one is to use jupyter notebook**
* For get up and running with jupyter notebook please refer to the official documentation website:
[Install Jupyter Notebook](https://jupyter.org/install)
* Installing opencv can be a bit tricky. Here is a good installation doc for mac [Install Documenation for OpenCV](https://medium.com/@nuwanprabhath/installing-opencv-in-macos-high-sierra-for-python-3-89c79f0a246a)
* It is recommended to make use of virtual environments.
* After the installation you should be able to start the project with
`jupyter notebook P1.ipynb`

**2) The second one is to use a docker environment**
* Install Docker environment
For mac:
[Install Docker on Mac](https://docs.docker.com/v17.12/docker-for-mac/install/)
* Then pull the udacity docker environment with
`docker pull udacity/carnd-term1-starter-kit`
* Run the jupyter notebook in a docker environment with the following command:
`docker run -it --rm --entrypoint "/run.sh" -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit`

## License

This project is licensed under the MIT License.
[LICENSE](https://github.com/sratgh/Lane-finding/blob/master/LICENSE)
