# mono_odo
Monocular Odometry

WORK IN PROGRESS

Monocular odometry pipeline based on 3d-2d correspondance, with bundle adjustement and loop closure.
Implemented in C++ with OpenCV, g2o and DLoopDetector
Hopefully this system will be adapted to work on thermal cameras (mainly the FLIR Vue)

PRESENT GOAL

1/ Get reasonable odometry results from 3d-2d correspondances

2/ Benchmarking feature descriptors on thermal images

TODO

2/ Add bundle adjustement with g2o

3/ Add loop closure with DLoopDetector

4/ Adapt to thermal images
