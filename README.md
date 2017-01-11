# UNREAL

## About

Replicating UNREAL algorithm described in Google Deep Mind's paper "Reinforcement learning with unsupervised auxiliary tasks."

https://arxiv.org/pdf/1611.05397.pdf

Implemented with TensorFlow and DeepMind Lab environment.

## Preview
seekavoid_arena_01

[![seekavoid_arena_01](./doc/display0.png)](https://youtu.be/1jF3gAdXfio)

stairway_to_melon

[![stairway_to_melon](./doc/display1.png)](https://youtu.be/FDA8QqUgdbo)

## Network
![Network](./doc/network0.png)

All weights of convolution layers and LSTM layer are shared.

## Requirements

- TensorFlow (Tested with r0.12)
- DeepMind Lab
- numpy
- cv2
- pygame
- matplotlib

## Result
Score plot of DeepMind Lab "seekavoid_arena_01" environment.

![seekavoid_01_score](./doc/graph_seekavoid_01.png)

## How to run
First, dowload and install DeepMind Lab
```
$ git clone https://github.com/deepmind/lab.git
```
Then build it following the build instruction. 
https://github.com/deepmind/lab/blob/master/docs/build.md

Clone this repo in lab directory.
```
$ cd lab
$ git clone https://github.com/miyosuda/unreal.git
```
Add this bazel instrution at the end of `lab/BUILD` file

```
package(default_visibility = ["//visibility:public"])
```

Then run bazel command to run training.
```
bazel run //unreal:train --define headless=osmesa
```

To show result after training, run this command.
```
bazel run //unreal:display --define headless=osmesa
```
