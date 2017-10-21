# UNREAL

[![CircleCI](https://circleci.com/gh/miyosuda/unreal.svg?style=svg)](https://circleci.com/gh/miyosuda/unreal)

## About

Replicating UNREAL algorithm described in Google Deep Mind's paper "Reinforcement learning with unsupervised auxiliary tasks."

https://arxiv.org/pdf/1611.05397.pdf

Implemented with TensorFlow and DeepMind Lab environment.

## Preview
seekavoid_arena_01

[![seekavoid_arena_01](./doc/display0.png)](https://youtu.be/1jF3gAdXfio)

stairway_to_melon

[![stairway_to_melon](./doc/display1.png)](https://youtu.be/FDA8QqUgdbo)

nav_maze_static_01

[![nav_maze_static_01](./doc/display2.png)](https://youtu.be/xHK0qBeH3-I)

## Network
![Network](./doc/network0.png)

All weights of convolution layers and LSTM layer are shared.

## Requirements

- TensorFlow (Tested with r1.0)
- DeepMind Lab
- numpy
- cv2
- pygame
- matplotlib

## Result
"seekavoid_arena_01" Level

![seekavoid_01_score](./doc/graph_seekavoid_01.png)

"nav_maze_static_01" Level

![nav_maze_static_01_score](./doc/graph_nav_maze_static_01.png)


## How to train
First, download and install DeepMind Lab
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
Add this bazel instruction at the end of `lab/BUILD` file

```
package(default_visibility = ["//visibility:public"])
```

Then run bazel command to run training.
```
bazel run //unreal:train --define headless=glx
```
`--define headlesss=glx` uses GPU rendering and it requires display not to sleep. (We need to disable display sleep.)

If you have any trouble with GPU rendering, please use software rendering with `--define headless=osmesa` option.

## How to show result

To show result after training, run this command.
```
bazel run //unreal:display --define headless=glx
```
