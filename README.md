# UNREAL

## About

An attempt to repdroduce UNREAL algorithm described in Google Deep Mind's paper "Reinforcement learning with unsupervised auxiliary tasks."

https://arxiv.org/pdf/1611.05397.pdf

Work in progress. Now testing with ATARI Breakout, but the scores are lower than the A3C agent.

## Preview
[![Display tool](./doc/display0.png)](https://youtu.be/k0KpBP5rs5I)

## Network
![Network](./doc/network0.png)

All weights of convolution layers and LSTM layer are shared.

## Requirements

- TensorFlow (Tested with r0.12)
- gym atari
- numpy
- cv2
- pygame
- matplotlib

## Usage

To train
```
$ ./train
```
Then Ctrl-C to stop training.


To display the result,
```
$ ./display
```


# TODO
- Try using faster OpenAI's A3C implementation.
- Try DeepMind Lab environment.

