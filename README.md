# Evolutionary-Robotics
A repo for my CS3060 assignments and projects.

## 1st milestone:
I would like to implement my own project idea involving testing different differential equation approximation methods
for ctrn's and traditional neural nets to determine the best performing neural structure for quadruped locomotion. To do so,
I will modify robot.py to transform its brain into a ctrn which can be evaluated with two separate approximation methods.
For the submission requirement I will take a video showing 2 working ctrn's, each using a different approximation method,
outputting its neuron values to the console.

## 2nd milestone:
I will modify robot.py again to make the robot able to sense and act using a ctrn instead. For
the submission requirement I will take a video showing a quadruped moving around using a ctrn for each approximation
method.


## 3rd milestone:
I will modify solution.py to be able to evolve both ctrn's and traditional neural nets using random modification. For
the submission requirement I will take a video showing a quadruped moving to the right using an evolved ctrn for each
approximation method.

## 4th milestone (preliminary A/B testing data):
I will modify search.py to randomly create 2 equal populations of crtns and traditional neural net brains. Then evolve them
by taking the brains that produce the highest displacement to the right and randomly mutating them to produce children
which we further simulate. We will then compare at the end which brian structur and approximation method produced the most
fit robots. For the submission requirement I will take a video showing the 3 most fit quadrupeds from each structure and
approximation method move in simulation.

