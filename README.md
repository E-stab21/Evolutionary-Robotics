# Evolutionary-Robotics
A repo for my CS3060 assignments and projects.

The first one is the traditinol NN method.
The second one is a CTRNN using Euler's method
The third one is a CTRNN using Runge-Kutta 4.
I am evolving for 30 generations.

## 1st milestone:
I would like to implement my own project idea involving testing 2 different differential equation approximation methods
for ctrnn's and a traditional neural net to determine the best performing neural structure for quadruped locomotion. To do so,
I will modify robot.py to transform its brain into a ctrnn which can be evaluated with two separate DE approximation methods,
Euler's method and Runge-Kutta 4. For the submission requirement, for milestone 1, I will take a video showing the
output of 2 working ctrnn's, each using a different approximation method, outputting all its neuron values to the console
at each time step.

## 2nd milestone:
I will modify robot.py again to make the robot able to sense and act using a ctrnn instead of a traditional neural net.
For the submission requirement I will take a video showing 2 quadrupeds being controlled by ctrnn's each being
evaluated using different approximation methods and their respective neural outputs at each time step.

## 3rd milestone:
I will modify solution.py to be able to evolve and mutate both ctrnn's and traditional neural nets using random modification.
For the submission requirement I will take a video showing 2 quadrupeds moving to the left being controlled by
evolved ctrnn's each using a different approximation methods.

## 4th milestone (preliminary A/B testing data):
I will modify search.py to randomly generate 3 equal populations of same-body robot designs with the first 2 being controlled
by ctrnn's evaluated using Euler's and Runge-Kutta 4 respectively and the last being controlled by traditional neural nets.
Then we will take the designs that produce the highest displacement to the left and randomly mutate them to produce children
which we further simulate and mutate. We will then compare at the end which brain structure and approximation method produced the most
fit robot brains. For the submission requirement I will take a video showing the 3 most fit quadrupeds from each structure and
approximation method combination mentioned above moving in the simulation after a certain number of generations.

