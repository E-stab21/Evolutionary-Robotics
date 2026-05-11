# Evolutionary Robotics Final Project Report

## Project Description
For my final project I decided to pick my own custom project idea where I would test three different types of
neural arcitetures to see which one produced the most fit quadruped robot. My fitness function was how far
the quadruped could move to the left in the simulation. The three brain variant's were as follows:
1. A traditional neural network connecting each sensor neuron to each motor neuron.
2. A 25 neuron CTRNN evaluated with Euler's method.
3. A 24 neuron CTRNN evaluated with Runge Kutta 4 aka RTK4.
*
I tackled the first milestone by coding 2 functional CRTNN's one using Euler's method and the other
using RTK4. For the second milestone
*

The underlying question I was attempting to answer was if CRTNN's improve the fitness of quadruped locomotion
and if so does a more sophisticated evaluation method matter in the context of performance. At the beginning, my
hypothesis was that the CRTNN would improve fitness but the evaluation method wouldn't matter since the
network would potentially be able to compensate for the error by adjusting its parameters.

## Implementation Details
To Implement the CRTNN's I created two separate python classes both inheriting from a parent Design class one for
the traditional NN's and one for the CTRNN's. The traditional class had the exact same brain structure as the original
___ class but the CRTNN class included a boolean variable for whether it should be evaluated as Euler or not as well as
the number of neurons, time constant, weights, biases, and activation function (which is tanh()). The way the two get send to
the simulation process is fundamentally different as well. The traditional class is send the same with a generated brain.nndf
but the CRTNN class is saved as a .pkl file using the pickle module so all of the brain's information encapsulated. On the
simulation side when the robot class is constructed its passed either one the files and based on the extension it sets the
is_ctrnn variable to true or false and either creates the NN using NEURAL_NETWORK() or load the pickle file. In the sense
function if the brain is a CRTNN then the values from all of the sensors are collected and then passed through the network
to get the neuron activation's for that time step represented as a vector. How the sensor values are pass through depends
on if the is_euler variable is true or false as well. In the act function if the brain is a CRTNN we iterate through all
of the motors and set their desired angle equal to the activation value of the corresponding motor neuron in the brain state
vector squeezed into a range between the LOWER_JOINT_LIMIT and UPPER_JOINT_LIMIT. I switched the method of starting a subprocess
from the os module to the subprocess module allowing the reading of process output for easy fitness value collection.
This was done by simply printing the fitness value on the last line of simulate.py and reading it from desgin.py.
### Evolutionary Algorithm
I modified the evolutionary algorithm so it sorts the population of each variant by fitness, destroys the bottom half, and then
repopulates by spawning randomly modified copies of the first half.
### Parameters
Population Size: 10


## Results
The results from my experiment showed that evolution occurred over time because when looking at the provided fitness curve its clear that over the 50 generations the fitness improved dramatically when comparing the randomly initalized brain. However the conclusions that can be drawn about specifically the differences in performance between the 3 brain variants is much more nuanced when looking at the data. It appears that the traditional neural net arcitecutre shown in red was the best overall performing model since its fitness values were consistently higher than the other two for most of the trials however in some of the traditnokl NN trials the fitness values were much lower than the top preforming CTRNN trials so it wasn't able to outpreform in all cases. When looking at the difference between the two CRTNN groups it is clear that they both share a very similar fitness ceiling however it appears that the RTK4 trials are more tightly grouped together meaning that the RTK4 method was more consistent in its robot performances. This could potentially demonstrate the fact that a more accuratly evaluated CRTNN will lead to more consistent evolutionary progress and performance since small tweaks to the parameters result in less unpredictable behavior. It is also clear that the Euler CTRNN trials were much more variation because there is a wide fitness range between the trials at each time step.
(image)
(pictures)

## Future Potential
I difficult decision I had to make while designing this experiment was what differential evaluation methods I wanted to use. I considered using Euler and Heun's method but I wanted two methods that were vastly different in their aproximation accuracly because the question I was set on answering was whether or not the accuracly had any real impact on the evolution of the robot. This is precisely why I settled on Euler's and RK4 because Euler's is the simplest and least accurate evaluation method while RK4 is one of the most accurate as a 4th order method. With this large gap between accuracy it would be much easier to see a difference in performance between the two.

Another difficult task during the course of the project was stripping out the prexisting robot brian and replacing it with a CRTNN. I had use pyrosim functions so get the values at the sensor neurons and pass those values through the CRTNN fucntion to get the current brain state. Then in the act I had to iterate through all of the desginated motors neurons in order to fire all the motors at the proper angle based on the neuron activation.

Some aspects of the project I assumed were going to be very difficult that ended up being quite easy was the actual implementation of RK4. I assumed it was going to be an extremely complicated and advanced technique but it ended up being very straight forward when implementing it in the robot.

If I had another year to work on this project I would likely try to test out many other neural arcitechure's for instance hybrid models, Spiking Neural Networks, or Hebbian Neural Networks. I would also like to run the exact same experiment but on a differntail simulator to see if there is a performance gain when using CTRNN's. Would could also be interesting is testing out techniques typically used in other areas of machine learning like transformers and attention mechanisms or Graph Neural Networks.

I presume that replacing the CTRNN with a different type of architecture would be relatively simple because the interaction between the brain and body are already in place however moving to a entirely different simulator would likely require lots of work to transfers the experiments functionality over.
