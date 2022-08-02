# 1. Cold days are encoded by 0 and hot days are a 1
# 2. The first day in our sequence has an 80% chance of being cold
# 3. A cold day has a 30% chance of being followed by a hot day
# 4. A hot day has a 20% chance of being followed by a cold day
# 5. On each day the temperature is normally distributed with a mean and standard deviation 
# of 0 and 5 on a cold day and mean and standard deviation 15 and 10 on a hot day.

import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions
initial_distribution = tfd.Categorical(probs=[0.8, 0.2]) # Refer to point 2
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3], 
                                                 [0.2, 0.8]]) # Refer to points 3 and 4
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.]) # Refer to point 5

# the loc argument represents the mean and the scale is the standard deviation

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7,
)

mean = model.mean()

# due to the way TensorFlow works on a lower level we need to evaluate part of the graph
# from within a session to see the value of this tensor

# in the new version of TensorFlow we need to use tf.compat.v1.Session() rather than just tf.Session()
with tf.compat.v1.Session() as sess:
    print(mean.numpy())