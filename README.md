This sample project implements a reinforcement learning agent that scans an image to identify a handwritten digit while minimizing the number of steps required. The project is functional but still requires hyperparameter tuning and further improvements in the code.

The agent explores the image looking for informative features (white pixels) and learns when it has gathered enough information to make a correct prediction.

Training is performed in three stages:
1. A simple perceptron is trained to estimate digit probabilities. It is used by environment as part of the state.
2. The agent learns to navigate using four actions: up, down, left, right.
3. A fifth action (digit guess) is enabled, and the agent learns when to make a decision. Guess terminates episode and results in reward

The environment state includes:
- agent position (x, y)
- perceptron probabilities for each digit
- nearby pixels within a configurable discovery radius (rng_discover)

Rewards can be configured for:
- movement cost
- hitting a wall / staying in place
- correct guess
- incorrect guess

The objective is to learn a policy that minimizes the number of steps while maintaining high classification accuracy.


## Demo

![Agent demo](assets/demo.gif)

