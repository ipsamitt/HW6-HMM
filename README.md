[![project6](https://github.com/ipsamitt/HW6-HMM/actions/workflows/main.yml/badge.svg)](https://github.com/ipsamitt/HW6-HMM/actions/workflows/main.yml)
# HW6-HMM

In this assignment, you'll implement the Forward and Viterbi Algorithms (dynamic programming). 


# Assignment

## Methods Description
The forward algorithm of an HMM computes the probability of the observed states. This algorithm iterates through the list of states within the input and updates the probability matrix by summing over all the potential previous states by applying transition and emission probabilities. It returns the sum of the final column of the matrix as the joint probability of the input observations.

The Viterbi algorithm finds the most likely sequence of hidden states that could have created the input observation states to decode the observation states and find the most probable underlying path. This is done by iterating through each state in the input observations and computing the most likely state transition at each step and finding the best previous state that leads to the highest probability. These best previous states are saved later for traceback, starting from the last state with the highest probability to reconstruct the most probable hidden state sequence.





