import pytest
from hmm import HiddenMarkovModel
import numpy as np


def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    #This is the standard test case
    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    hmm = HiddenMarkovModel(mini_hmm['observation_states'], mini_hmm['hidden_states'], mini_hmm['prior_p'], mini_hmm['transition_p'], mini_hmm['emission_p'])
        
    forward = hmm.forward(mini_input['observation_state_sequence'])
    viterbi = hmm.viterbi(mini_input['observation_state_sequence'])

    pytest.approx(0.03506, forward)

    assert np.array_equal(viterbi, mini_input['best_hidden_state_sequence']) 

    # Edge case 1: No observation states in the input
    input_obs = []
    edge1_forward = hmm.forward(input_obs)
    edge1_viterbi = hmm.viterbi(input_obs)
    assert edge1_forward == 1
    assert np.array_equal(edge1_viterbi, []) 

    # Edge case 2: One observation state in the input
    input_obs = ['sunny']
    edge2_forward = hmm.forward(input_obs)
    edge2_viterbi = hmm.viterbi(input_obs)
    assert edge2_forward == 0.45
    assert np.array_equal(edge2_viterbi, ['hot']) 

    # Edge case 3: Probabilities are 1 and 0
    prior_p = np.array([1, 0])
    test_p = np.array([[1,0], [1,0]])
    hmm = HiddenMarkovModel(mini_hmm['observation_states'], mini_hmm['hidden_states'], prior_p, test_p, test_p)
    edge3_forward = hmm.forward(mini_input['observation_state_sequence'])
    edge3_viterbi = hmm.viterbi(mini_input['observation_state_sequence'])
    assert(0, edge3_forward)
    assert(np.array_equal(edge3_viterbi, ['hot', 'hot', 'hot', 'hot', 'hot']))

def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    mini_hmm=np.load('./data/full_weather_hmm.npz')
    mini_input=np.load('./data/full_weather_sequences.npz')

    hmm = HiddenMarkovModel(mini_hmm['observation_states'], mini_hmm['hidden_states'], mini_hmm['prior_p'], mini_hmm['transition_p'], mini_hmm['emission_p'])
        
    forward = hmm.forward(mini_input['observation_state_sequence'])
    viterbi = hmm.viterbi(mini_input['observation_state_sequence'])

    assert(viterbi == mini_input['best_hidden_state_sequence']).all()

    













