import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states
        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        N = len(self.hidden_states)
        T = len(input_observation_states)
        
                
        # edge case if there are no observation states
        if T == 0:
            return 1
        
        #probability matrix 
        forward_prob = np.zeros((T, N))

        # Initialization step
        for i in range(N):
            forward_prob[0, i] = self.prior_p[i] * self.emission_p[i, self.observation_states_dict[input_observation_states[0]]]
        
        # edge case of what if the observation states is only one state long        
        if T == 1:
            return np.sum(forward_prob[0,:])
        # Induction step
       
        for t in range(1, T):
            for j in range(N):
                forward_prob[t, j] = np.sum(forward_prob[t-1, :] * self.transition_p[:, j]) * self.emission_p[j, self.observation_states_dict[input_observation_states[t]]]
        print(forward_prob)

        # Termination step
        forward_probability = np.sum(forward_prob[T-1, :])
        return forward_probability


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        # Step 1. Initialize variables
        N = len(self.hidden_states)
        T = len(decode_observation_states)
        viterbi_table = np.zeros((T, N))
        best_path = np.zeros((T, N), dtype=int)
        
        if T == 0:
            return []
        
        for i in range(N):
            viterbi_table[0, i] = self.prior_p[i] * self.emission_p[i, self.observation_states_dict[decode_observation_states[0]]]
        
        # Step 2. Calculate Probabilities
        for t in range(1, T):
            for j in range(N):
                probabilities = viterbi_table[t-1, :] * self.transition_p[:, j]
                best_prev_state = np.argmax(probabilities)
                viterbi_table[t, j] = probabilities[best_prev_state] * self.emission_p[j, self.observation_states_dict[decode_observation_states[t]]]
                best_path[t, j] = best_prev_state
        
        # Step 3. Traceback
        best_last_state = np.argmax(viterbi_table[T-1, :])
        best_hidden_state_sequence = [best_last_state]
        
        for t in range(T-1, 0, -1):
            best_hidden_state_sequence.insert(0, best_path[t, best_hidden_state_sequence[0]])
        
        # Step 4. Return best hidden state sequence 
        return [self.hidden_states_dict[state] for state in best_hidden_state_sequence]


mini_hmm=np.load('./data/mini_weather_hmm.npz')
mini_input=np.load('./data/mini_weather_sequences.npz')


hmm = HiddenMarkovModel(mini_hmm['observation_states'], mini_hmm['hidden_states'], mini_hmm['prior_p'], mini_hmm['transition_p'], mini_hmm['emission_p'])

print(mini_input['observation_state_sequence'])


input_obs = []
forward = hmm.forward(input_obs)
viterbi = hmm.viterbi(input_obs)

print(mini_input['observation_state_sequence'])

print(forward)
print(viterbi)
print(mini_input['best_hidden_state_sequence'])


