import numpy as np
import pprint
import random

# Forward Algorithm - Alpha(t)

def forward(data, hidden_transitions, emission_transitions, initial_distribution):
    # Create an "alpha" array with same dimensions of data's for the calculation at Alpha(t):
    # Alpha[t,j] = Alpha(t) with respect of m_t = j: 0 -> 1; e.g: alpha[t,0] = p(m_t = 0 | x: 1 -> t)
    alpha = np.zeros((data.shape[0], data.shape[1]))

    #Calculating Alpha(1):
    alpha[0,:] = initial_distribution * emission_transitions[:, int(data[0,1])]

    # Calculating Alpha(2 -> T)
    for t in range(1, data.shape[0]):
        for j in range(data.shape[1]):
            # Alpha(t) = Sum (of all m_t-1) of [Alpha(t-1) * p(m_t | m_t-1)] * p(x_t | m_t)
            m_t = j
            x_t = int(data[t, 1])

            alpha[t,j] = alpha[t-1].dot(hidden_transitions[:, m_t]) * emission_transitions[m_t, x_t]

    return alpha

# ALPHA TEST
# alpha_test = forward(test, hidden_transitions, emission_transitions, initial_distribution)
#print(alpha_test)

############################################################################################################

# Backward Algorithm - Beta(t)

def backward(data, hidden_transitions, emission_transitions):
    # Create an "beta" array with same dimensions of data's for the calculation at Beta(t):
    beta = np.zeros((data.shape[0], data.shape[1]))

    # Theoretically, Beta(t) = 1 when t = T
    # Thus setting up Beta(T) = 1
    beta[data.shape[0] - 1] = np.ones(data.shape[1])

    # Calculating Beta(T-2 -> t)
    for t in range(data.shape[0] - 2, -1, -1):
        for j in range(data.shape[1]):
            # Beta(T) = Sum (of all T+1) of [Beta(T+1) * p(m_T+1 | m_T) * p(x_T+1 | m_T+1)]
            m_T = j
            x_T = int(data[t+1, 1])

            beta[t,j] = (beta[t + 1] * hidden_transitions[m_T, :]).dot(emission_transitions[:, x_T])

    return beta


# BETA TEST
#beta_test = backward(test, hidden_transitions, emission_transitions)
#print(beta_test)


############################################################################################################

# Baum-Welch Algorithm (Multiple Observations)

def baum_welch(data, hidden_transitions, emission_transitions, initial_distribution, epsilon):

    # Total hidden states of hidden_transitions:
    M = hidden_transitions.shape[0]

    # Total times:
    T = data.shape[1]

    # Total set of Observations:
    D = data.shape[0]



    # "count" records the consecutive number of times when the overall loss/parameters estimation changes remain
    # relatively small.
    convergence_count = 0

    # We repeat the algorithm until reaches Convergence:
    while True:
        numerator_hidden = np.zeros((2,2))
        denominator_hidden = np.zeros((2))
        numerator_emission = np.zeros((2,2))
        denominator_emission = np.zeros((2))

        for d in range(D):
            # Extract data at d; data.shape = (D, T, 2)
            data_d = data[d]

            # Then we compute parameters for data_d:
            # "Visible" Column from data_d:
            data_visible = data_d[:, 1]

            # Calculate Alpha and Beta tables:
            alpha = forward(data_d, hidden_transitions, emission_transitions, initial_distribution)
            beta = backward(data_d, hidden_transitions, emission_transitions)

            # Initialising the table of Xi:
            # Dimension of Xi depends on i, j and t; thus Xi is 3-D matrix
            Xi = np.zeros((M, M, T - 1))

            # Calculate Xi(t:1 -> T-1) because of calculating beta in line 126:
            for t in range(T - 1):
                x_T = int(data_visible[t + 1])  # x_t+1

                # Denominator_xi = S_i S_j of [alpha(m_t = i) * p(m_t+1 = j| m_t = i) * p(x_t+1 = k| m_t+1 = j) * beta(m_t+1 = j)]
                denominator_xi = np.dot(emission_transitions[:, x_T] * beta[t + 1, :],
                                        np.dot(alpha[t, :], hidden_transitions))

                for i in range(M):
                    # Calculate Xi_i:0 -> 1,j(t:0 -> T-1):
                    numerator_xi = alpha[t, i] * hidden_transitions[i, :] * emission_transitions[:, x_T] * beta[t + 1,
                                                                                                           :]

                    Xi[i, :, t] = numerator_xi / denominator_xi

                    # Note: Iterate through i and t are more efficient to calculate Xi_i,j(t)

            # Calculate Gamma: (note that: resulted from t:1 -> T-1) because of Xi's length is T-1
            # Gamma_i(t) = S_j of Xi_i,j(t):
            Gamma = np.sum(Xi, axis=1)  # Sum of columns (t) of each matrix i in Xi.

            # We update denominator_hidden and numerator_hidden:
            numerator_hidden += np.sum(Xi, axis= 2)
            denominator_hidden += np.sum(Gamma, axis= 1)

            # Since Xi's length is T-1, Gamma(T) needs to be added to Gamma
            # Note that Gamma(T) cannot be expressed in term of Xi(T), thus we need backward-forward expression:
            Gamma_T = alpha[T-1] / np.sum(alpha[T-1], axis= 0)

            # Add Gamma_T into Gamma
            Gamma = np.hstack((Gamma, Gamma_T.reshape(-1, 1)))

            # Update denominator_emission:
            denominator_emission += np.sum(Gamma, axis= 1)

            # Calculate Emission_transitions:
            K = emission_transitions.shape[1]  # K: numbers of observed states

            for k in range(K):
                X_k = np.array((data_visible == k)).astype(int)

                # Sum over Gamma that has "k" observed at t.
                numerator_emission[:, k] += np.sum(Gamma * X_k, axis= 1)


        # Calculating parameters of new Hidden_transitions and Emission_transitions:
        cur_hidden_transitions = numerator_hidden / denominator_hidden.reshape((-1,1))
        cur_emission_transitions = numerator_emission / denominator_emission.reshape((-1,1))

        # Check convergence of new parameters and old parameters:
        old_parameters = np.array([hidden_transitions[0, 1], hidden_transitions[1, 0],
                                   emission_transitions[0, 1], emission_transitions[1, 0]])
        new_parameters = np.array([cur_hidden_transitions[0, 1], cur_hidden_transitions[1, 0],
                                   cur_emission_transitions[0, 1], cur_emission_transitions[1, 0]])

        if np.linalg.norm(new_parameters - old_parameters) <= epsilon:
            # if Δ is small, we know it has reached its local maximum.
            convergence_count += 1
            if convergence_count == 10: break  # break the EM.

        else:
            convergence_count = 0


        hidden_transitions = cur_hidden_transitions
        emission_transitions = cur_emission_transitions

    return {"hidden_transitions": hidden_transitions, "emission_transitions": emission_transitions,
            "total_probability": np.hstack((np.sum(hidden_transitions, axis=1), np.sum(emission_transitions, axis=1)))}

############################################################################################################
# Baum-Welch for Single Observation:
def single_baum_welch(data, hidden_transitions, emission_transitions, initial_distribution, epsilon):

    # Total hidden states of hidden_transitions:
    M = hidden_transitions.shape[0]

    # Total times:
    T = data.shape[0]

    # "Visible" Column from data:
    data_visible = data[:, 1]

    # "count" records the consecutive number of times when the overall loss/parameters estimation changes remain
    # relatively small.
    count = 0

    # Repeat the algorithm until convergence (reaches the n_iter):
    while True:
        # Set up new Hidden and Emission matrix for current computation:
        cur_hidden_transitions = np.zeros((2,2))
        cur_emission_transitions = np.zeros((2,2))

        # Calculate Alpha and Beta tables:
        alpha = forward(data, hidden_transitions, emission_transitions, initial_distribution)
        beta = backward(data, hidden_transitions, emission_transitions)

        # Initialising the table of Xi:
        # Dimension of Xi depends on i, j and t; thus Xi is 3-D matrix
        Xi = np.zeros((M, M, T-1))

        # Calculate Xi(t:1 -> T-1) because of calculating beta in line 126:
        for t in range(T-1):
            x_T = int(data_visible[t+1])  # x_t+1

            # Denominator_xi = S_i S_j of [alpha(m_t = i) * p(m_t+1 = j| m_t = i) * p(x_t+1 = k| m_t+1 = j) * beta(m_t+1 = j)]
            denominator_xi = np.dot(emission_transitions[:, x_T] *  beta[t+1, :],
                                    np.dot(alpha[t, :], hidden_transitions))

            for i in range(M):
                # Calculate Xi_i:0 -> 1,j(t:0 -> T-1):
                numerator_xi = alpha[t, i] * hidden_transitions[i, :] * emission_transitions[:, x_T] * beta[t+1, :]

                Xi[i, :, t] =  numerator_xi / denominator_xi

                # Note: Iterate through i and t are more efficient to calculate Xi_i,j(t)

        # Calculate Gamma: (note that: resulted from t:1 -> T-1) because of Xi's length is T-1
        # Gamma_i(t) = S_j of Xi_i,j(t):
        Gamma = np.sum(Xi, axis = 1) # Sum of columns (t) of each matrix i in Xi.

        # Calculate entrances of hidden_transitions:
        cur_hidden_transitions = np.sum(Xi, axis= 2) / np.sum(Gamma, axis= 1).reshape(-1,1)

        Gamma_T = alpha[T-1] / np.sum(alpha[T-1], axis= 0)

        Gamma = np.hstack((Gamma, Gamma_T.reshape((-1,1))))
        # Calculate Emission_transitions:
        K = emission_transitions.shape[1] # K: numbers of observed states

        # First we calculate the denominator_gamma to avoid repeating computation.
        denominator_gamma = np.sum(Gamma, axis= 1)
        for k in range(K):
            X_k = np.array((data_visible == k)).astype(int)

            cur_emission_transitions[:, k] = np.sum(Gamma * X_k, axis= 1) / denominator_gamma

        # Check for Convergence with previous Hidden and Emission transitions:
        # Stack the first columns from old and new Hidden and Emission transitions:
        old_parameters = np.array([hidden_transitions[0,1], hidden_transitions[1,0],
                                  emission_transitions[0,1], emission_transitions[1,0]])
        new_parameters = np.array([cur_hidden_transitions[0,1], cur_hidden_transitions[1,0],
                                  cur_emission_transitions[0,1], cur_emission_transitions[1,0]])

        # We compared old to new parameters by calculating Δ(θ[n],θ[n+1])<ε, where Δ is a distance metric:

        if np.linalg.norm(new_parameters - old_parameters) <= epsilon:
            # if Δ is small, we know it has reached its local maximum.
            count += 1
            if count == 10: break # break the EM.

        else:
            count = 0

        hidden_transitions = cur_hidden_transitions
        emission_transitions = cur_emission_transitions

    return {"hidden_transitions": hidden_transitions, "emission_transitions": emission_transitions,
            "total_probability": np.hstack((np.sum(hidden_transitions, axis=1), np.sum(emission_transitions, axis=1)))}

# Generate Sequence using A, B and pi (Used for Single Observation Baum-Welch)
def sequenceGenerator(T, hidden_transitions, emission_transitions, initial_distribution):

    # Generate a table of dimension (T,2):
    data = np.zeros((T,2))

    # Set of transition and emission states:
    states = np.arange(0,2)

    # Generate m_0 and x_0:
    m_0 = np.random.choice(states, p = initial_distribution)
    x_0 = np.random.choice(states, p = emission_transitions[m_0, :])

    data[0, :] = [m_0, x_0]

    # Generating the rest of data:
    for t in range(1, T):
        # data[t-1, 0] = m_t-1
        m_t = np.random.choice(states, p = hidden_transitions[int(data[t-1, 0]), :])
        x_t = np.random.choice(states, p = emission_transitions[m_t, :])

        data[t, :] = [m_t, x_t]

    return data

# Generate D-size set of Sequence (Used for Multiple Observations Baum-Welch)
def dataGenerator(D, T, hidden_transitions, emission_transitions, intial_distribution):
    data = np.zeros((D,T,2))

    for d in range(D):
        data[d] = sequenceGenerator(T, hidden_transitions, emission_transitions, intial_distribution)

    return data

def error_percentage(true_hidden, true_emission, test_hidden, test_emission):
    error_learning = abs(test_hidden[0,1] - true_hidden[0,1])
    error_forgetting = abs(test_hidden[1,0] - true_hidden[1,0])
    error_slipping = abs(test_emission[1,0] - true_emission[1,0])
    error_guessing = abs(test_emission[0,1] - true_emission[0,1])

    error_hidden = {"learning_parameters_error": error_learning, "forgetting_parameters_error": error_forgetting}
    error_emission = {"slipping_parameters_error": error_slipping, "guessing_parameters_errors": error_guessing}

    return {"hidden": error_hidden, "emission": error_emission}

############################################################################################################
# MAIN TEST:

# Generating data for the Hidden Markov Model.

# Transition Probabilities:
true_hidden_transitions = np.array(([0.4, 0.6],
                                    [0.2, 0.8]))

# Emission Probabilities:
true_emission_transitions = np.array(([0.7, 0.3],
                                      [0.1, 0.9]))

# Initial distribution:
initial_distribution = np.array([0.5, 0.5])

sequence_data = dataGenerator(1000, 20, true_hidden_transitions, true_emission_transitions, initial_distribution)
data = sequenceGenerator(100, true_hidden_transitions, true_emission_transitions, initial_distribution)

hidden_transitions = np.array(([0.2, 0.8],
                               [0.5, 0.5]))

#([0.2, 0.8],
#[0.3, 0.7]))
emission_transitions = np.array(([0.5, 0.5],
                                [0.3, 0.7]))

# ([0.5, 0.5],
# [0.3, 0.7])
print(single_baum_welch(data, hidden_transitions, emission_transitions, initial_distribution, 0.001))
object = baum_welch(sequence_data, hidden_transitions, emission_transitions, initial_distribution, 0.001)

print("\n")

print(object)

error = error_percentage(true_hidden_transitions, true_emission_transitions, object["hidden_transitions"],
                         object["emission_transitions"])
print("\n")

pprint.pprint(error)


