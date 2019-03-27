"""
In this file, you should implement the forward calculation of the basic RNN model and the RNN model with GRUs. 
Please use the provided interface. The arguments are explained in the documentation of the two functions.
"""

import numpy as np
from scipy.special import expit as sigmoid

def rnn(wt_h, wt_x, bias, init_state, input_data):
    """
    RNN forward calculation.
    inputs:
        wt_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation
        wt_x: shape [input_size, hidden_size], weight matrix for input transformation
        bias: shape [hidden_size], bias term
        init_state: shape [batch_size, hidden_size], the initial state of the RNN
        input_data: shape [batch_size, time_steps, input_size], input data of `batch_size` sequences, each of
                    which has length `time_steps` and `input_size` features at each time step. 
    outputs:
        outputs: shape [batch_size, time_steps, hidden_size], outputs along the sequence. The output at each 
                 time step is exactly the hidden state
        final_state: the final hidden state
    """
    N, T, input_size = input_data.shape
    input_size_, hidden_size = wt_x.shape
    assert(input_size_ == input_size)
    hidden_size_, hidden_size__ = wt_h.shape
    assert(hidden_size ==  hidden_size_)
    assert(hidden_size_ == hidden_size__)
    assert(init_state.shape == (N, hidden_size))

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    outputs = np.zeros((N, T, hidden_size))
    hidden  = init_state
    # Input layer
    for i in range(N):
        hidden = np.copy(init_state[i, :])
        assert(hidden.shape == (hidden_size,))
        X      = input_data[i, :, :] # shape = (timesteps, input_size)
        assert(X.shape == (T, input_size))

        for t in range(T):
            Xt     = X[t, :]
            # temp1  = np.dot(np.transpose(hidden), wt_h) # 
            temp1  = np.dot(np.transpose(wt_h), hidden)
            # temp2  = np.dot(np.transpose(X[t, :]), wt_x) # 
            temp2  = np.dot(np.transpose(wt_x), Xt)

            assert(temp1.shape == (hidden_size,))
            assert(temp2.shape == (hidden_size,))

            hidden = temp1 + temp2 + bias
            assert(hidden.shape == (hidden_size,))

            hidden = np.tanh(hidden)
            # outputs[i, 0, :] = np.copy(np.dot(np.transpose(wt_h), hidden))
            outputs[i, 0, :] = np.copy(hidden)
    
    final_state = hidden
    
    ##################################################################################################
    # Please implement the basic RNN here. You don't need to considier computational efficiency.     #
    ##################################################################################################

    return outputs, final_state


def gru(wtu_h, wtu_x, biasu, wtr_h, wtr_x, biasr, wtc_h, wtc_x, biasc, init_state, input_data):
    """
    RNN forward calculation.

    inputs:
        wtu_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation for u gate
        wtu_x: shape [input_size, hidden_size], weight matrix for input transformation for u gate
        biasu: shape [hidden_size], bias term for u gate

        wtr_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation for r gate
        wtr_x: shape [input_size, hidden_size], weight matrix for input transformation for r gate
        biasr: shape [hidden_size], bias term for r gate
        
        wtc_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation for candicate
               hidden state calculation
        wtc_x: shape [input_size, hidden_size], weight matrix for input transformation for candicate
               hidden state calculation
        biasc: shape [hidden_size], bias term for candicate hidden state calculation

        init_state: shape [batch_size, hidden_size], the initial state of the RNN
        input_data: shape [batch_size, time_steps, input_size], input data of `batch_size` sequences, each of
                    which has length `time_steps` and `input_size` features at each time step. 
    outputs:
        outputs: shape [batch_size, time_steps, hidden_size], outputs along the sequence. The output at each 
                 time step is exactly the hidden state
        final_state: the final hidden state
    """
    N, T, input_size = input_data.shape
    input_size_, hidden_size = wtu_x.shape
    assert(input_size_ == input_size)
    hidden_size_, hidden_size__ = wtu_h.shape
    assert(hidden_size ==  hidden_size_)
    assert(hidden_size_ == hidden_size__)

    outputs = np.zeros((N, T, hidden_size))

    ##################################################################################################
    # Please implement an RNN with GRU here. You don't need to considier computational efficiency.   #
    ##################################################################################################
    for i in range(N):
        X = input_data[i, :, :]
        hidden  = init_state[i, :]
        assert(X.shape == (T,input_size))
        assert(hidden.shape == (hidden_size,))

        for t in range(T):
            xt = X[t, :]
            # Compute u gate (update gate) on input
            # u = np.dot(wtu_x, X[t, :]) + np.dot(wtu_h, hidden) + biasu
            # u = np.dot(np.transpose(X[t, :]), wtu_x) + np.dot(np.transpose(hidden),wtu_h) + biasu
            u = np.dot(np.transpose(wtu_x), xt) + np.dot(np.transpose(wtu_h),hidden) + biasu
            u = np.tanh(u)
            assert(u.shape == (hidden_size,))

            # Compute r gate (relevance gate)
            # r = np.dot(wtr_x, X[t, :]) + np.dot(wtr_h, hidden) + biasr
            # r = np.dot(np.transpose(X[t, :]), wtr_x) + np.dot(np.transpose(hidden), wtr_h) + biasr
            r = np.dot(np.transpose(wtr_x), xt) + np.dot(np.transpose(wtr_h), hidden) + biasr
            r = np.tanh(r)
            assert(r.shape == (hidden_size,))

            # Compute c
            # c = np.dot(wtc_x, X[t, :]) + np.dot(wtc_h, r * hidden) + biasc
            # c = np.dot(np.transpose(X[t, :]), wtc_x) + np.dot(np.transpose(r*hidden), wtc_h) + biasc
            c = np.dot(np.transpose(wtc_x), xt) + np.dot(np.transpose(wtc_h), r*hidden) + biasc
            c = np.tanh(c)
            assert(c.shape == (hidden_size,))

            # hidden = u * c + (1-u) * hidden
            hidden =  u * hidden + (1-u) * c

            assert(hidden.shape == (hidden_size,))
            outputs[i, t, :] = np.copy(hidden)

    final_state = hidden
    
    return outputs, final_state

