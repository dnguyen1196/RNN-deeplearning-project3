"""
Helper functions for getting parameters from RNN models (the basic one and the one with GRUs). The tensorflow session runs the parameter
matrices within RNN cells and the dispatch their values to corresponding weights or biases for the calculation of gates and hidden states. 

The setting procedure is the inverse of the getting procedure. It allows updating model parameters. 
"""

import numpy as np


def get_rnn_params(rnn_cell, session):
    """Get parameters from an RNN cell
    
    inputs: 
        session: the tf session
        rnn: the tensorflow cell
    outputs: 
        wt_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation
        wt_x: shape [input_size, hidden_size], weight matrix for input transformation
        bias: shape [hidden_size], bias term
    """
    
    
    # get the parameters of the RNN cell
    weights, bias = session.run(rnn_cell.weights)

    input_size = weights.shape[0] - weights.shape[1]

    wt_x = weights[:input_size, :]
    wt_h = weights[input_size:, :]
    bias = bias
    
    return wt_h, wt_x, bias

def get_gru_params(gru_cell, session):
    """Get parameters from a GRU cell
    
    inputs: 
        session: the tf session
        rnn: the tensorflow cell
    outputs: 
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
    
    """

    wt_g, bias_g, wt_c, biasc = session.run(gru_cell.weights)

    input_size = wt_c.shape[0] - wt_c.shape[1]
    hidden_size = wt_c.shape[1]

    # The weights and biases for two gates are concatenated into one matrix. 
    # Separate parameters. The first `hidden_size` columns/elements are for r gate, and the second `hidden_size`
    # columns are for u gate
    wtr_x = wt_g[:input_size, :hidden_size]
    wtr_h = wt_g[input_size:, :hidden_size]
    biasr = bias_g[:hidden_size]

    wtu_x = wt_g[:input_size, hidden_size:]
    wtu_h = wt_g[input_size:, hidden_size:]
    biasu = bias_g[hidden_size:]

    wtc_x = wt_c[:input_size]
    wtc_h = wt_c[input_size:]
    biasc = biasc
    
    return wtu_h, wtu_x, biasu, wtr_h, wtr_x, biasr, wtc_h, wtc_x, biasc



def set_rnn_params(rnn_cell, session, wt_h=None, wt_x=None, bias=None):
    """Set parameters to an RNN cell
    
    inputs: 
        rnn_cell: the tensorflow cell
        session: the tf session
        wt_h: shape [hidden_size, hidden_size] or None, weight matrix for hidden state transformation
        wt_x: shape [input_size, hidden_size] or None, weight matrix for input transformation
        bias: shape [hidden_size] or None, bias term
    """
    
    
    # get the parameters of the RNN cell
    weights = session.run(rnn_cell.weights[0])
    hidden_size = weights.shape[1]
    input_size = weights.shape[0] - hidden_size
    
    # set a value if it is not None
    if wt_x is not None:
        weights[:input_size] = wt_x
    
    if wt_h is not None:
        weights[input_size:] = wt_h
    
    session.run(rnn_cell.weights[0].assign(np.concatenate([wt_x, wt_h])))
    
    if bias is not None:
        session.run(rnn_cell.weights[1].assign(bias))
    
    return


def set_gru_params(gru_cell, session, wtu_h=None, wtu_x=None, biasu=None, 
                                          wtr_h=None, wtr_x=None, biasr=None, 
                                          wtc_h=None, wtc_x=None, biasc=None):
    """Set parameters to a GRU  cell
    
    inputs: 
        session: the tf session
        rnn: the tensorflow cell
        wtu_h: shape [hidden_size, hidden_size] or None, weight matrix for hidden state transformation for u gate
        wtu_x: shape [input_size, hidden_size] or None, weight matrix for input transformation for u gate
        biasu: shape [hidden_size] or None, bias term for u gate
        wtr_h: shape [hidden_size, hidden_size] or None, weight matrix for hidden state transformation for r gate
        wtr_x: shape [input_size, hidden_size] or None, weight matrix for input transformation for r gate
        biasr: shape [hidden_size] or None, bias term for r gate
        wtc_h: shape [hidden_size, hidden_size] or None, weight matrix for hidden state transformation for candicate
               hidden state calculation
        wtc_x: shape [input_size, hidden_size] or None, weight matrix for input transformation for candicate
               hidden state calculation
        biasc: shape [hidden_size] or None, bias term for candicate hidden state calculation
    """
    
    weights_g = session.run(gru_cell.weights[0])
    bias_g = session.run(gru_cell.weights[1])
    weights_c = session.run(gru_cell.weights[2])
    bias_c = session.run(gru_cell.weights[3])
    
    hidden_size = weights_c.shape[1]
    input_size = weights_c.shape[0] - hidden_size
    
    if wtr_x is not None:
        weights_g[:input_size, :hidden_size] = wtr_x
    if wtr_h is not None:
        weights_g[input_size:(input_size + hidden_size), :hidden_size] = wtr_h
    if biasr is not None:
        bias_g[:hidden_size] = biasr
    
    if wtu_x is not None:
        weights_g[:input_size, hidden_size:] = wtu_x
    if wtu_h is not None:
        weights_g[input_size:(input_size + hidden_size), hidden_size:] = wtu_h
    if biasu is not None:
        bias_g[hidden_size:] = biasu
    
    if wtc_x is not None:
        weights_c[:input_size] = wtc_x
    if wtc_h is not None:
        weights_c[input_size:(input_size + hidden_size)] = wtc_h
    if biasc is not None:
        bias_c = biasc
    
    
    session.run(gru_cell.weights[0].assign(weights_g))
    session.run(gru_cell.weights[1].assign(bias_g))
    
    session.run(gru_cell.weights[2].assign(weights_c))
    session.run(gru_cell.weights[3].assign(bias_c))
    
    return


