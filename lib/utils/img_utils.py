import numpy as np


def horizon_concate(inp0, inp1):
    h0, w0 = inp0.shape[:2]
    h1, w1 = inp1.shape[:2]
    if inp0.ndim == 3:
        inp = np.zeros((max(h0, h1), w0 + w1, 3), dtype=inp0.dtype)
        inp[:h0, :w0, :] = inp0
        inp[:h1, w0:(w0 + w1), :] = inp1
    else:
        inp = np.zeros((max(h0, h1), w0 + w1), dtype=inp0.dtype)
        inp[:h0, :w0] = inp0
        inp[:h1, w0:(w0 + w1)] = inp1
    return inp