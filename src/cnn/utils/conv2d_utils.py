
from typing import Tuple

import cupy as cp


def get_im2col_idx(
    array_shape,
    filter_dim,
    pad,
    stride
):

    _, c, h_in, w_in = array_shape
    h_f, w_f = filter_dim

    h_out = (h_in + 2 * pad - h_f) // stride + 1
    w_out = (w_in + 2 * pad - w_f) // stride + 1

    i0 = cp.repeat(cp.arange(h_f), w_f)
    i0 = cp.tile(i0, c)
    i1 = stride * cp.repeat(cp.arange(h_out), w_out)
    j0 = cp.tile(cp.arange(w_f), h_f * c)
    j1 = stride * cp.tile(cp.arange(w_out), h_out)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = cp.repeat(cp.arange(c), h_f * w_f).reshape(-1, 1)
    return k, i, j


def im2col(
    array,
    filter_dim,
    pad=0,
    stride=1
):

    _, c, _, _ = array.shape
    h_f, w_f = filter_dim
    array_pad = cp.pad(
        array=array,
        pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode='constant'
    )
    k, i, j = get_im2col_idx(
        array_shape=array.shape,
        filter_dim=filter_dim,
        pad=pad,
        stride=stride
    )
    cols = array_pad[:, k, i, j]
    return cols.transpose(1, 2, 0).reshape(h_f * w_f * c, -1)


def col2im(
    cols,
    array_shape,
    filter_dim,
    pad=0,
    stride=1
):

    n, c, h_in, w_in = array_shape
    h_f, w_f = filter_dim
    h_pad, w_pad = h_in + 2 * pad, w_in + 2 * pad
    array_pad = cp.zeros((n, c, h_pad, w_pad), dtype=cols.dtype)
    k, i, j = get_im2col_idx(
        array_shape=array_shape,
        filter_dim=filter_dim,
        pad=pad,
        stride=stride
    )
    cols_reshaped = cols.reshape(c * h_f * w_f, -1, n)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    array_pad[slice(None), k, i, j] = cp.add(array_pad[slice(None), k, i, j], array_pad[slice(None), k, i, j]) + cols_reshaped
    return array_pad[:, :, pad:pad+h_in, pad:pad+w_in]
