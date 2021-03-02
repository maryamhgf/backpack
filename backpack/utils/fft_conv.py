from functools import partial
from typing import Tuple, Union, Iterable

import torch
from torch import nn, Tensor
from torch.fft import rfftn, irfftn, fftn, ifftn
import torch.nn.functional as f

import time

def complex_matmul(a: Tensor, b: Tensor, groups: int = 1) -> Tensor:
    # n = a[0]
    n = a.shape[0]
    v = b.shape[0] // n
    b = b.reshape(v, n, b.shape[1], b.shape[2], b.shape[3])
    a = a.unsqueeze(1).unsqueeze(0)
    b = b.unsqueeze(3)
    # a = a.real
    # b = b.real
    # print('a,b,c:', a.shape, b.shape)
    c = a * b
    return c

def my_complex_matmul(a: Tensor, b: Tensor, groups: int = 1) -> Tensor:
    """Multiplies two complex-valued tensors."""
    scalar_matmul = partial(torch.einsum, "na...,nb...-> nab...")
    # a = a.view(a.size(0), -1, *a.shape[2:])
    # b = b.view( -1, *b.shape[1:])

    # Compute the real and imaginary parts independently, then manually insert them
    # into the output Tensor.  This is fairly hacky but necessary for PyTorch 1.7.0,
    # because Autograd is not enabled for complex matrix operations yet.  Not exactly
    # idiomatic PyTorch code, but it should work for all future versions (>= 1.7.0).
    # print('Line24: a shape, b shape:', a.shape, b.shape)
    real = scalar_matmul(a.real, b.real) - scalar_matmul(a.imag, b.imag)
    imag = scalar_matmul(a.imag, b.real) + scalar_matmul(a.real, b.imag)
    c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    c.real, c.imag = real, imag
    return c

def to_ntuple(val: Union[int, Iterable[int]], n: int) -> Tuple[int, ...]:
    """Casts to a tuple with length 'n'.  Useful for automatically computing the
    padding and stride for convolutions, where users may only provide an integer.

    Args:
        val: (Union[int, Iterable[int]]) Value to cast into a tuple.
        n: (int) Desired length of the tuple

    Returns:
        (Tuple[int, ...]) Tuple of length 'n'
    """
    if isinstance(val, Iterable):
        out = tuple(val)
        if len(out) == n:
            return out
        else:
            raise ValueError(f"Cannot cast tuple of length {len(out)} to length {n}.")
    else:
        return n * (val,)


def fft_conv(
    signal: Tensor,
    kernel: Tensor,
    bias: Tensor = None,
    padding: Union[int, Iterable[int]] = 0,
    stride: Union[int, Iterable[int]] = 1,
    groups: int = 1,
) -> Tensor:
    """Performs N-d convolution of Tensors using a fast fourier transform, which
    is very fast for large kernel sizes. Also, optionally adds a bias Tensor after
    the convolution (in order ot mimic the PyTorch direct convolution).

    Args:
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Tensor) Bias tensor to add to the output.
        padding: (Union[int, Iterable[int]) Number of zero samples to pad the
            input on the last dimension.
        stride: (Union[int, Iterable[int]) Stride size for computing output values.

    Returns:
        (Tensor) Convolved tensor
    """
    # Cast padding & stride to tuples.
    # st = time.time()
    # padding_ = to_ntuple(padding, n=signal.ndim - 2)
    # stride_ = to_ntuple(stride, n=signal.ndim - 2)

    padding_ = padding

    stride_ = (1, 1)
    # print('padding_:', padding_)
    # print('stride_:', stride_)
    # padding_time = time.time() - st
    # print('padding_time:', padding_time)
    # Pad the input signal & kernel tensors
    signal_padding = [p for p in padding_[::-1] for _ in range(2)]
    signal = f.pad(signal, signal_padding)
    
    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.
    if signal.size(-1) % 2 != 0:
        signal_ = f.pad(signal, [0, 1])
    else:
        signal_ = signal

   

    # st = time.time()
    kernel_padding = [
        pad
        for i in reversed(range(2, signal_.ndim))
        for pad in [0, signal_.size(i) - kernel.size(i)]
    ]
    # print(kernel_padding)
    # print(kernel.shape)
    padded_kernel = f.pad(kernel, kernel_padding)
    # padding_time_kernel = time.time() - st
    # print('padding_time_kernel:', padding_time_kernel)
    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    # signal_ = signal_.reshape(signal_.size(0), groups, -1, *signal_.shape[2:])
    
    # st = time.time()
    # signal_fr = rfftn(signal_, dim=tuple(range(2, signal.ndim)))
    # kernel_fr = rfftn(padded_kernel, dim=tuple(range(2, signal.ndim)))

    signal_fr = rfftn(signal_, dim=tuple(range(2, signal.ndim)))
    kernel_fr = rfftn(padded_kernel, dim=tuple(range(2, signal.ndim)))
    # rfft_time = time.time() - st
    # print('rfft_time:', rfft_time)
    # print('Line: padded signal shape:', signal_.shape)
    # print('Line: signal_ shape:', signal_.shape)
    # print('Line: padded_kernel shape:', padded_kernel.shape)
    # print('Line: kernel_fr shape:', kernel_fr.shape)

    # st = time.time()
    kernel_fr.imag *= -1

    # output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)/torch.numel(signal_fr[0,0,0,:])
    # print('KOOOOME:', output_fr.shape)
    # print('KOOOOME:', output_fr)
    # x = signal_fr[0,:]
    # print(x)
    # output_fr = my_complex_matmul(signal_fr, kernel_fr, groups=groups)/torch.numel(signal_fr[0,0,0,:])
    # output_fr = my_complex_matmul(signal_fr, kernel_fr, groups=groups)
    output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)
    output = output_fr
    # print('output.shape:', output.shape)

    # matmul_time = time.time() - st
    # print('matmul_time:', matmul_time)
    
    # st = time.time()
    output = irfftn(output_fr, dim=tuple(range(4, signal.ndim + 2)))
    # print('output irfftn .shape:', output.shape)
    # output = ifftn(output_fr, dim=tuple(range(3, signal.ndim+1)))
    # inverse_time = time.time() - st
    # print('inverse_time:', inverse_time)
    # st = time.time()
    # Remove extra padded values
    # print('signal, kernel, padding', signal.shape, kernel.shape, padding)
    crop_slices = [slice(0, output.size(0)), slice(0, output.size(1)), slice(0, output.size(2)), slice(0, output.size(3))] + [
         slice(padding_[i-3] - 1 , (signal.size(i - 1) - kernel.size(i - 1) - padding_[i-3] +2  ), stride_[i - 3])
        for i in range(3, signal.ndim + 1)
    ]
    # crop_slices = 
    # print('crop_slices:', crop_slices)
    # print('my output before croping:', output.shape)
    output = output[crop_slices].contiguous()
    # output = output[:,:,:,1:].contiguous()
    # print('output after crop:', output.shape)
    # print('output norm 2:', torch.norm(output))
    # crop_time = time.time() - st
    # print('crop_time:', crop_time)
    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output += bias.view(bias_shape)

    return output

