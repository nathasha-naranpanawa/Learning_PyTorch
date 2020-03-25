"""
PyTorch is a Python-based scientific computing package targetd at
-a replacement for NumPy to use the power of GPUs
-a deep learnign research platform that provides maximum flexibility and speed

---------------------
Getting Started

Tensors

Tensors are similar to numpy's ndarrays. But Tensors can also be used on a GPU to accelerate computing.
"""

from __future__ import print_function
import torch
import numpy as np

# Constructing an uninitialized matrix
# An uninitialized matrix is declared, but does not contain definite known values before it is used.
# When it is created, whatever values were in the allocated memory at the time will appear as initial values.
x = torch.empty(5, 3)   # a 5x3 matrix

# Construct a randomly initialized matrix
x = torch.rand(5, 3)

# Construct a tensor directly from data
x = torch.tensor([5.5, 3])

# create a tensor board using an existing tensor. These methods will reuse properties of the input tensor unless new values are provided by the user
x = x.new_ones(5, 3, dtype=torch.double)    # new_* methods take in sizes
# print(x)

x = torch.randn_like(x, dtype=torch.float)  # override dtype!
# print(x)                                    # result has the same size

# get the size
# print(x.size()) # torch.size is a tuple, so it supports all tuple operations

"""
-------------------
Operations

There are multiple syntaxes for operations

"""
# Addition: syntax 1
y = torch.rand(5,3)
# print(x+y)

# Addiiton: syntax 2
# print(torch.add(x, y))

# Addition: providing an output tensor as argument
result = torch.empty(5, 3)
torch.add(x, y, out=result)
# print(result)

# Addition in place
# adds x to y
y.add_(x)   # any operation that mutates a tensor in-place is post-fixed with an _. ex: x.copy_t(), x.t_() will change x
# print(y)


"""
"""
# using numpy-like indexing
# print(x[:,1])

# resizing/reshaping tensor
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8) # the size -1 is inferred from other dimensions
# print(x.size(), y.size(), z.size())

# if you have a one element tensor, use .item() to get the value as a Python number
x = torch.randn(1)
# print(x)
# print(x.item())

"""
-------
NumPy Bridge

Converting a torch tensor to a NumPy array and vice versa
The Torch Tensor and NumPy array will share their underlying memory locations (if the Torch Tensor is on CPU), and changing one will change the other.

"""

# Converting a torch tensor to a NumPy array
a = torch.ones(5)
# print(a)

b = a.numpy()
# print(b)

# to see how the numpy array changed in value
a.add_(1)
# print(a)
# print(b)

# converting NumPy array to Torch tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# All the Tensors on the CPU except a CharTensor support converting to NumPy and back.

"""
CUDA tensors

Tensors can be moved onto any device using the .to method.

"""

# let us run this cell only if CUDA is available
# we will use the 'torch.device' objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")           # a CUDA device object
    y = torch.ones_like(x, device=device)   # directly create a tensor on GPU
    x = x.to(device)                        # or just use strings ''.to("cuda")''
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))