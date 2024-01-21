---
layout: post
title:  "When Fourier fails: Subtraction"
date:   2024-01-21
description: Learn how to interpolate non-periodic, uniform data using subtraction methods!
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<p class="intro"><span class="dropcap">I</span>n today's post, we study ways to accurately interpolate non-periodic, smooth data on a uniform grid. </p>


## Intro
This series of posts looks into different strategies for interpolating non-periodic, smooth data on a uniform grid with high accuracy. For an introduction, see the <a href="https://kunkelalexander.github.io/blog/when-fourier-fails-filters-post/">first post of this series </a>. In the following, we will look at subtraction methods.  You may find the accompanying <a href="https://github.com/KunkelAlexander/nonperiodicinterpolation-python"> Python code on GitHub </a>.


## Subtraction Methods
To use the Fourier transform with high accuracy, we need to find a way to make our data periodic. Subtraction methods achieve this by decomposing the data into a periodic and a non-periodic part. They first estimate the data's derivatives at the domain boundary, usually by employing finite difference approximations, and then find a suitable set of non-periodic functions that satisfies the estimated boundary conditions, usually by solving a linear system. These basis functions are then subtracted from the original data and leave a periodic function that allows for an accurate Fourier transform. At the same time, the basis functions can be manipulated analytically. Subtraction methods commonly subtract either polynomials or trigonometric functions. Their accuracy depends on being able to estimate the data's derivatives at the boundaries accurately. Since the Fourier transform exhibits $$n+1$$-th order convergence if the $$n$$-th derivative is continuous, satisfying $$m$$ boundary conditions with $$m$$ basis functions leads to $$m+1$$th order convergence. Finally, the maximum order of convergence is fundamentally limited by a variant of Runge's phenomenon again since it is impossible to estimate arbritrarily high-order derivatives using polynomial stencils.

In the following, I implement a subtraction method described in Matthew Green's M.Sc. thesis <a href="https://core.ac.uk/download/215443759.pdf"> Spectral Solution with a Subtraction Method to Improve Accuracy</a>. It estimates the boundary derivatives using finite-difference stencils and subtracts a inhomogeneous linear combination of cosine functions to leave a homogeneous remainder. This remainder can either be expanded using a cosine transform or less efficiently antisymmetrically expanded into a periodic function and then manipulated using a Fourier transform.
The following figure demonstrates the latter process using an exponential function.
<img src="{{ site.baseurl }}/assets/img/nonperiodicinterpolation-python/subtraction_1.png" alt="">
Computing derivatives by summing analytical derivatives of cosine functions with the appropriate coefficients and numerical derivatives of the Fourier transform, we see that while the first derivate can be computed very accurately without the need of a buffer zone, higher derivatives become less and less accurate.
<img src="{{ site.baseurl }}/assets/img/nonperiodicinterpolation-python/subtraction_2.png" alt="">

Finally,


{%- highlight python -%}import numpy as np
import scipy
import matplotlib.pyplot as plt

def forward_difference_matrix(order, dx):
    """
    Create a forward difference matrix of a given order and spacing.

    Args:
        order (int): The order of the matrix.
        dx (float): The spacing between points.

    Returns:
        numpy.ndarray: The forward difference matrix.
    """
    size = order
    mat = np.zeros((size, size))
    for k in range(1, size + 1):
        for j in range(1, size + 1):
            mat[j - 1, k - 1] = (j * dx) ** k / np.math.factorial(k)

    return mat

def backward_difference_matrix(order, dx):
    """
    Create a backward difference matrix of a given order and spacing.

    Args:
        order (int): The order of the matrix.
        dx (float): The spacing between points.

    Returns:
        numpy.ndarray: The backward difference matrix.
    """
    size = order
    mat = np.zeros((size, size))
    for k in range(1, size + 1):
        for j in range(1, size + 1):
            mat[j - 1, k - 1] = (-j * dx) ** k / np.math.factorial(k)

    return mat

def forward_difference_vector(order, f):
    """
    Create a forward difference vector for a given function and order.

    Args:
        order (int): The order of the difference.
        f (numpy.ndarray): The function values.

    Returns:
        numpy.ndarray: The forward difference vector.
    """
    diff = np.zeros(order)
    for j in range(1, order + 1):
        diff[j - 1] = f[j] - f[0]
    return diff

def backward_difference_vector(order, f):
    """
    Create a backward difference vector for a given function and order.

    Args:
        order (int): The order of the difference.
        f (numpy.ndarray): The function values.

    Returns:
        numpy.ndarray: The backward difference vector.
    """
    diff = np.zeros(order)
    for j in range(1, order + 1):
        diff[j - 1] = f[-1 - j] - f[-1]
    return diff

def iterative_refinement(A, b, tolerance=1e-9):
    """
    Solve a system of linear equations Ax = b using iterative refinement.

    Args:
        A (numpy.ndarray): The matrix A.
        b (numpy.ndarray): The vector b.
        tolerance (float): The tolerance for the residual error.

    Returns:
        numpy.ndarray: The solution vector x.
    """
    x = np.linalg.solve(A, b)
    residual = b - A @ x
    residual_error = np.sum(np.abs(residual))

    iteration = 0
    while residual_error > tolerance:
        correction = np.linalg.solve(A, residual)
        x += correction
        residual = b - A @ x
        residual_error = np.sum(np.abs(residual))
        iteration += 1
        if iteration > 100:
            break

    return x

def shift_x(x):
    """
    Normalize the x values to a range of [0, 1].

    Args:
        x (numpy.ndarray): The input x values.

    Returns:
        numpy.ndarray: The normalized x values.
    """
    return (x - x[0]) / (x[-1] - x[0])

def cosine_difference_vector(order, f, Dl, Dr):
    """
    Create a cosine difference vector for a given function and order.

    Args:
        order (int): The order of the difference.
        f (numpy.ndarray): The function values.
        Dl (numpy.ndarray): The left difference values.
        Dr (numpy.ndarray): The right difference values.

    Returns:
        numpy.ndarray: The cosine difference vector.
    """
    b = np.zeros(2 * order)
    b[0] = f[0]
    b[1] = f[-1]
    for i in range(1, order):
        b[i * 2] = Dl[i] / (np.pi) ** (2 * i)

    for i in range(1, order):
        b[i * 2 + 1] = Dr[i] / (np.pi) ** (2 * i)

    return b

def cosine_difference_matrix(order):
    """
    Create a cosine difference matrix of a given order.

    Args:
        order (int): The order of the matrix.

    Returns:
        numpy.ndarray: The cosine difference matrix.
    """
    A = np.zeros((order * 2, order * 2))
    for i in range(order):
        derivative = 2 * i
        for j in range(1, 2 * order + 1):
            A[2 * i, j - 1] = j ** derivative * (-1) ** i
            A[2 * i + 1, j - 1] = j ** derivative * (-1) ** i * (-1) ** j

    return A

def reconstruct(C, x, derivative_order=0):
    """
    Reconstruct a function from its cosine series coefficients.

    Args:
        C (numpy.ndarray): The cosine series coefficients.
        x (numpy.ndarray): The x values.
        derivative_order (int): The order of the derivative to reconstruct.

    Returns:
        numpy.ndarray: The reconstructed function values.
    """
    f = np.zeros(x.shape)
    L = x[-1] - x[0]
    x_eval = shift_x(x)
    for k in range(1, len(C) + 1):
        f += C[k - 1] * np.real((1j * k * np.pi / L) ** derivative_order * np.exp(1j * k * np.pi * x_eval))

    return f

def get_shift_function(f, order, x):
    """
    Calculate the shift function for a given function, order, and x values.

    Args:
        f (numpy.ndarray): The function values.
        order (int): The order of the shift function.
        x (numpy.ndarray): The x values.

    Returns:
        tuple: The shift function values and the coefficients.
    """
    x_eval = shift_x(x)
    dx = x_eval[1] - x_eval[0]
    A = forward_difference_matrix(order, dx)
    b = forward_difference_vector(order, f)
    Dl = iterative_refinement(A, b)

    A = backward_difference_matrix(order, dx)
    b = backward_difference_vector(order, f)

    Dr = iterative_refinement(A, b)

    A = cosine_difference_matrix(int(order / 2) + 1)
    b = cosine_difference_vector(int(order / 2) + 1, f, Dl, Dr)
    C = iterative_refinement(A, b)

    shift = reconstruct(C, x_eval)
    return shift, C

def antisymmetric_extension(f):
    """
    Extend a function with its antisymmetric part.

    Args:
        f (numpy.ndarray): The function values.

    Returns:
        numpy.ndarray: The extended function values.
    """
    f_ext = np.concatenate([f, -np.flip(f)[1:-1]])
    return f_ext

def get_k(p, dx):
    """
    Calculate the k values for a given array and spacing.

    Args:
        p (numpy.ndarray): The input array.
        dx (float): The spacing between points.

    Returns:
        numpy.ndarray: The k values.
    """
    N = len(p)
    L = N * dx
    k = 2 * np.pi / L * np.arange(-N / 2, N / 2)
    return np.fft.ifftshift(k)



# Configurable parameters
N = 100  # Size of input domain
order = 5  # Order of the subtraction variables

# Define the domain and the function
L = np.pi
x = np.linspace(0, L, N)
def func(x):
    return np.exp(x)
f = func(x)
dx = x[1] - x[0]

# Get the shift function and coefficients
shift, C = get_shift_function(f, order, x)
hom      = f - shift
f_ext    = antisymmetric_extension(hom)
f_hat    = scipy.fft.fft(f_ext)

# Get the k values
k = get_k(f_hat, dx)


colors = [
    '#08F7FE',  # teal/cyan
    '#FE53BB',  # pink
    '#F5D300',  # yellow
    '#00ff41', # matrix green
]

plt.style.use('dark_background')
plt.figure(figsize=(8, 3), dpi=200)
plt.axis("off")
plt.plot(f, c = colors[0], label=r"$f(x) = e^x = f_{hom} + f_{inhom}$")
plt.plot(hom, c = colors[1], label=r"$f_{hom}$")
plt.plot(shift, c = colors[2], label=r"$f_{inhom}$")
plt.axvline(len(x), ls="dashed", c="w")
plt.plot(f_ext, c = colors[1], ls="dashed", label="Antisymmetric extension")
plt.legend()
plt.savefig("figures/subtraction_1.png")
plt.show()

# Number of subplots
num_subplots = 3

# Create subplots
fig, axs = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5), dpi=200)

# Loop through different subtraction orders
for i, o in enumerate([1, 4, 6]):
    forg = func(x)
    frec = scipy.fft.ifft(f_hat * (1j * k) ** o).real[:N]  # Use only the first N elements (the original domain)
    reco = reconstruct(C, x, derivative_order = o)
    sumo = frec + reco


    # Plot the sum and the original function in subplots
    axs[i].set_title(f"Derivative order {o} with L1 error {np.mean(np.abs(sumo - forg)):3.3e}")
    axs[i].plot(x, sumo, label="Reconstructed Derivative", c = colors[0])
    axs[i].plot(x, forg, label="Original", c = colors[1])
    axs[i].legend()

# Adjust layout
plt.tight_layout()
plt.savefig("figures/subtraction_2.png")
plt.show()

{%- endhighlight -%}

When the spectrum is filtered with a constant function, the decay of the Fourier coefficients is not modified. Accordingly, we can observe Gibb's phenomenon.
<img src="{{ site.baseurl }}/assets/img/nonperiodicinterpolation-python/filter_1.png" alt="">

Filtering with a smoothly decaying filter function significantly reduces oscillations and increases the accuracy of the reconstruction.
<img src="{{ site.baseurl }}/assets/img/nonperiodicinterpolation-python/filter_2.png" alt="">




[runge-wiki]: https://en.wikipedia.org/wiki/Runge's_phenomenon
[gibbs-wiki]: https://en.wikipedia.org/wiki/Gibbs_phenomenon
[boyd-cheby]: https://depts.washington.edu/ph506/Boyd.pdf