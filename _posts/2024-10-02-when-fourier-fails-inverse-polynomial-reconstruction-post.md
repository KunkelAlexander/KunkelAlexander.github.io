---
layout: post
title:  "When Fourier fails: Inverse Polynomial Reconstruction"
date:   2024-02-10
description: Learn more how to change to an exponentially accurate function basis.
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<p class="intro"><span class="dropcap">I</span>n today's post, we study the truncated Inverse Polynomial Reconstruction method. </p>


## Intro
This series of posts looks into different strategies for interpolating non-periodic, smooth data on a uniform grid with high accuracy. For an introduction, see the <a href="https://kunkelalexander.github.io/blog/when-fourier-fails-filters-post/">first post of this series</a>. In this post, we delve into a very interesting topic: Inverse Polynomial Reconstruction (IPR). The method presented here is described in Jung's and Shizgal's paper <a href="https://www.sciencedirect.com/science/article/abs/pii/S0021999107000332"> On the numerical convergence with the inverse polynomial reconstruction method for the resolution of the Gibbs phenomenon </a>. The logic of this method is as follows: We compute the Fourier extension of a non-periodic function and then change to a basis with better convergence properties. The result is a highly accurate reconstruction of the original function in a polynomial basis on a uniform grid. This result is surprising since direct polynomial expansion leads to the Runge instability. The idea to restore convergence of the Fourier series in a different basis has been pioneered by <a href="https://www.sciencedirect.com/science/article/pii/0377042792902605"> Gottlieb et al. using Gegenbauer polynomials</a>. However, there is an important difference between the Gegenbauer and IPR methods. While the Gegenbauer method achieve stability by projection of the Fourier basis onto a polynomial basis spanning a smaller subspace, the IPR method computes an invertible change-of-base matrix between arbitrary functional bases. Convergence is independent of the polynomial basis used. Stability is remedied by the truncation suggested by Jung et al. which is effectively the truncation the Gegenbauer method started with. I only present IPR theory since it supersedes Gegenbauer methods according my understanding. You may find the accompanying <a href="https://github.com/KunkelAlexander/when-fourier-fails-python"> Python code on GitHub </a>.


## Change-of-basis matrix
Let us study the Fourier transform of $$f(x) = \exp(x)$$ on $$[-1, 1]$$.

{%- highlight python -%}
import numpy as np
import scipy
import matplotlib.pyplot as plt
N   = 100
x   = np.linspace(-1, 1, N)
f   = np.exp(x)
f_k = scipy.fft.fft(f)
{%- endhighlight -%}

The Fourier coefficients $$\hat{f}_k$$ of a discrete Fourier series $$f_N(x) = \sum_{k=-N}^{N} \hat{f}_k \exp(i k \pi x)$$ can be defined via the Fourier inner product

$$ \hat{f}_k \equiv (f(x), \exp(i k \pi x))_F = \frac{1}{2} \int_{-1}^1 f(x) \exp(-i\pi x k) \mathrm{d}x $$

This expression, the Fourier transform of $$f(x)$$ can be understood as $$ \hat{f}_k $$ being the projection of $$f(x)$$ on the $$k$$-th basis element of the Fourier basis. Let us introduce a different basis set $$\{\phi_l(x)|l = 0, ..., m}$$ and choose $$\phi_l$$ to be the Chebyshev polynomials. We can compute the change-of-basis matrix $$\mathb{W}$$ from the Chebyshev basis to the Fourier basis using the Fourier transform as

$$W_{kl} = (\phi_l(x), \exp(i k \pi x))_F = \frac{1}{2} \int_{-1}^1 \phi_l(x) \exp(-i\pi x k) \mathrm{d}x $$

{%- highlight python -%}
W = np.zeros((N, N), dtype=complex)

for l in range(N):
    W[:, l] = scipy.fft.fft(scipy.special.chebyt(l)(x))
{%- endhighlight -%}

Assuming that $$\mathb{W}$$ is a non-singular matrix, we can invert it, multiply it with $$f_k$$ and reconstruct $$f(x)$$ by summing the Chebyshev polynomials

{%- highlight python -%}
W_inv = np.linalg.inv(W)
g     = W_inv @ f_k

f_rec = np.poly1d([])
for l, coeff in enumerate(g):
    f_rec += coeff * scipy.special.chebyt(l)


fig, axs = plt.subplots(figsize=(5 , 3), dpi=200)
plt.style.use('dark_background')
plt.plot(x, f, label = r"$f(x)$", c = '#08F7FE')
plt.plot(x, f_rec(x), label = r"Reconstruction of $f(x)$", c = '#FE53BB')
{%- endhighlight -%}

The result looks as follows:
<img src="{{ site.baseurl }}/assets/img/nonperiodicinterpolation-python/ipr_failure.png" alt="">

What went wrong?