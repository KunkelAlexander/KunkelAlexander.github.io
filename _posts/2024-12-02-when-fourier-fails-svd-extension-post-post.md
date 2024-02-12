---
layout: post
title:  "When Fourier fails: SVDs and Fourier extensions of the third kind"
date:   2024-02-11
description: Learn about the magic of high-precision extensions using SVDs.
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<p class="intro"><span class="dropcap">I</span>n today's post, we study the Fourier extensions of the third kind, my personal favourite. </p>


## Intro
This series of posts looks into different strategies for interpolating non-periodic, smooth data on a uniform grid with high accuracy. For an introduction, see the <a href="https://kunkelalexander.github.io/blog/when-fourier-fails-filters-post/">first post of this series</a>. In this post, we delve into Fourier extensions using truncated singular value decompositions. The method presented here is known as Fourier extension of the third kind as described in Boyd's insightful paper <a href="https://www.sciencedirect.com/science/article/abs/pii/S0021999102970233"> A Comparison of Numerical Algorithms for Fourier Extension of the First, Second, and Third Kinds </a>.
SVD extensions are particularly simple and beautiful: Instead of aiming to find a periodic extension and then Fourier transform, one instead solves a linear system whose solutions are the Fourier coeffients. You may find the accompanying <a href="https://github.com/KunkelAlexander/when-fourier-fails-python"> Python code on GitHub </a>. Can you guess the definition of the linear operator from looking at its matrix representation?
<img src="{{ site.baseurl }}/assets/img/nonperiodicinterpolation-python/svd_W.png" alt="">

## Fourier Physical Interval Collocation—Spectral Coefficients as the Unknowns (FPIC-SU)
The search for a suitable Fourier extension can be expressed as a minimisation problem:
$$\min_{a_k \forall k \in t(m)} \sum^{n-1}_{j=0} \left| \sum_{k\in t(m)} a_k e^{\frac{2 \pi i}{b} k x_j} - f(x_j)\right|^2 = \min_{\hat{x}} \sum^{n-1}_{j=0} \left|A\hat{x} - b\right|^2$$


Let $$f(x)$$ be a function that is symmetrix w.r.t. $$x=0$$ and defined on the positive interval $$[0, \chi]$$. This assumption can be satisfied for an arbitrary function $$g(x)$$ by splitting it into its symmetric and antisymmetric parts as $$S(x) \equiv g(x)/2 + g(-x)/2$$ and $$A(x) \equiv g(x)/2 - g(-x)/2$$. Let $$\hat{f}(x)$$ be the desired extension of $$f(x)$$ into the interval $$[0, \theta]$$ with $$\theta > \chi$$. Since $$\hat{f}(x)$$ is also symmetric w.r.t. $$x=0$$, it allows for an expansion in terms of cosine functions.
\item Optimise linear, under-determined least squares problem via SVD $A = U \Sigma V^T$
\item Drawback: Cost for SVD of $n\times m$ matrix $\mathcal{O}(\min(m, n)^2 \cdot \max(m, n))$
\end{itemize}

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

In this expression, the Fourier transform of $$f(x)$$ can be understood as $$\hat{f}_k$$ being the projection of $$f(x)$$ onto the $$k$$-th basis element of the Fourier basis. Let us introduce a different basis set $$\{\phi_l(x), l = 0, ..., m\}$$ and choose $$\phi_l$$ to be the Chebyshev polynomials. We can compute the change-of-basis matrix $$W$$ from the Chebyshev basis to the Fourier basis using the Fourier transform as

$$W_{kl} = (\phi_l(x), \exp(i k \pi x))_F = \frac{1}{2} \int_{-1}^1 \phi_l(x) \exp(-i\pi x k) \mathrm{d}x $$

{%- highlight python -%}
W = np.zeros((N, N), dtype=complex)

for l in range(N):
    W[:, l] = scipy.fft.fft(scipy.special.chebyt(l)(x))
{%- endhighlight -%}

Assuming that $$W$$ is a non-singular matrix, we can invert it, multiply it with $$f_k$$ and reconstruct $$f(x)$$ by summing the Chebyshev polynomials

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
What went wrong? The reconstruction is clearly divergent and the problem lies in the conditioning of $$W$$, as is confirmed by plotting its singular values:
<img src="{{ site.baseurl }}/assets/img/nonperiodicinterpolation-python/ipr_svd.png" alt="">
Its condition number in this case is $$\mathrm{cond}(W) = 10^{16}$$. Fortunately for us, Jung's paper proposes a solution: A projection to  a polynomial subspace performed via Gaussian elimination with a truncation:

{%- highlight python -%}# LU decomposition with pivot
pivot, L, U = scipy.linalg.lu(W, permute_l=False)

# forward substitution to solve for L x y = f_k
y = np.zeros(f_k.size, dtype=complex)
for m, b in enumerate((pivot.T @ f_k).flatten()):
    y[m] = b
    # skip for loop if m == 0
    if m:
        for n in range(m):
            y[m] -= y[n] * L[m,n]
    y[m] /= L[m, m]

# truncation for IPR
c = np.abs(y) < 1000 * np.finfo(float).eps
y[c] = 0

# backward substitution to solve for y = U x
g = np.zeros(f_k.size, dtype=complex)
lastidx = f_k.size - 1  # last index
for midx in range(f_k.size):
    m = f_k.size - 1 - midx  # backwards index
    g[m] = y[m]
    if midx:
        for nidx in range(midx):
            n = f_k.size - 1  - nidx
            g[m] -= g[n] * U[m,n]
    g[m] /= U[m, m]


f_rec = np.poly1d([])
for l, coeff in enumerate(g):
    f_rec += coeff * scipy.special.chebyt(l)

fig, axs = plt.subplots(figsize=(5 , 3), dpi=200)
plt.style.use('dark_background')
plt.plot(x, f, label = r"$f(x)$", c = '#08F7FE')
plt.plot(x, f_rec(x), label = r"Reconstruction of $f(x)$", c = '#FE53BB')
plt.xlabel(r"$x$")
plt.ylabel(r"$f(x)$")
plt.legend()
plt.tight_layout()
plt.savefig("figures/ipr_success.png", bbox_inches='tight')
plt.show()
{%- endhighlight -%}

The truncated reconstruction beautifully converges. The only downside of this method is that the truncation threshold needs to be empirically determined and should be set high enough to ensure stability.
<img src="{{ site.baseurl }}/assets/img/nonperiodicinterpolation-python/ipr_success.png" alt="">

### Accuracy
The accuracy of the truncated IPR is very high and can reach machine precision. The first plot in the series demonstrates that despite a high precision, the IPR shares a feature that I have observed in all spectral reconstructions of non-periodic data: The errors close to the discontinuity at the domain boundaries is orders of magnitude higher than in the domain center. Whatever the approach, having a ghost boundary that can be discarded helps to achieve high accuracy. The second plot shows that even an $$11$$th order derivative can be calculated within $$0.1$$% error. From my experience, the IPR has very good convergence properties for large enough $$N$$. However, for low-resolution data, the error of the reconstruction is unbounded which makes the IPR algorithm unsuitable for the interpolation of low-resolution data with high accuracy.

<img src="{{ site.baseurl }}/assets/img/nonperiodicinterpolation-python/ipr_accuracy.png" alt="">