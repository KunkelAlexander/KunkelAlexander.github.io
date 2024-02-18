---
layout: post
title:  "When Fourier fails: Gram-Fourier extensions"
date:   2024-02-18
description: One extension to rule them all. How to efficiently reuse accurate SVD extensions.
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<p class="intro"><span class="dropcap">I</span>n today's post, we study the Gram-Fourier extensions. They are a cross between SVD and polynomial expansions and offer high accuracy and stability at the asymptotic cost of a single Fourier transform.</p>


## Intro
This series of posts looks into different strategies for interpolating non-periodic, smooth data on a uniform grid with high accuracy. For an introduction, see the <a href="https://kunkelalexander.github.io/blog/when-fourier-fails-filters-post/">first post of this series</a>. In this post, we study Gram-Fourier extensions. This method was first described in Lyon's PhD thesis <a href="https://thesis.library.caltech.edu/2992/1/lyon_thesis_A100Final.pdf"> High-order unconditionally-stable FC-AD PDE solvers for general domains </a>.
This method combines the accuracy of SVD extensions with the computational advantages of polynomial expansions. You may find the accompanying <a href="https://github.com/KunkelAlexander/when-fourier-fails-python"> Python code on GitHub </a>.

## The merits of Gram-Fourier extensions
In the [previous post][svd-post], we looked at SVD extensions: Periodic extensions obtained through solving a least-squares optimisation problem. They can be highly accurate, but are computationally expensive and require many collocation points for good results. The Gram-Fourier extension method remedies these drawbacks. Instead of computing SVD extensions of an interpolant at runtime, one precomputes SVD extensions of a set of polynomial basis functions, so-called Gram polynomial. The interpolant is expanded in terms of this polynomial basis set at the domain boundaries. But instead of summing the original polynomials, one then sums the precomputed periodic extensions of the polynomials and obtains a periodic function. In other words, one precomputes a change-of-basis from a polynomial basis to a periodic basis.
This has several advantages: Firstly, the SVD extension of an analytically known function can be arbitrarily accurate because it can use arbitrarily many collocation points and arbitrarily high precision. Secondly, the polynomial expansion is only carried out at the interpolant's boundaries and therefore has a constant computational cost. Since the change-of-basis is precomputed, the periodic extension operation has constant computational cost vs $$\mathcal{O}(N^3)$$ for a standard SVD extension. The fact that the interpolant is only expanded at the boundaries means that the extension is oblivious to what the interpolant looks like away from the domain boundary. This limits the accuracy of Gram-Fourier extensions since the extension cannot be arbitrarily smooth at the domain boundary. If one uses $$N_{boundary}$$ points to compute the extension, the extension can at most be smooth up to order $$N_{boundary}-1$$. Imagine a polynomial of order $$4$$, for instance. Its $$5^{th}$$ derivative is inevitably zero, regardless of the true value of the $$5^{th}$$ derivative of the interpolant at the boundary. Accordingly, the Fourier coefficients of a Gram-Fourier extension can at best decay like $$\mathcal{O}(k^{-N_{boundary} - 1})$$.

## Gram Polynomials
In the first step towards Gram-Fourier extensions, we start with a set of $$N$$ polynomials on $$N$$ points defined on the left and right boundary of the physical domain.
<img src="{{ site.baseurl }}/assets/img/nonperiodicinterpolation-python/gramfe_polynomials.png" alt="">
In order to expand the interpolant in terms of these polynomials, we use the Gram-Schmidt orthogonalisation algorithm to obtain an orthonormal basis set. Since this is a one-off computation, we carry it out in high precision using Python's mpmath library. Alternatively, all of the following computations could be carried out symbolically.
<img src="{{ site.baseurl }}/assets/img/nonperiodicinterpolation-python/gramfe_orthonormal_polynomials.png" alt="">

{%- highlight python -%}
import matplotlib.pyplot as plt
import numpy as np
import scipy
from mpmath import *

mp.dps = 64
eps = 1e-64

import matplotlib as mpl

plt.style.use('dark_background')

# Define your custom colors
colors = [
    '#08F7FE',  # teal/cyan
    '#FE53BB',  # pink
    '#F5D300',  # yellow
    '#00ff41',  # matrix green
    '#FF00FF',  # magenta
    '#FFA500',  # orange
    '#00FFFF',  # cyan
]

# Set the custom color cycle
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

class GramSchmidt:
    def __init__(self, x, m):
        self.x = x
        self.m = m
        self.A = mp.zeros(m, len(x))
        #Linear map for polynomial scalar product
        for i in range(m):
            for j in range(len(x)):
                #Polynomial basis {1, x, x^2, x^3, x^4, ..., x^m}
                self.A[i, j] = x[j]**i

        #Write basis vector as columns of matrix V
        self.V = mp.eye(m)

        self.U = self.modified_gram_schmidt_algorithm(self.V)

    def evaluate_basis(self, x, basis_element_index):
        #Linear map for polynomial scalar product
        A = mp.zeros(self.m, len(x))
        for i in range(self.m):
            for j in range(len(x)):
                #Polynomial basis {1, x, x^2, x^3, x^4, ..., x^m}
                A[i, j] = x[j]**i
        ei = self.U[:, basis_element_index].T * A

        return ei

    def scalar_product(self, u, v):
        return mp.fsum((u.T * self.A) * (v.T * self.A).T)

    def project_u_onto_v(self, u, v):
        a1 = self.scalar_product(v, u)
        a2 = self.scalar_product(u, u)
        return a1/a2 * u

    def norm(self, u):
        return mp.sqrt(self.scalar_product(u, u))

    def modified_gram_schmidt_algorithm(self, V):
        n, k = V.rows, V.cols
        U    = V.copy()
        U[:, 0] = V[:, 0] / self.norm(V[:, 0])

        for i in range(1, k):
            for j in range(i, k):
                U[:, j] = U[:, j] - self.project_u_onto_v(U[:, i - 1], U[:, j])


            U[:, i] = U[:, i] / self.norm(U[:, i])
        return U

    def project_f_onto_basis(self, f):
        coeffs = mp.matrix(1, self.m)

        for i in range(self.m):
            basis = (self.U[:, i].T * self.A)
            coeffs[0, i] = mp.fsum(f * basis.T)


        return coeffs

    def reconstruct_f(self, coeffs, x = None):
        if x == None:
            A = self.A
        else:
            A = mp.zeros(self.m, len(x))
            for i in range(self.m):
                for j in range(len(x)):
                    #Polynomial basis {1, x, x^2, x^3, x^4, ..., x^m}
                    self.A[i] = x[j]**i

        frec = mp.matrix(1, A.cols)
        for i in range(self.m):
            frec += coeffs[0, i] * (self.U[:, i].T * A)
        return frec

    def plot_polynomials(self):
        m = self.m
        u_ij = mp.zeros(m)

        fig, axs = plt.subplots(figsize=(5, 3), dpi=200)
        plt.title(f"Polynomials")
        plt.axis("off")
        for i in range(m):
            plt.plot(x, self.V[:, i].T * self.A, label=f"$x^{i}$")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/gramfe_polynomials.png", bbox_inches='tight')
        plt.show()

        fig, axs = plt.subplots(figsize=(5, 3), dpi=200)
        plt.title(f"Orthonormalised polynomials")
        plt.axis("off")
        for i in range(m):
            plt.plot(self.x, self.U[:, i].T * self.A, label=f"{i}")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/gramfe_orthonormal_polynomials.png", bbox_inches='tight')
        plt.show()

        print("The orthonormalised polynomials and their scalar products")
        for i in range(m):
            for j in range(m):
                u_ij[i, j] = self.scalar_product(self.U[:, i], self.U[:, j])
            print(f"i = {i} u_ij = {u_ij[i, :]}")

x = mp.linspace(0, 1, 20)
gs = GramSchmidt(x, 5)
gs.plot_polynomials()
{%- endhighlight -%}

## SVD Extensions
For an introduction to SVD extensions, please see the [previous post][svd-post]
<img src="{{ site.baseurl }}/assets/img/nonperiodicinterpolation-python/svd_fig_8.png" alt="">. In the following, we compute suitable SVD extensions of the orthonormal basis set derived in the previous section.  Since we work in arbitrary precision, we do not need to truncate the singular values of the SVD and also do not require iterative refinement. If the calculation is not accurate enough, one can simply increase the number of significant digits. Furthermore, we can directly solve the complex optimisation problem without a split into symmetric and antisymmetric part. This simplifies the code. At the same time, there is an additional complication because we compute two independent extensions for the left and right domain boundary: How to stitch them together? Lyon proposes to compute an even and an odd extensions respectively using only even and odd wave vectors. By taking linear combinations of the two, one obtains extensions that smoothly decay to zero. These extensions can be stitched together to obtain a global periodic extension.

The following plot shows even and odd extensions (blue graphs on the left and right) for the Gram polynomials (pink) of order $$0$$ to $$4$$ (top to bottom).
<img src="{{ site.baseurl }}/assets/img/nonperiodicinterpolation-python/gramfe_even_and_odd_extensions.png" alt="">


## The need for iterative refinement

With this knowledge, we can set out to compute a periodic extension of the even function $$f(x) = x^2$$ shown in the next plot.

<img src="{{ site.baseurl }}/assets/img/nonperiodicinterpolation-python/svd_1.png" alt="">

The mismatch between the extension and the original function in the physical domain is good, but far from the desired machine precision:
<img src="{{ site.baseurl }}/assets/img/nonperiodicinterpolation-python/svd_1_accuracy" alt="">

As explained in Boyd's paper, iterative refinement is another helpful trick for ill-conditioned linear systems. Applying it increases the precision of the extension drastically:

<img src="{{ site.baseurl }}/assets/img/nonperiodicinterpolation-python/svd_2_accuracy" alt="">


{%- highlight python -%}
def func(x):
    return x**2

def truncated_svd_invert(M, cutoff):
    U, s, Vh = scipy.linalg.svd(M)
    sinv = np.zeros(M.T.shape)
    for i in range(np.min(M.shape)):
        if s[i] < cutoff:
            sinv[i, i] = 0
        else:
            sinv[i, i] = 1/s[i]
    return Vh.T @ sinv @ U.T

def iterative_refinement(M, Minv, f, threshold = 100, maxiter = 5):
    a       = Minv @ f
    r       = M @ a - f
    counter = 0
    while np.linalg.norm(r) > threshold * np.finfo(float).eps * np.linalg.norm(a) and counter < maxiter:
        delta    = Minv @ r
        a        = a - delta
        r        = M @ a - f
        counter += 1
    return a

def reconstruct(x, a, theta):
    rec = np.zeros(x.shape)
    for j, coeff in enumerate(a):
        rec += coeff * np.cos(np.pi / theta * j * x)
    return rec

N     = 32
Ncoll = N
theta = np.pi
chi   = theta/2
M, x  = get_fpic_su_matrix(N, Ncoll, theta, chi)
f     = func(x)
Minv  = truncated_svd_invert(M, cutoff = 1e-13)
a1    = Minv @ f
a2    = iterative_refinement(M, Minv, f, threshold = 1000, maxiter = 4)
xext  = np.linspace(0, 2 * theta, 1000)
frec1 = reconstruct(xext, a1, theta)
frec2 = reconstruct(xext, a2, theta)


for i, frec in enumerate([frec1, frec2]):
    # Plot f and extended f
    fig, axs = plt.subplots(figsize=(5 , 3), dpi=200)
    plt.title(r"Periodic extension of $f(x) = x^2$ for $\chi=\pi/2$ and $\Theta = \pi$")
    # Graphs
    plt.plot(xext, frec, label="Extension", c=colors[0])
    plt.plot(xext, func(xext), label="Original", c=colors[1])
    # Axes
    plt.ylim(np.min(frec) - 1, np.max(frec) + 1)
    plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [0, r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_visible(False)
    axs.get_yaxis().set_ticks([])
    # Save plot
    plt.tight_layout()
    plt.savefig(f"figures/svd_{i+1}.png", bbox_inches='tight')
    plt.legend()
    plt.show()

    # Plot error
    fig, axs = plt.subplots(figsize=(5 , 3), dpi=200)
    plt.title(r"Mismatch between $f$ and $\hat{f}$ in physical domain")
    # Physical domain
    ul = np.argwhere(xext<chi)[-1][0]
    xorg = xext[:ul]
    plt.yscale("log")
    plt.xticks([0, np.pi/4, np.pi/2], [0, r"$\pi/4$", r"$\pi/2$"])
    plt.plot(xorg, np.abs(func(xorg) - frec[:ul]), c=colors[0])
    plt.tight_layout()
    plt.savefig(f"figures/svd_{i+1}_accuracy.png", bbox_inches='tight')
    plt.show()
{%- endhighlight -%}

## The need for overcollocation

A part of the ill-conditioning of Fourier extensions stems from the fact that there are functions which are small on collocation grid, but large in the gaps between the interpolation points. This can be remedied by having a number of collocation points $$N_{coll}$$ much larger than $$N$$, the number of Fourier coefficients.  In this case, the scheme can detect large-amplitude oscillations of $$f$$ and produce more accurate extensions. The following plot shows reconstruction errors of $$f(x) = \cos(40.5 x)$$ on $$[0, \pi/2]$$ for different numbers of Fourier modes in two cases: a square-matrix $$M$$ where $$N_{coll} = N$$ and an overcollocation case where $$N_{coll} = 2\cdot N$$.

<img src="{{ site.baseurl }}/assets/img/nonperiodicinterpolation-python/svd_why_overcollocation.png" alt="">

The achievable accuracy in the second case is significantly higher.
{%- highlight python -%}

theta = np.pi
chi   = theta/2
xext  = np.linspace(0, theta, 1000)
ul    = np.argwhere(xext<chi)[-1][0]

def func(x):
    return np.cos(40.5*x)

fig, ax = plt.subplots(1, 2, figsize=(5, 3), dpi=200, sharey=True)
ax[0].set_title(r"$N_{coll}=N$")
ax[0].set_xlabel(r"$N$")
ax[1].set_xlabel(r"$N$")
ax[0].set_ylabel("Reconstruction error")
ax[0].set_yscale("log")
ax[0].set_ylim([1e-14, 1e8])
ax[1].set_ylim([1e-14, 1e8])
ax[1].set_title(r"$N_{coll} = 2 N$")

Ns         = np.arange(5, 100, 1)
iterations = [0, 1, 2, 3]
cutoff     = 1e-13
threshold  = 2
for axis, alpha in zip([0, 1], [1, 2]):
    for iteration in iterations:
        err = []
        for N in Ns:
            Ncoll = N * alpha
            M, x  = get_fpic_su_matrix(N, Ncoll, theta, chi)
            f     = func(x)
            Minv  = truncated_svd_invert(M, cutoff)
            a    = iterative_refinement(M, Minv, f, threshold = threshold, maxiter = iteration)
            frec = reconstruct(xext, a, theta)
            err.append(np.linalg.norm((frec - func(xext))[:ul]))

        ax[axis].plot(Ns, err, label=r"$N_{iter}$" + f" = {iteration}", c = colors[iteration])


ax[1].legend()
fig.subplots_adjust(wspace = 0)
plt.savefig(f"figures/svd_why_overcollocation.png", bbox_inches='tight')
plt.show()
{%- endhighlight -%}

## Remaining issues
Fourier extensions of the third kind can be very accurate when overcollocation and iterative refinement are used. This accuracy comes at a price, however: A typical SVD factorisation requires $$\mathcal{O}(N^3)$$ operations. The iteration refinement adds $$\mathcal{O}(2 N^2 + 2 N_{coll}^2)$$ operations per iteration. The overcollocation requirement dictates that $$\Delta x$$ is at least halved in the extension domain compared to the physical domain. When the number of collocation points is limited as in the case of interpolation or a PDE solver, the requirements of Fourier extensions of the third kind can be hardly met and are computationally prohibitively expensive. In addition, my own numerical experiments indicate that numerical stability of a PDE solver built with this method is an issue. Lastly, there are no analytical convergence guarantees for Fourier extensions of the third kind. Fortunately, all of these issues are resolved by Gram-Fourier extensions that we are going to look at in the last part of this series.

[svd-post]: https://kunkelalexander.github.io/blog/when-fourier-fails-svd-extension-post/