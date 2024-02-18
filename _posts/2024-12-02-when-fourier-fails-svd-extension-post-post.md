---
layout: post
title:  "When Fourier fails: SVDs and Fourier extensions of the third kind"
date:   2024-02-12
description: Learn about the magic of high-precision extensions using SVDs.
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<p class="intro"><span class="dropcap">I</span>n today's post, we study the Fourier extensions of the third kind, my personal favourite. </p>


## Intro
This series of posts looks into different strategies for interpolating non-periodic, smooth data on a uniform grid with high accuracy. For an introduction, see the <a href="https://kunkelalexander.github.io/blog/when-fourier-fails-filters-post/">first post of this series</a>. In this post, we delve into Fourier extensions using truncated singular value decompositions. The method presented here is known as Fourier extension of the third kind as described in Boyd's insightful paper <a href="https://www.sciencedirect.com/science/article/abs/pii/S0021999102970233"> A Comparison of Numerical Algorithms for Fourier Extension of the First, Second, and Third Kinds </a>.
SVD extensions are particularly simple and beautiful: Instead of aiming to find a periodic extension and then Fourier transform, one instead solves a linear system whose solutions are the Fourier coeffients. You may find the accompanying <a href="https://github.com/KunkelAlexander/when-fourier-fails-python"> Python code on GitHub </a>. Can you guess the definition of the linear operator from looking at its matrix representation?

<img src="{{ site.baseurl }}/assets/img/nonperiodicinterpolation-python/svd_W.png" alt="">

## Fourier Physical Interval Collocation—Spectral Coefficients as the Unknowns (FPIC-SU)
Let $$\hat{f}(x)$$, defined in $$[0, \Theta]$$ be the periodic extension of $$f(x)$$, defined in $$[0, \chi]$$ where $$\Theta > \chi$$.
$$\hat{f}(x)$$ allows for an accurate Fourier expansion as $$\hat{f}(x) = \sum_k a_k e^{\frac{2 \pi i}{\Theta} k x}$$. The crucial idea of Fourier extensions of the third kind is that the coefficients $$a_k$$ of the Fourier series can be obtained by solving a linear optimisation problem:

$$\min_{a_k \forall k} \sum^{n-1}_{j=0} \left| \sum_{k} a_k e^{\frac{2 \pi i}{b} k x_j} - f(x_j)\right|^2 = \min_{\hat{a}} \sum^{n-1}_{j=0} \left|M\hat{a} - f\right|^2$$

The points $$x_j$$ are collocation points in the physical domain $$[0, \chi]$$. By solving the optimisation problem, we ensure that the mismatch between $$f(x)$$ and its extension $$\hat{f}(x)$$ in the physical domain is small. Does that mean that we can just invert $$M$$ to solve the optimisation problem? That would be the case if $$M$$ was invertible. But for a given function $$f$$, one can imagine many periodic extensions that follow $$f$$ in the physical domain and take different values in the extended domain. Clearly, the system is underdetermined and admits infinitely many solutions in the continuous case. In the discrete case, a certain choice of collocation points and wave vectors may or may not uniquely specify a solution. In any case, the system is ill-conditioned because it is close to being uninvertible in finite precision. Fortunately, one can still solve the optimisation problem by computing the singular value decomposition of $$M$$ and truncating small singular values before inverting $$M$$.

## Collocation matrix
According to my experience, solving the above complex expressions directly leads to poor results. Instead, one should compute separate extensions for the real and imaginary parts of a complex input function. Moreover, Boyd suggests to split a general real input function $$g(x)$$ into its symmetric and antisymmetric parts $$S(x) = g(x)/2 + g(-x)/2$$ and $$A(x) = A(x)/2 - A(-x)/2$$. The optimisation is then carried out separately for $$S$$ and $$A$$. The symmetric part $$S(x) \equiv f(x)$$ allows for a periodic extension in terms of a cosine series whereas the antisymmetric part requires a sine series. In the following, we focus on the symmetric part, but the antisymmetric part follows analogously.

The Fourier coefficients of the cosine interpolation are the solution of the matrix problem

$$M\hat{a} = f$$,

where

$$ M_{ij} = \cos\left([j-1] \frac{\pi}{\Theta} x_i\right),\qquad i = 1, 2, ..., N_{coll},\qquad j = 1, 2, ..., N$$
and $$f_i = f(x_i), \qquad i = 1, 2, ..., N_{coll}$$,

where the collocation points are uniformly distributed over the positive half of the physical interval $$x\in [0, \chi]$$ with

$$ x_i \equiv \frac{(i-1) \chi}{N_{coll} - 1}, \qquad i = 1, 2, ..., N_{coll} $$. The matrix $$M$$ is shown in the introduction.
In the next step, it is