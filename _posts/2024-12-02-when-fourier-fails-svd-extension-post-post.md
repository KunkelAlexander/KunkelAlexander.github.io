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

$$\min_{a_k \forall k \in t(m)} \sum^{n-1}_{j=0} \left| \sum_{k\in t(m)} a_k e^{\frac{2 \pi i}{b} k x_j} - f(x_j)\right|^2 = \min_{\hat{x}} \sum^{n-1}_{j=0} \left|A\hat{x} - b\right|^2$$

The wave vectors are $$k$$ are chosen from a suitable set $$t(m)$$ and the points $$x_j$$ are collocation points in the physical domain $$[0, \chi]$$. By solving the optimisation problem, we ensure that the mismatch between $$f(x)$$ and its extension $$\hat{f}(x)$$ in the physical domain is small.