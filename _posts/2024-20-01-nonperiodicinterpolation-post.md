---
layout: post
title:  "Accurate interpolation of non-periodic data"
date:   2024-01-21
description: Learn how to interpolate non-periodic, uniform data accurately using the Fourier transform!
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<p class="intro"><span class="dropcap">I</span>n today's post, we study ways to accurately interpolate non-periodic, smooth data on a uniform grid. </p>


## Intro
This post looks at different strategies to interpolate non-periodic, smooth data on a uniform grid with high accuracy. I came across this problem when trying to increase the accuracy of interpolation in the astrophysical AMR simulation code [GAMER][gamer-gh]. GAMER solves various partial differential equations in large simulation domains. It adaptively increases the grid resolution where necessary. To achieve this, it divides data on a uniform grid into smaller domains and interpolates them independently using various polynomial interpolants up to fourth order. The accuracy of the interpolation is determined by the order of the interpolants. Yet, the interpolant order is limited by [Runge's phenomenon][runge-wiki]. Fourier methods for periodic data do not share this limitation and achieve spectral accuracy: Exponential convergence of the interpolant to the data with the number of grid points. Yet, Fourier methods expect smooth, periodic data. If the data's n-th derivative is discontinuous, the order of convergence for Fourier methods generally drops to n-th order polynomial convergence. To circumvent this limitation and achieve higher accuracy, there are two approaches: Firstly, one could switch to a different grid or different interpolants to increase the stability of the interpolation. Polynomial interpolation on a Chebyshev grid is the preferred way of representing non-periodic data with high accuracy. For more information, read the [excellent book on the subject][boyd-cheby] by John P. Boyd. If one is in the unfortunate situation to be limited to a uniform grid, however, there are alternative approaches.
Such alternative approaches include
- Subtraction methods
- Gegenbauer methods
- Inverse polynomial reconstructions
- Filters and mollifiers
- Modified Fourier transforms using Eckhoff's method
- Transparent boundary conditions
- Local Fourier bases
- Singular Fourier-Padé expansions
- Polynomial least squares methods
- Periodic SVD extensions
- Gram-Fourier extensions
Some of these approaches are also suitable for the accurate solution of PDEs with non-periodic boundary conditions. After having extensively studied the above methods, I highly recommend interested readers to take a look at Gram-Fourier extensions for fast, accurate periodic extensions that can be integrated into existing PDE solvers on uniform grids.66

You may find the accompanying <a href="https://github.com/KunkelAlexander/nonperiodicinterpolation-python"> Python code on GitHub </a>.

<img src="{{ site.baseurl }}/assets/img/nonperiodicinterpolation-python/runge.png" alt="">




[gamer-gh]: https://github.com/gamer-project/gamer
[runge-wiki]: https://en.wikipedia.org/wiki/Runge's_phenomenon
[boyd-cheby]: https://depts.washington.edu/ph506/Boyd.pdf
[periodicinterpolation-python-gh]: https://github.com/KunkelAlexander/periodicinterpolation-python
[numpy-fft-documentation]: https://numpy.org/doc/stable/reference/routines.fft.html