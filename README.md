This repository contains code to calculate PTE and number of sigma for a given point using three different methods:

- Gaussian Approximation: Approximates the distribution by a Gaussian, from which it can easily generate samples
- KDE Method: Uses samples from a distribution and estimates the likelihood using Kernel Density Estimation
- Nearest Neighbour: Finds the nearest neighbour in an MCMC/Nested Sampling chain, and uses its likelihood as an estimate of the points's likelihood

Created by Pablo Lemos (UCL)

For questions or comments, contact me on pablo.lemos.18@ucl.ac.uk
