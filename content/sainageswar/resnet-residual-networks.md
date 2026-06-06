---
author: "[Sai Nageswar Satchidanand](/sainageswar)"
date: '2026-06-06T00:00:00Z'
draft: false
title: 'ResNet: Residual Networks'
description: 'The intuition behind residual networks — why learning the difference is easier than learning the answer, and how it connects to Euler integration.'
tags: ['Deep Learning', 'ResNet', 'Image Super-Resolution', 'Computer Vision']
categories: ['tech']
math: true
comments: true
---

ResNet, introduced by Microsoft Research in 2015, is one of the most influential deep learning architectures ever published — with over 100,000 citations. The core idea is surprisingly simple: instead of learning a mapping from input to output, learn only the **difference** between them.

## The Super-Resolution Intuition

The easiest way to understand ResNet is through image super-resolution. Say you have a low-resolution image $X$ and you want to produce a high-resolution version $Y$.

You could train a neural network to learn the full mapping $X \to Y$. But think about what that means — the network has to reconstruct everything from scratch, even though most of the information in $Y$ already exists in $X$. The shapes, colors, and broad structure are all there. What's missing is a relatively small set of fine details: sharp edges, high-frequency textures.

So instead of learning $Y$ directly, learn only the **residual**:

$$Y = X + \mathcal{F}(X)$$

where $\mathcal{F}(X)$ is the small correction the network needs to add. You pass a copy of $X$ forward via a skip connection and add it to whatever the network produces. The network only has to focus its capacity on the part it doesn't already know.

This is the key insight behind ResNet. The skip connection — also called a jump connection — carries the original input forward and adds it to the output of the block.

## The Forgetting Problem in Deep Networks

After the 2012 ImageNet breakthrough, it was clear that deeper networks generally perform better on hard tasks. So everyone tried to go deeper. But a strange problem emerged: adding more layers made training accuracy *worse*, not better — even on the training set. This wasn't overfitting. It was an optimization failure.

The root cause is forgetting. Over many layers, it becomes hard for the network to remember the original input $X$. And in many tasks, the output is actually quite close to the input, so forgetting it is a real loss.

ResNet fixes this directly. The skip connection means the input is never truly forgotten — it's manually carried forward and added back at each block. Even if the learned residual $\mathcal{F}(X)$ is noisy or unhelpful, the original $X$ is always there. This also helps backpropagation: gradients have a direct path back through the skip connection, avoiding the vanishing gradient problem that plagues very deep plain networks.

The result: ResNet-34 outperformed all previous architectures on ImageNet, COCO, and CIFAR-10 simultaneously when it was published — sweeping the leaderboards.

## The Residual Block

![Residual Block](/images/sainageswar/resnet-residual-block.svg)

The convolutional layers use stride 1 and padding 1, which keeps the spatial dimensions unchanged. So input and output are the same size — which is exactly what you want for image-to-image tasks like super-resolution.

For classification tasks where you want to progressively shrink the spatial dimensions, you use stride 2 (halving the resolution) while doubling the number of filters — the standard way to extract increasingly abstract features while reducing spatial size.

## ResNet as Euler Integration

There's a deep connection between ResNet and numerical integration of differential equations. Start with a concrete example.

### The Bunny Population Problem

Suppose you have a population of bunnies. Bunnies reproduce at a rate proportional to how many bunnies there are — the more bunnies, the faster the population grows. If $B(t)$ is the population at time $t$ and the growth rate is $r$, this is described by:

$$\frac{dB}{dt} = r \cdot B(t)$$

This is a differential equation. It tells you the *rate of change* at every moment, but not the population directly. To find $B$ at some future time, you need to integrate.

The exact solution is $B(t) = B_0 \cdot e^{rt}$. But suppose you didn't know that — you only know the rule "population grows at rate $r \times$ current population." How do you estimate the future population step by step?

**Euler's method**: take a small time step $\Delta t$, use the current rate to project forward, repeat:

$$B_{k+1} = B_k + \Delta t \cdot r \cdot B_k$$

The $\Delta t \cdot r \cdot B_k$ term is the key. The differential equation gives you the *instantaneous rate of change* $\frac{dB}{dt} = r \cdot B_k$ — how many bunnies are being added per unit time right now. Multiplying by $\Delta t$ converts that rate into an actual count: if the population is growing at $r \cdot B_k$ bunnies per month, then over $\Delta t$ months it grows by $\Delta t \cdot r \cdot B_k$ bunnies. You are approximating a small slice of the integral — the area under the rate curve over the interval $[t_k,\ t_k + \Delta t]$ — as a rectangle of height $r \cdot B_k$ and width $\Delta t$.

Summing all these rectangular slices from $t=0$ to $t=T$ is exactly what the integral computes:

$$B(T) = B_0 + \int_0^T r \cdot B(t)\, dt$$

Euler replaces that continuous integral with a finite sum of rectangles:

$$B(T) \approx B_0 + \sum_{k=0}^{N-1} \Delta t \cdot r \cdot B_k$$

Say you start with $B_0 = 100$ bunnies, $r = 0.5$ per month, and you step forward one month at a time. After month 1:

$$B_1 = 100 + 1.0 \times 0.5 \times 100 = 150$$

After month 2:

$$B_2 = 150 + 1.0 \times 0.5 \times 150 = 225$$

The exact answer after 2 months is $100 \cdot e^{0.5 \times 2} \approx 272$. Euler gives 225. It undershoots — and the bigger the time step, the worse it gets.

### Each Residual Block Is One Euler Step

Now look at a residual block:

$$\mathbf{x}_{k+1} = \mathbf{x}_k + \mathcal{F}(\mathbf{x}_k)$$

This is exactly the Euler update. The network $\mathcal{F}$ is learning the *rate of change* — the residual to add at each step. Each block is one time step. A deeper ResNet is more Euler steps, giving a finer approximation of the underlying continuous transformation.

In the bunny analogy: one residual block says "given the current image, what small correction should I add?" — just like Euler asks "given the current population, how much does it grow this month?" Stack 34 blocks and you have 34 months of fine-grained corrections.

### Euler Is a Lower Bound — It Misses Compounding

Euler undershoots for a fundamental reason: it uses the slope at the *start* of each step and projects linearly, ignoring how the rate itself changes *within* the step.

With the bunnies: in reality, the bunnies born in the first two weeks of month 1 are themselves reproducing by week 3. Euler ignores this — it only counts the babies born by the original 100, not the babies' babies. The true integral accounts for continuous compounding; Euler only does simple addition.

For dynamics with positive curvature (like exponential growth), the true trajectory always curves upward faster than any linear projection. Euler is therefore always a lower bound for such systems.

A ResNet has the same blind spot. Each block sees only its input $\mathbf{x}_k$ and applies one correction. It cannot account for how that correction would itself evolve if broken into smaller sub-steps.

Higher-order integrators like Runge-Kutta fix this by evaluating the slope at intermediate points within each step. The Neural ODE (2018) takes it to the limit: model the continuous vector field $f$ directly and use an adaptive ODE solver — like having infinitely many infinitesimal residual blocks with true compounding. That's the natural next step beyond ResNet for modeling dynamical systems.

### References

[1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*. https://arxiv.org/abs/1512.03385

[2] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity Mappings in Deep Residual Networks. *ECCV 2016*. https://arxiv.org/abs/1603.05027

[3] Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. *NeurIPS 2018*. https://arxiv.org/abs/1806.07366

[4] Lim, B., Son, S., Kim, H., Nah, S., & Lee, K. M. (2017). Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR). *CVPRW 2017*. https://arxiv.org/abs/1707.02921
