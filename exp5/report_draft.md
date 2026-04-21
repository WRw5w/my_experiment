# Week 5 Assignment: Image Denoising with DnCNN and Comparison with BM3D/ISTA/FISTA/ADMM

**Abstract**  
This report provides a comparative analysis of five distinct image denoising algorithms: Block-Matching and 3D filtering (BM3D), three iterative optimization methods (ISTA, FISTA, ADMM) solving the Total Variation (TV) model, and a deep learning approach (DnCNN). Experiments demonstrate that DnCNN achieves the highest restoration quality while maintaining rapid inference speeds, significantly outperforming traditional optimization frameworks. 

## I. Introduction
Image denoising aims to recover a clean image $x$ from a noisy observation $y = x + v$, where $v$ is typically assumed to be Additive White Gaussian Noise (AWGN). Optimization methods iteratively solve an objective function based on hand-crafted physical priors, which is mathematically rigorous but often computationally expensive. In contrast, deep learning methods like DnCNN learn a direct mapping from noisy to clean images using large datasets, offering rapid inference and superior generalization to complex noise patterns.

## II. Methods
### A. Block-Matching and 3D Filtering (BM3D)
BM3D is a state-of-the-art traditional algorithm relying on non-local patch similarity. The core concept involves grouping similar 2D image fragments into 3D data arrays, applying a 3D collaborative filter (e.g., using DCT or wavelet transforms), and then aggregating the estimates back to their original positions.

### B. Total Variation (TV) via ISTA, FISTA, and ADMM
Optimization-based methods minimize the following Total Variation objective function:
$$\min_x \frac{1}{2}\|x-y\|_2^2 + \lambda \|\nabla x\|_1$$
1. **ISTA**: Iterative Shrinkage-Thresholding Algorithm applies a forward gradient descent step followed by a proximal mapping step (soft-thresholding) on the TV penalty.
2. **FISTA**: A fast variant of ISTA that introduces a Nesterov-style momentum term to accelerate the convergence rate from $\mathcal{O}(1/k)$ to $\mathcal{O}(1/k^2)$.
3. **ADMM**: Alternating Direction Method of Multipliers tackles the problem by splitting the variables and solving the subproblems via an augmented Lagrangian, often yielding faster empirical convergence per iteration.

### C. Denoising Convolutional Neural Network (DnCNN)
DnCNN approaches denoising through a deep fully convolutional architecture utilizing residual learning. Instead of directly outputting the denoised image $x$, the network $\mathcal{R}(\cdot)$ is trained to predict the residual noise $v$:
$$\mathcal{R}(y) \approx v \implies \hat{x} = y - \mathcal{R}(y)$$
By incorporating Batch Normalization and ReLU activations, DnCNN effectively mitigates the vanishing gradient problem and accelerates training.

## III. Experiments
**Setup:** We evaluated the algorithms on a standard `camera` image contaminated with AWGN ($\sigma = 25/255$). For the optimization methods (ISTA, FISTA, ADMM), we tuned the regularization parameter $\lambda=0.1$ and ran for 100 iterations. For BM3D, we used the analytical baseline. For DnCNN, we utilized the pre-trained `dncnn_25.pth` model from the KAIR repository. 

**Results:**
The quantitative performance measured in Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) is summarized in Table I.

| Method | PSNR (dB) | SSIM | Time (s) |
|--------|-----------|--------|----------|
| Noisy  | 20.61   | 0.3012 | N/A     |
| ADMM   | 27.75   | 0.6438 | 0.2953  |
| ISTA   | 28.52   | 0.7216 | 2.1029  |
| FISTA  | 28.53   | 0.7221 | 1.5234  |
| BM3D   | 29.70   | 0.7963 | 10.9503 |
| DnCNN  | **29.99**| **0.8148** | 0.9677 |

*Table I: Quantitative comparison of denoising methods on AWGN sigma=25.*

**Analysis:**
DnCNN heavily outperformed the other models, yielding the highest PSNR (29.99 dB) and SSIM (0.8148) while requiring minimal inference time (0.96s) by exploiting GPU acceleration. BM3D achieved robust results closely following DnCNN but suffered from severe computational overhead (10.95s). Among the TV-based models, FISTA and ISTA saturated around 28.5dB, whereas ADMM was exceptionally fast per iteration but yielded a slightly lower PSNR.

## IV. Conclusion
This study compared classical and learning-based denoising frameworks. While optimization techniques like FISTA and ADMM provide strong theoretical guarantees, empirical evidence confirms that data-driven mechanisms like DnCNN achieve vastly superior restoration fidelity and practical runtime efficiency.
