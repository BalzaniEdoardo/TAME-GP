# TAME-GP
The Taske Aligned Manifold Estimation (TAME-GP), is an extended Poisson-Gaussian CCA with GP-priors for the latent factors.

The code implements a targeted dimensionality reduction of neural spiking data fitting a latent variable probabilistic model.<br>
The observed variables are spike counts from m different simoultaneusly recorded neural populations <b>x</b><sub>1</sub>,..,<b>x</b><sub>m</sub>, and some task relevant variables <b>s</b>.<br>
We introduce a latent variable  <b>z</b><sub>0</sub> capturing correlation between task varaibles and spiking (as in a probabilistic CCA, fig A), and other 
 <b>z</b><sub>j</sub>, j=1,..,m latents capturing whithin brain area correlations (as  in a factor analysis). <br>
 
 For each factor we include a GP prior with RBF kernels, <br>
 <img src="https://latex.codecogs.com/svg.latex?\Large&space;p(\mathbf{z}_j)\sim\text{GP}\left(0,K_j\right)" title="\Large xx" />,<br>
 and we model the likelihood task variables given the factor <b>z</b><sub>0</sub> as a Gaussian with,<br>
  <img src="https://latex.codecogs.com/svg.latex?\Large&space;p(\mathbf{s}|\mathbf{z}_0)\sim\mathcal{N}\left(C\cdot\mathbf{z}_0+d,\psi\right)" title="\Large xx" />,<br>
while the spike counts of the units of the j-th neural population are assumed independent given the latent variables and Poisson distributed, for the i-th unit we set<br>


<img src="https://latex.codecogs.com/svg.latex?\Large&space;p(\mathbf{x}_j^i|\mathbf{z}_0,\mathbf{z}_j)\sim\text{Poisson}\left(C_{ij}\cdot\mathbf{z}_j+C_{i0}\cdot\mathbf{z}_0+d_{ij}\right)" title="\Large xx" />.<br>

Given the assumptions, the model factorizes according to,

<img src="https://latex.codecogs.com/svg.latex?\Large&space;p(\mathbf{x},\mathbf{s},\mathbf{z})=\prod_{j\ge0}p(\mathbf{z}_j)p(\mathbf{s}|\mathbf{z}_0)\prod_{j>0,i}p(\mathbf{x}_j^i|\mathbf{z}_j,\mathbf{z}_0)" title="\Large xx" />

The resulting graphical model is depicted in fig B.
![pCCA_schemes](https://user-images.githubusercontent.com/28624715/148568234-66c0f179-c839-4d56-ad0e-826940bb324d.png)


# Implementation
The code implements parameter learning and inference in the mdoel presented above. Parameters are learned with an approximate EM algorithm where the latent posterior distribution is approximated via the Laplace method.
Inference is implemented as numerical optimization of the MAP for the latents. Latent covariacnce is approximated via the Laplace method.

