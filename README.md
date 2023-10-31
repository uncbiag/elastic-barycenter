# elastic-barycenter
Computes the elastic barycenter of an array of images.
<p align="center">
  <img alt="Barycenter of a cross and a square" src="https://github.com/uncbiag/elastic-barycenter/blob/main/Readme_imgs/barycenter_cross.png" width=60% heights=50%>
</p>
  
## Use

Uncomment the wanted example of the file `toy_barycenter.py` and run it using. 
```
python toy_barycenter.py
```
If you are connecting with `ssh`, remember to connect with `-X` or `-Y` to visualize the transformations.  
You can find other useful image reading functions in `image_functions.py` in case the amount of images you want to compute a barycenter for are more complex.  
Make sure that you update the variable `outpath` in `toy_barycenter.py` to describe where you want to save the final barycenter array. 
  
Find more comments written in the code.
  
## Description of Algorithm
  
### Elastic registration

Let $I_s(x_{i,j})$ be the intensity of a source image at the pixel location $(i, j)$ and $I_t(y_{i, j})$ be the intensity of the target image at the pixel location $(i, j)$. We want to find a registration map $\phi^{-1}(y) = y + u(y)$ so that $I_s(\phi^{-1}(y))$ is as close as possible to $I_t(y)$.  
In order to find $\phi^{-1}(y)$, we can use a gradient descent approach to minimize the regularized SSD loss $$\sum (I_s(\phi^{-1}(y)) - I_t(y))^2 dy + \gamma ||u_y||^2 dy$$
  
A gradient descent algorithm (in this case ADAM), will try to minimize this loss so that the source gets registered to the target image. For more information about elastic registration, you can refer to [Chapter 9](https://academic.oup.com/book/1500/chapter/140922432) of Modersitzki's book "Numerical Methods for Image Registration".
  
### Computation of the Barycenter
Given images $I_0, I_1, ..., I_N$, we want to find an image $I_b$ that corresponds to an "average" of the images, which we will call the center of mass, or barycenter. This barycenter will be computed based on the type of distance we are using between images $I_i$. In our case, the distance between the images and the barycenter is measured by the regularized SSD loss. For an example of a barycenter computed using a Wasserstein distance, refer to [this example in GeomLoss](https://www.kernel-operations.io/geomloss/_auto_examples/optimal_transport/plot_wasserstein_barycenters_2D.html).
  
The following is the algorithm used to compute the elastic barycenter of images, as illustrated by the equal radius circles distributed along the diagonals example in `toy_barycenter.py`.  
  
First, initialize the images that you want to compute a barycenter for:
<p align="center">
  <img alt="Equal radius circles displaced along the diagonals" src="https://github.com/uncbiag/elastic-barycenter/blob/main/Readme_imgs/Circles_Ex.png" width=60% height=60%>
</p>

For every single image, we want to find a function $\phi^{-1}_i(y) = y + u_i(y)$ that registers $I_i(x)$ to $I_b(y)$. Since we only need a movement vector $u_i(y)$ for every image, these are the parameters that we want to optimize using a regularized SSD loss. 
  
In order to compute the barycenter, we need two updating loops:
1. One external loop that updates the barycenter $I_b$ based on the registered images $I_i(\phi^{-1}_i(y))$
2. An internal loop that registers via gradient descent every image $I_i(x)$ to $I_b$

When we start the external loop, we want the barycenter to be computed via an unweighted average of the images' intensities. That is $$I_b = \frac{1}{N} \sum_{i=0}^{N} I_i(\phi^{-1}_i(y))$$

Before any registration from $I_i$ to $I_b$ has occurred, the barycenter will be the average of the original images:
<p align="center">
  <img alt="First barycenter" src="https://github.com/uncbiag/elastic-barycenter/blob/main/Readme_imgs/First_barycenter.png" width=20% height=20%>
</p>



