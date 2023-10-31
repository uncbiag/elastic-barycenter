# elastic-barycenter
Computes the elastic barycenter of an array of images.

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
In order to find $\phi^{-1}(y)$, we can use a gradient descent approach to minimize the regularized SSD loss $$\Sigma (I_s(\phi^{-1}(y)) - I_t(y))^2 dy + \gamma ||u_y||^2 dy$$

A gradient descent algorithm (in this case ADAM), will try to minimize this loss so that the source gets registered to the target image.

### Computation of the Barycenter
Given images $I_0, I_1, ..., I_N$, we want to find an image $I_b$ that corresponds to an "average" of the images. This average will be that which corresponds to the type of distance we are using between an image $I_i$ and its barycenter. In our case, the distance between the images and the barycenter is measured by the regularized SSD loss. For an example of a barycenter computed using a Wasserstein distance, refer to [this example in GeomLoss](https://www.kernel-operations.io/geomloss/_auto_examples/optimal_transport/plot_wasserstein_barycenters_2D.html)
