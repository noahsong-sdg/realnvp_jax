1. find out how to get cifar 10 access? local download or something in the cloud something

2. 





for each channel, it divides the image into subsquares of hsape 2 x 2 x c, then reshapes them
into subsquares of size 1 x 1 x 4c. 

three coupling layers with alternating checkerboard masks, then perform a squeezing operation,
then apply 3 coupling layers with alternating channel wise masking, which is selected so that the resultant
partitionining is not redundant with the prior checkerboard masking
for the final scale, we only apply four coupling layers with alternating checkerboard masks


i dont understand what it means that the model must Gaussianize units which are factored out at a finer scale

they use batch norm and residual connections and weight norm

Q: 
its confusing to me why they emphasize exploiting the local correlation structure of images and then use a partition that destroys local information, the checkerboard

For CIFAR-10, we use 8 residual blocks, 64 feature maps, and downscale only once.
 ADAM [33 ] with default hyperparameters and use an L2 regularization on the weight scale parameters with coefficient 5 · 10−5.


eqx.filter_jit instead of jax.jit, eqx.nn.Lambda?


