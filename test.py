import jax
import jax.numpy as jnp
from layers import coupling_layer, squeeze_layer

key = jax.random.PRNGKey(1)
shape = (3, 8, 8)  
x = jax.random.normal(jax.random.PRNGKey(2), shape)

# Test checkerboard masking
print("Testing checkerboard coupling")
layer_cb = coupling_layer(key, checkerboard=True, shape=shape)

# Forward 
y,  = layer_cb(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
assert y.shape == x.shape

# Inverse
x_reconstructed = layer_cb.inverse(y)
print(f"Reconstructed shape: {x_reconstructed.shape}")
error = jnp.abs(x - x_reconstructed).max()
assert error < 1e-5, f"Reconstruction error too large: {error}"

# Log-det-jacobian
log_det = layer_cb.log_jac(x)
print(f"Log-det shape: {log_det.shape}")
print(f"Log-det value: {log_det}")

# channel mask
print("channel-wise coupling")
layer_ch = coupling_layer(key, checkerboard=False, shape=shape)
y2 = layer_ch(x)
x_reconstructed2 = layer_ch.inverse(y2)
error2 = jnp.abs(x - x_reconstructed2).max()
print(f"Max reconstruction error: {error2}")

# squeeeeeeze
squeeze = squeeze_layer()
x_squeeze = jax.random.normal(jax.random.PRNGKey(2), (3, 8, 8))
# Forward
y_squeeze = squeeze(x_squeeze)
print(f"Input shape: {x_squeeze.shape}")
print(f"After squeeze: {y_squeeze.shape}")
# Inverse
x_unsqueeze = squeeze.inverse(y_squeeze)
print(f"After unsqueeze: {x_unsqueeze.shape}")
