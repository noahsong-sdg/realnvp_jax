import jax 
import jax.numpy as jnp
import equinox as eqx
from typing import Callable 

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class coupling_1d(eqx.Module):
    s: eqx.nn.MLP 
    t: eqx.nn.MLP
    D: int 
    d: int
    def __init__(self, key, shape):
        self.D = shape
        self.d = self.D // 2
        skey, tkey = jax.random.split(key, 2)
        self.s = eqx.nn.MLP(self.d, self.d, 6, 2, key=skey)
        self.t = eqx.nn.MLP(self.d, self.d, 6, 2, key=tkey)
        
    def __call__(self, x: jnp.array):
        keep = x[:self.d]
        change = x[self.d:] * jnp.exp(self.s(x[:self.d])) + self.t(x[:self.d])
        return jnp.concatenate([keep, change])
        
    def inverse(self, y):
        keep = y[:self.d]
        change = (y[self.d:] - self.t(y[:self.d])) * jnp.exp(-1 * self.s(y[:self.d]))
        return jnp.concatenate([keep, change])

    def log_det_jacobian(self, x):
        return jnp.sum(self.s(x[:self.d]))

key = jax.random.PRNGKey(0) 
layer = coupling_1d(key, shape=8) 
x = jax.random.normal(jax.random.PRNGKey(1), (8,))
y = layer(x)
x_reconstructed = layer.inverse(y) 
print(x)
print(x_reconstructed)
