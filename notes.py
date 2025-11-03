import jax 
import jax.numpy as jnp
import equinox as eqx
from typing import Callable 


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class coupling_layer(eqx.Module):
    # what do s and t need to be here
    s: eqx.nn.Conv
    t: eqx.nn.Conv
    mask: jnp.Array
    def __init__(self, key, checkerboard: bool):
        skey, tkey, = jax.random.split(key)
        # mask depends on checkerboard or channel wise
        mask = 

    def __call__(self, d: int, x: jnp.array):
        
        b = self.mask
        y = b * x[d+1:] + (1 - b) * (x * jnp.exp(self.s(b @ x[:d])) + self.t(b @ x[:d]))
        return y

class Block(eqx.Module):
    # i dont know how to write this like at all
    something: tuple

    def __init__(self):
        self.something = (coupling_layer(checkerboard=True),
                            coupling_layer(checkerboard=True),
                            coupling_layer(checkerboard=True),
                            squeeze,
                            coupling_layer(checkerboard=False),
                            coupling_layer(checkerboard=False),
                            coupling_layer(checkerboard=False),
        )


class NVP(eqx.Module):
    layers: tuple 

    def __init__(self, key):
        keys = jax.random.split(key)
        self.layers = ()
