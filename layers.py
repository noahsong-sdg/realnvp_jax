import jax 
import jax.numpy as jnp
import equinox as eqx

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class ReLU(eqx.Module):
    # only exists as a hacky fix to eqx / jax interface
    def __call__(self, x, key):
        return jnp.maximum(0, x)

class coupling_layer(eqx.Module):
    # what do s and t need to be here? in the paper, they are rectified conv
    s: eqx.nn.Conv
    t: eqx.nn.Conv
    mask: jnp.array # either checkerboard or channelwise masking
    def __init__(self, key, checkerboard: bool, shape):
        skey, tkey, = jax.random.split(key, 2)
        # spatial dim, in channel, out channel, kernel size, 1
        skey1, skey2, skey3 = jax.random.split(skey, 3)
        tkey1, tkey2, tkey3 = jax.random.split(tkey, 3)
        c, h, w = shape
        self.s = eqx.nn.Sequential((eqx.nn.Conv(2, c, c, 3, 1, padding=1, key=skey1),
                            ReLU(),
                            eqx.nn.Conv(2, c, c, 3, 1, padding=1, key=skey2),
                            ReLU(),
                            eqx.nn.Conv(2, c, c, 3, 1, padding=1, key=skey3)))
        self.t = eqx.nn.Sequential((eqx.nn.Conv(2, c, c, 3, 1, padding=1, key=tkey1),
                                    ReLU(),
                                    eqx.nn.Conv(2, c, c, 3, 1, padding=1, key=tkey2),
                                    ReLU(),
                                    eqx.nn.Conv(2, c, c, 3, 1, padding=1, key=tkey3)))
        # im thinking i define the masks somewhere else, and do sm like
        if checkerboard:
            self.mask = self.checkerboard_mask(shape)
        else:
            self.mask = self.channel_mask(shape)

    def channel_mask(self, shape):
        c, h, w = shape
        mask = jnp.ones((c, 1, 1), dtype=jnp.float32)
        mask = mask.at[c//2:, 0, 0].set(0.0)
        return mask

    def checkerboard_mask(self, shape):
        c, h, w = shape  
        mask = jnp.zeros((1, h, w), dtype=jnp.float32)  # (1, H, W)
        for i in range(h):
            for j in range(w):
                if (i + j) % 2 == 0:
                    mask = mask.at[0, i, j].set(0.0)
                else:
                    mask = mask.at[0, i, j].set(1.0)
        return mask

    def __call__(self, x: jnp.array):
        b = self.mask
        identity_part = b * x
        transformed_part = (1-b) * (x * jnp.exp(self.s(b * x)) + self.t(b * x))
        return identity_part + transformed_part, self.log_jac(x)

    def inverse(self, y):
        b = self.mask
        identity_part = b * y
        transformed_part = (1-b) * ((y - self.t(b * y)) * jnp.exp(-self.s(b * y)))
        return identity_part + transformed_part, - self.log_jac(y)
    def log_jac(self, x):
        # shape is scalar
        return jnp.sum(self.s(self.mask * x))


class squeeze_layer(eqx.Module):
    # divides image into subsquares of shape 2 x 2 x c
    # reshapes them to be shape 1 x 1 x 4c
    def __init__(self):
        pass
        # this joint only reshapes
    def __call__(self, x):
        assert x.ndim == 3
        C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, "Dimensions must be even"
        #  (C, H//2, 2, W//2, 2)
        x = jnp.reshape(x, (C, H//2, 2, W//2, 2))
        # (C, 2, 2, H//2, W//2)
        x = x.transpose(0, 2, 4, 1, 3)
        # (4C, H//2, W//2)
        return jnp.reshape(x, (4*C, H//2, W//2)), 0.0

    def inverse(self, y):
        C4, H, W = y.shape
        assert C4 % 4 == 0, "Channels must be divisible by 4"
        C = C4 // 4
        y = jnp.reshape(y, (C, 2, 2, H, W))
        y = y.transpose(0, 3, 1, 4, 2)
        return jnp.reshape(y, (C, 2*H, 2*W)), 0.0
        
class Block(eqx.Module):
    # i dont know how to write at all, should i use a sequential object or just a tuple
    # maybe eqx sequential would be better later
    layers: tuple
    def __init__(self, key, shape):
        keys = jax.random.split(key, 7)  
        self.layers = (
            coupling_layer(keys[0], checkerboard=True, shape=shape),
            coupling_layer(keys[1], checkerboard=True, shape=shape),
            coupling_layer(keys[2], checkerboard=True, shape=shape),
            squeeze_layer(),
            # squeeeeeze (C, H, W) → (4C, H/2, W/2)
            coupling_layer(keys[4], checkerboard=False, shape=(4*shape[0], shape[1]//2, shape[2]//2)),
            coupling_layer(keys[5], checkerboard=False, shape=(4*shape[0], shape[1]//2, shape[2]//2)),
            coupling_layer(keys[6], checkerboard=False, shape=(4*shape[0], shape[1]//2, shape[2]//2)),
        )
    def __call__(self, x):
        detall = 0
        for layer in self.layers:
            x, det = layer(x)
            detall += det
        return x, detall
    def inverse(self, y):
        detall = 0
        for layer in reversed(self.layers):
            y, det = layer.inverse(y)
            detall += det
        return y, detall


class NVP(eqx.Module):
    blocks: tuple
    def __init__(self, key, shape, num_blocks=3):
        keys = jax.random.split(key, num_blocks)
        
        blocks_list = []
        current_shape = shape
        
        for i in range(num_blocks):
            blocks_list.append(Block(keys[i], current_shape))
            # (C, H, W) → (4C, H/2, W/2)
            c, h, w = current_shape
            new_shape = (4*c, h//2, w//2)
            
            # factor out half the channels except for last block
            if i < num_blocks - 1:
                current_shape = (new_shape[0]//2, new_shape[1], new_shape[2])
            else:
                current_shape = new_shape
        
        self.blocks = tuple(blocks_list)
    def __call__(self, x):
        z_list = []
        total_log_det = 0
        
        for i, block in enumerate(self.blocks):
            x, log_det = block(x)
            total_log_det += log_det
            
            if i < len(self.blocks) - 1:
                # Split channels: first half to z_list, second half continues
                c = x.shape[0]
                z_i = x[:c//2, :, :]  # First half channels
                x = x[c//2:, :, :]    # Second half continues
                z_list.append(z_i)
        
        # Last block output goes to z_list
        z_list.append(x)
        return z_list, total_log_det
    
    def inverse(self, z_list):
        x = z_list[-1]
        
        for i in range(len(self.blocks) - 1, -1, -1):
            # before running inverse block, concatenate factored-out z
            if i < len(self.blocks) - 1:
                x = jnp.concatenate([z_list[i], x], axis=0)
            
            x, _ = self.blocks[i].inverse(x)
        
        return x
    def loss(self, x):
        z_list, log_det = self(x)
                
        z_flat = jnp.concatenate([z.flatten() for z in z_list])
        
        log_pz = -0.5 * jnp.sum(z_flat**2) - 0.5 * z_flat.size * jnp.log(2 * jnp.pi)
        
        log_px = log_pz + log_det
        return -log_px
