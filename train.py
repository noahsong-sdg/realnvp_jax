# train.py
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from layers import NVP, unpickle
import time

# Load CIFAR-10 batch
def load_cifar_batch(filename):
    data = unpickle(filename)
    images = data[b'data']  # (10000, 3072)
    images = images.reshape(-1, 3, 32, 32)  # (N, C, H, W), dataset has channels last
    images = images.astype(jnp.float32)
    images = (images / 255.0) * 2.0 - 1.0 # norms to [-1 1]
    return images

def batch_loss(model, x_batch):
    losses = jax.vmap(model.loss)(x_batch)
    return jnp.mean(losses)


@eqx.filter_jit
def train_step(model, opt_state, x_batch, optimizer):
    loss_fn = lambda m: batch_loss(m, x_batch)
    loss, grads = jax.value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

def train():
    batch_size = 64
    num_epochs = 10
    learning_rate = 1e-5
    weight_decay = 1e-5
    
    # Initialize model
    key = jax.random.PRNGKey(0)
    shape = (3, 32, 32)
    num_blocks = 3
    model = NVP(key, shape, num_blocks)
    
    print(f"Model initialized with {num_blocks} blocks")
    print(f"Input shape: {shape}")
    
    # Load training data
    print("\nLoading CIFAR-10...")
    train_images = []
    for i in range(1, 6):
        batch = load_cifar_batch(f'cifar-10-batches-py/data_batch_{i}')
        train_images.append(batch)
    train_images = jnp.concatenate(train_images, axis=0)
    print(f"Loaded {train_images.shape[0]} training images")
    print(f"Image shape: {train_images.shape[1:]}, range: [{train_images.min():.2f}, {train_images.max():.2f}]")

    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    num_samples = train_images.shape[0]
    num_batches = num_samples // batch_size
    
    print(f"\nTraining: {num_epochs} epochs, {num_batches} batches/epoch, batch_size={batch_size}")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        # Shuffle 
        key = jax.random.PRNGKey(epoch)
        perm = jax.random.permutation(key, num_samples)
        train_images_shuffled = train_images[perm]

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            x_batch = train_images_shuffled[start_idx:end_idx]
            
            model, opt_state, loss = train_step(model, opt_state, x_batch, optimizer)
            epoch_loss += loss
            
            if (batch_idx + 1) % 100 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{num_batches}, Loss: {avg_loss:.4f}")
        
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Average Loss: {avg_epoch_loss:.4f}")

    return model

if __name__ == "__main__":
    trained_model = train()
