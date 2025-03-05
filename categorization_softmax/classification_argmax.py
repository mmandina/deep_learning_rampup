import mlx.core as mx
import mlx.nn as nn
import time
import matplotlib.pyplot as plt

EPOCHS = 1000
ALPHA = 0.01
BATCH_SIZE = 500
FEATURES_COUNT = 4  # rgba
NUM_CATEGORIES = 7
INPUTS_COUNT = 1000

def get_loss(w,b,X,true_labels):
    guess_logits = X @ w + b
    return mx.mean(nn.losses.cross_entropy(guess_logits,true_labels))

grad_neg_ll_w = mx.grad(get_loss, argnums=0)
grad_neg_ll_b = mx.grad(get_loss, argnums=1)

def update(w_guess, b_guess, X_batch, Y_batch):
    w_grad = grad_neg_ll_w(w_guess, b_guess, X_batch, Y_batch)
    b_grad = grad_neg_ll_b(w_guess, b_guess, X_batch, Y_batch)
    w_guess = w_guess - ALPHA * w_grad
    b_guess = b_guess - ALPHA * b_grad
    return w_guess, b_guess

def train(X_features, labels, w_guess, b_guess):
    for idx in range(0, len(labels), BATCH_SIZE):
        X_batch = X_features[idx:idx+BATCH_SIZE]
        Y_batch = labels[idx:idx+BATCH_SIZE]
        w_guess, b_guess = update(w_guess, b_guess, X_batch, Y_batch)
    return w_guess, b_guess

def classification():
    start_time = time.time()
    X = mx.random.normal((INPUTS_COUNT, FEATURES_COUNT))
    w_actual = mx.random.normal((FEATURES_COUNT, NUM_CATEGORIES))
    w_guess = mx.random.normal((FEATURES_COUNT, NUM_CATEGORIES))

    b_actual = mx.random.normal(shape=((NUM_CATEGORIES,)))
    b_guess = mx.random.normal(shape=((NUM_CATEGORIES,)))

    true_outputs = X @ w_actual + b_actual
    true_labels = mx.argmax(true_outputs, axis=1)

    loss_history = []
    for epoch in range(0, EPOCHS):
        # Shuffle data
        shuffled_indices = mx.random.permutation(mx.arange(len(X)))
        X_shuffled = X[shuffled_indices]
        labels_shuffled = true_labels[shuffled_indices]  
        
        loss = get_loss(w_guess, b_guess, X, true_labels)
        print(loss)
        loss_history.append(mx.sum(loss))
        
        w_guess, b_guess = train(X_shuffled, labels_shuffled, w_guess, b_guess)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {mx.sum(loss):.6f}")
    
    mx.eval(b_guess, w_guess)
    loss = get_loss(w_guess, b_guess, X, true_labels)
    loss_history.append(mx.sum(loss))

    elapsed_time = time.time() - start_time
    print(f"After {EPOCHS} epochs: Loss = {loss}")
    print(f"Execution time: {elapsed_time:.4f} seconds")

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.show()

classification()
