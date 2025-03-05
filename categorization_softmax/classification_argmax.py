import mlx.core as mx
import mlx.nn as nn
import time
import matplotlib.pyplot as plt

EPOCHS = 5000
ALPHA = 0.005
BATCH_SIZE = 1000
FEATURES_COUNT = 4  # rgba
NUM_CATEGORIES = 7
INPUTS_COUNT = 10000

def one_hot(labels, num_classes):
    """
    One-hot encodes a batch of labels.

    Args:
        labels (mx.array): A 1D array of shape (N,) containing integer class labels.
        num_classes (int): The number of unique classes.

    Returns:
        mx.array: A 2D array of shape (N, num_classes) with one-hot encoding.
    """
    return mx.equal(mx.arange(num_classes), labels[:, None]).astype(mx.float32)

def get_accuracy(w,b,X,true_labels):
    guess_logits = X @ w + b
    guess_probs = mx.softmax(guess_logits, axis=1)
    guess_labels = mx.argmax(guess_probs,axis=1)
    return mx.mean(guess_labels == true_labels)

def get_loss(w,b,X,true_labels):
    guess_logits = X @ w + b
    losses = nn.losses.cross_entropy(guess_logits,one_hot(true_labels,NUM_CATEGORIES),axis=-1)
    mean = mx.mean(losses)
    return mean

def get_loss_probs(w,b,X,true_probs):
    guess_logits = X @ w + b
    losses = nn.losses.cross_entropy(guess_logits,true_probs,axis=-1)
    mean = mx.mean(losses)
    return mean

grad_neg_ll_w = mx.grad(get_loss, argnums=0)
grad_neg_ll_b = mx.grad(get_loss, argnums=1)
grad_neg_ll_w_prob = mx.grad(get_loss_probs, argnums=0)
grad_neg_ll_b_prob = mx.grad(get_loss_probs, argnums=1)

def update(w_guess, b_guess, X_batch, Y_batch):
    w_grad = grad_neg_ll_w(w_guess, b_guess, X_batch, Y_batch)
    b_grad = grad_neg_ll_b(w_guess, b_guess, X_batch, Y_batch)
    w_guess = w_guess - ALPHA * w_grad
    b_guess = b_guess - ALPHA * b_grad
    return w_guess, b_guess

def train(X_features, labels, w_guess, b_guess,):
    for idx in range(0, len(labels), BATCH_SIZE):
        X_batch = X_features[idx:idx+BATCH_SIZE]
        Y_batch = labels[idx:idx+BATCH_SIZE]
        w_guess, b_guess = update(w_guess, b_guess, X_batch, Y_batch)
    return w_guess, b_guess

def update_probs(w_guess, b_guess, X_batch, Y_batch):
    w_grad = grad_neg_ll_w_prob(w_guess, b_guess, X_batch, Y_batch)
    b_grad = grad_neg_ll_b_prob(w_guess, b_guess, X_batch, Y_batch)
    w_guess = w_guess - ALPHA * w_grad
    b_guess = b_guess - ALPHA * b_grad
    return w_guess, b_guess

def train_probs(X_features, probabilities, w_guess, b_guess,):
    for idx in range(0, len(probabilities), BATCH_SIZE):
        X_batch = X_features[idx:idx+BATCH_SIZE]
        Y_batch = probabilities[idx:idx+BATCH_SIZE]
        w_guess, b_guess = update_probs(w_guess, b_guess, X_batch, Y_batch)
    return w_guess, b_guess

def classification():
    start_time = time.time()
    X = mx.random.normal((INPUTS_COUNT, FEATURES_COUNT))
    w_actual = mx.random.normal((FEATURES_COUNT, NUM_CATEGORIES))
    w_guess = mx.random.normal((FEATURES_COUNT, NUM_CATEGORIES))
    w_guess_prob = w_guess

    b_actual = mx.random.normal(shape=((NUM_CATEGORIES,)))
    b_guess = mx.random.normal(shape=((NUM_CATEGORIES,)))
    b_guess_prob = b_guess
    true_outputs = X @ w_actual + b_actual
    true_probs = mx.softmax(true_outputs, axis=1)
    true_labels = mx.argmax(true_probs,axis=1)
    diffs_w=[]    
    diffs_b=[]
    diffs_w_prob=[]    
    diffs_b_prob=[]


    loss_history = []
    loss_history_probs = []
    for epoch in range(0, EPOCHS):
        # Shuffle data
        shuffled_indices = mx.random.permutation(mx.arange(len(X)))
        X_shuffled = X[shuffled_indices]
        labels_shuffled = true_labels[shuffled_indices]  
        probabilities_shuffled = true_probs[shuffled_indices]
        loss = get_loss(w_guess, b_guess, X, true_labels)
        prob_loss = get_loss_probs(w_guess_prob,b_guess_prob,X,true_probs)
        loss_history.append(mx.sum(loss))
        loss_history_probs.append(prob_loss)
        diff_w = mx.linalg.norm(w_actual - w_guess)
        diff_b = mx.linalg.norm(b_actual - b_guess)
        diffs_w.append(diff_w)
        diffs_b.append(diff_b)
        diff_w_prob = mx.linalg.norm(w_actual - w_guess_prob)
        diff_b_prob = mx.linalg.norm(b_actual - b_guess_prob)
        diffs_w_prob.append(diff_w_prob)
        diffs_b_prob.append(diff_b_prob)
        w_guess, b_guess = train(X_shuffled, labels_shuffled, w_guess, b_guess)
        w_guess_prob, b_guess_prob = train_probs(X_shuffled, probabilities_shuffled, w_guess_prob, b_guess_prob)

        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {mx.sum(loss):.6f} probLoss = {prob_loss}")
            print(get_accuracy(w_guess,b_guess,X,true_labels))
            print(get_accuracy(w_guess_prob,b_guess_prob,X,true_labels))
    
    mx.eval(b_guess, w_guess)
    loss = get_loss(w_guess, b_guess, X, true_labels)
    loss_history.append(mx.sum(loss))

    elapsed_time = time.time() - start_time
    print(f"After {EPOCHS} epochs: Loss = {loss}")
    print(f"Execution time: {elapsed_time:.4f} seconds")

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Training Loss w/ CE on argmax classes")
    plt.plot(diffs_w, label="Weight Difference")
    plt.plot(diffs_b, label="Bias Difference")
    plt.plot(diffs_w_prob, label="Weight Difference for prob approach")
    plt.plot(diffs_b_prob, label="Bias Difference for prob approach")
    plt.plot(loss_history_probs, label="Training Loss w/ entropy on probs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.show()

classification()
