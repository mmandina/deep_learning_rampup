import time
import mlx.core as mx
import matplotlib.pyplot as plt


EPOCHS = 1000
ALPHA = 0.01
BATCH_SIZE = 500
FEATURES_COUNT = 1000
INPUTS_COUNT = 10000

def mlx_mse(w,b,X,y):
    preds = X @ w + b
    return mx.mean(mx.square(preds-y))

grad_w = mx.grad(mlx_mse, argnums=0)
grad_b = mx.grad(mlx_mse, argnums=1)

def update(w_guess,b_guess,X_batch,Y_batch):
    w_grad = grad_w(w_guess,b_guess,X_batch,Y_batch)
    b_grad = grad_b(w_guess,b_guess,X_batch,Y_batch)
    w_guess = w_guess - ALPHA*w_grad
    b_guess = b_guess - ALPHA*b_grad
    return w_guess,b_guess

def train(X_features,true_outputs,w_guess,b_guess):
    for idx in range(0,len(true_outputs),BATCH_SIZE):
        X_batch = X_features[idx:idx+BATCH_SIZE]
        Y_batch = true_outputs[idx:idx+BATCH_SIZE]
        w_guess,b_guess = update(w_guess,b_guess,X_batch,Y_batch)
    return w_guess,b_guess

def stochastic_linear_regression():
    start_time = time.time()
    X_features = mx.random.normal((INPUTS_COUNT,FEATURES_COUNT))
    w_actual = mx.random.normal((FEATURES_COUNT,))
    b_actual = mx.random.normal(shape=(()))

    w_guess = mx.random.normal((FEATURES_COUNT,))
    b_guess = mx.random.normal(shape=(()))
    
    true_outputs = X_features@w_actual + b_actual
    loss_history = []
    for epoch in range(0,EPOCHS):
        shuffled_indices = mx.random.permutation(mx.arange(len(X_features)))
        X_shuffled = X_features[shuffled_indices]
        y_shuffled = true_outputs[shuffled_indices]

        outputs_mse = mlx_mse(w_guess,b_guess,X_features,true_outputs)
        loss_history.append(outputs_mse.item())
        w_guess, b_guess = train(X_shuffled, y_shuffled, w_guess, b_guess)
        mx.eval(w_guess)
        mx.eval(b_guess)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {outputs_mse.item():.6f}")
    elapsed_time = time.time() - start_time
    print(f"After {EPOCHS} epochs: Loss = {outputs_mse}")
    print(f"Execution time: {elapsed_time:.4f} seconds")


    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.show()


stochastic_linear_regression()
        
        
