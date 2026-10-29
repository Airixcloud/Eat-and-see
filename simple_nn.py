import numpy as np

# Input
x = np.array([2.0, 3.0])

# Correct answer
y = 20.0

# Weights for two neurons
w1 = np.array([1.0, 1.0])
w2 = np.array([1.0, 1.0])

# Output weights
wo = np.array([1.0, 1.0])

# Biases
b1 = 0.0
b2 = 0.0
bo = 0.0

learning_rate = 0.01

for epoch in range(1000):

    # ----------------
    # Forward pass
    # ----------------

    z1 = np.dot(x, w1) + b1
    z2 = np.dot(x, w2) + b2

    h1 = np.maximum(0, z1)  # ReLU
    h2 = np.maximum(0, z2)

    prediction = h1 * wo[0] + h2 * wo[1] + bo

    # ----------------
    # Loss
    # ----------------

    loss = (prediction - y) ** 2

    # ----------------
    # Backpropagation
    # ----------------

    dL_dprediction = 2 * (prediction - y)

    # Output weights
    dL_dwo = dL_dprediction * np.array([h1, h2])

    # Hidden neurons
    dL_dh1 = dL_dprediction * wo[0]
    dL_dh2 = dL_dprediction * wo[1]

    # ReLU derivative
    dz1 = dL_dh1 if z1 > 0 else 0
    dz2 = dL_dh2 if z2 > 0 else 0

    # Hidden weights
    dL_dw1 = dz1 * x
    dL_dw2 = dz2 * x

    # Hidden biases
    dL_db1 = dz1
    dL_db2 = dz2

    # Output bias
    dL_dbo = dL_dprediction

    # ----------------
    # Update weights
    # ----------------

    w1 -= learning_rate * dL_dw1
    w2 -= learning_rate * dL_dw2

    wo -= learning_rate * dL_dwo

    b1 -= learning_rate * dL_db1
    b2 -= learning_rate * dL_db2
    bo -= learning_rate * dL_dbo

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}: "
            f"prediction={prediction:.4f}, "
            f"loss={loss:.4f}"
        )