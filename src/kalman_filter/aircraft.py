import numpy as np
import matplotlib.pyplot as plt
import os

# Initialize parameters
np.random.seed(42)
T = 3030  # Number of time steps (1000s)
Delta_t = 1.0  # Time interval per step (1s)

# Initialize state transition matrix F (6x6)
F = np.array([
    [1, 0, 0, Delta_t, 0, 0],
    [0, 1, 0, 0, Delta_t, 0],
    [0, 0, 1, 0, 0, Delta_t],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
])

# Control matrix G (6x3)
G = np.array([
    [0.5 * Delta_t**2, 0, 0],
    [0, 0.5 * Delta_t**2, 0],
    [0, 0, 0.5 * Delta_t**2],
    [Delta_t, 0, 0],
    [0, Delta_t, 0],
    [0, 0, Delta_t]
])

# Measurement matrix H (3x6)
H = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0]
])

# Covariance matrices
Q = np.eye(6) * 0.01
R = np.eye(3) * 10
P = np.eye(6) * 500

# Initialize states
x_est = np.zeros((6, 1))
x_true = np.zeros((6, 1))

# Lists for recording
true_positions = []
est_positions = []
measurements = []

# Time loop
for t in range(T):
    # Random acceleration to make the airplane fly dynamically
    u = np.random.randn(3, 1) * 2.0

    # Add a sinusoidal component to simulate more lively movement
    u[0] += 5.0 * np.sin(0.01 * t)
    u[1] += 5.0 * np.cos(0.01 * t)
    u[2] += 2.0 * np.sin(0.02 * t)

    # Update true state
    x_true = F @ x_true + G @ u

    # Generate measurement with noise
    v = np.random.multivariate_normal(mean=[0, 0, 0], cov=R).reshape(3, 1)
    z = H @ x_true + v

    # === Kalman Filter: Predict ===
    x_est = F @ x_est + G @ u
    P = F @ P @ F.T + Q

    # === Kalman Filter: Update ===
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x_est = x_est + K @ (z - H @ x_est)
    P = (np.eye(6) - K @ H) @ P

    # Save data
    true_positions.append(x_true[:3, 0])
    est_positions.append(x_est[:3, 0])
    measurements.append(z[:, 0])

# Convert to numpy arrays
true_positions = np.array(true_positions)
est_positions = np.array(est_positions)
measurements = np.array(measurements)

# Plot the results
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(true_positions[:, 0], true_positions[:, 1], true_positions[:, 2], label='True Position', color='blue')
ax.plot(est_positions[:, 0], est_positions[:, 1], est_positions[:, 2], label='Kalman Estimate', color='red')
ax.scatter(measurements[:, 0], measurements[:, 1], measurements[:, 2], label='Measurements', color='green', s=1)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Airplane Dynamic Flying with Kalman Filter')
ax.legend()

# Save the plot as PNG file with incrementing number
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
existing_files = [f for f in os.listdir(output_dir) if f.startswith('random_') and f.endswith('.png')]
file_numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files if f.split('_')[1].split('.')[0].isdigit()]
next_number = max(file_numbers, default=0) + 1

# Limit number of saved files to 100
if len(file_numbers) >= 100:
    oldest_file = sorted(existing_files, key=lambda x: int(x.split('_')[1].split('.')[0]))[0]
    os.remove(os.path.join(output_dir, oldest_file))

filename = os.path.join(output_dir, f'random_{next_number}.png')

plt.savefig(filename)
plt.close(fig)