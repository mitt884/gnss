import numpy as np
import matplotlib.pyplot as plt

def kalman_filter_gold_bar(num_iterations, true_weight, process_noise, measurement_noise):
    """
    Kalman filter implementation for estimating gold bar weight.
    
    Parameters:
    - num_iterations: number of measurements/iterations
    - true_weight: actual weight of gold bar
    - process_noise: variance in process (Q)
    - measurement_noise: variance in measurements (R)
    
    Returns:
    - measurements: array of noisy measurements
    - estimates: array of kalman filter estimates
    - true_values: array of true weights
    """
    # Initialize arrays
    measurements = np.zeros(num_iterations)
    estimates = np.zeros(num_iterations)
    true_values = np.full(num_iterations, true_weight)
    
    # Initial state estimate
    x_est = 1000.0  # Initial estimate x^0,0 = 1000g
    
    # Initial error covariance
    P = 1.0
    
    # Generate measurements with noise
    measurements = true_weight + np.random.normal(0, np.sqrt(measurement_noise), num_iterations)
    
    # Kalman filter iteration
    for i in range(num_iterations):
        # Prediction step (static system)
        x_pred = x_est  # x^k,0 = x^(k-1),0 (no change in prediction for static system)
        P_pred = P + process_noise  # Process noise increases uncertainty
        
        # Update step
        K = P_pred / (P_pred + measurement_noise)  # Kalman gain
        x_est = x_pred + K * (measurements[i] - x_pred)  # Updated estimate
        P = (1 - K) * P_pred  # Updated error covariance
        
        # Store estimate
        estimates[i] = x_est
    
    return measurements, estimates, true_values

# Set parameters
np.random.seed(42)  # For reproducibility
num_iterations = 1000  # Number of measurements
true_weight = 1000.0  # True weight of gold bar in grams
process_noise = 0.01  # Small process noise (Q)
measurement_noise = 25.0  # Measurement noise variance (R)

# Run Kalman filter
measurements, estimates, true_values = kalman_filter_gold_bar(
    num_iterations, true_weight, process_noise, measurement_noise
)

# Create iteration points
iterations = np.arange(1, num_iterations + 1)

# Calculate errors
mse_raw = np.mean((measurements - true_values)**2)
mse_kalman = np.mean((estimates - true_values)**2)

# Plot results
plt.figure(figsize=(12, 8))

plt.plot(iterations, measurements, 'b-', alpha=0.3, label='Measurements')
plt.plot(iterations, estimates, 'r-', label='Kalman Filter Estimates')
plt.plot(iterations, true_values, 'g-', label='True Weight')

# Highlight the beginning for visibility
plt.plot(iterations[:20], measurements[:20], 'b-', linewidth=2)
plt.plot(iterations[:20], estimates[:20], 'r-', linewidth=2)

plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Weight (g)', fontsize=14)
plt.title('Gold Bar Weight Estimation', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()

# Add text with MSE information
plt.figtext(0.15, 0.15, f"Measurement MSE: {mse_raw:.2f}\nKalman Filter MSE: {mse_kalman:.2f}", 
            bbox=dict(facecolor='white', alpha=0.8))

# Zoom in to better see the variations
y_min = min(np.min(measurements), np.min(estimates)) - 5
y_max = max(np.max(measurements), np.max(estimates)) + 5
plt.ylim(y_min, y_max)

plt.tight_layout()

# Save as PNG
output_file = "gold_weight.png"
plt.savefig(output_file, format="png", dpi=300, bbox_inches='tight')
plt.close()

# Print summary statistics
print(f"Initial estimate: 1000.0 g")
print(f"True weight: {true_weight} g")
print(f"Final Kalman estimate: {estimates[-1]:.2f} g")
print(f"Measurement MSE: {mse_raw:.2f}")
print(f"Kalman Filter MSE: {mse_kalman:.2f}")
print(f"Improvement: {100 * (mse_raw - mse_kalman) / mse_raw:.2f}%")
print(f"The plot has been saved as '{output_file}'.")
