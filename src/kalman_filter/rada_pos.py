import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Time parameters
dt = 1.0  # Time step (1 second)
n_steps = 1000  # Number of time steps

# Initialize arrays for true state, measurements, and time
time = np.arange(0, n_steps) * dt
true_position = np.zeros(n_steps)
true_velocity = np.zeros(n_steps)
measurements = np.zeros(n_steps)

# Create realistic aircraft trajectory
# Initial conditions
true_position[0] = 30000  # Initial position (30,000 m)
true_velocity[0] = 40     # Initial velocity (40 m/s)

# Generate true trajectory (aircraft with varying velocity)
for i in range(1, n_steps):
    # Add some small random acceleration
    acceleration = np.random.normal(0, 0.5)
    
    # Update velocity with a small random acceleration
    true_velocity[i] = true_velocity[i-1] + acceleration * dt
    
    # Update position based on velocity
    true_position[i] = true_position[i-1] + true_velocity[i-1] * dt

# Generate noisy measurements (radar measurements have noise)
# Tăng nhiễu lên để tạo sự khác biệt lớn hơn (phục vụ hiển thị)
measurement_noise = 300  # Tăng từ 100 lên 300 để làm nổi bật sự khác biệt
measurements = true_position + np.random.normal(0, measurement_noise, n_steps)

# Implement Kalman filter for position and velocity tracking
def kalman_filter(measurements, dt, R, Q):
    n = len(measurements)
    estimated_positions = np.zeros(n)
    estimated_velocities = np.zeros(n)
    
    # Initial state
    x = np.array([measurements[0], 40.0])  # [position, velocity]
    
    # Initial covariance matrix
    P = np.array([[100.0, 0], [0, 100.0]])
    
    # State transition matrix
    F = np.array([[1, dt], [0, 1]])
    
    # Measurement matrix (we only measure position)
    H = np.array([[1, 0]])
    
    # Measurement covariance
    R = np.array([[R]])
    
    for i in range(n):
        # Prediction step
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        
        # Update step
        y = measurements[i] - H @ x_pred  # Measurement residual
        S = H @ P_pred @ H.T + R  # Residual covariance
        K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
        
        # Update state and covariance
        x = x_pred + K @ y
        P = (np.eye(2) - K @ H) @ P_pred
        
        # Store estimates
        estimated_positions[i] = x[0]
        estimated_velocities[i] = x[1]
    
    return estimated_positions, estimated_velocities

# Configure Kalman Filter parameters
R = measurement_noise**2  # Measurement noise covariance
# Process noise covariance matrix
Q = np.array([
    [dt**4/4, dt**3/2],
    [dt**3/2, dt**2]
]) * 1.0  # Process noise intensity

# Run Kalman filter
estimated_positions, estimated_velocities = kalman_filter(measurements, dt, R, Q)

# Calculate errors
position_rmse = np.sqrt(np.mean((estimated_positions - true_position)**2))
measurement_rmse = np.sqrt(np.mean((measurements - true_position)**2))

# Chỉ vẽ một số điểm để tránh đám đông và nhìn rõ hơn
sample_rate = 10  # Vẽ 1 điểm cho mỗi 10 phép đo

# Plot results với các thay đổi để hiển thị rõ ràng
plt.figure(figsize=(15, 8))

# Plot position tracking - CHỈ VẼ BIỂU ĐỒ VỊ TRÍ
plt.plot(time[::sample_rate], measurements[::sample_rate], 'bo', markersize=4, 
         label='Radar Measurements', alpha=0.7)
plt.plot(time, estimated_positions, 'r-', linewidth=2.5, 
         label='Kalman Filter Estimate')
plt.plot(time, true_position, 'g-', linewidth=1.5, 
         label='True Position')

# Thêm các điểm đánh dấu để chỉ ra sự khác biệt rõ hơn
plt.plot(time[0:200:50], measurements[0:200:50], 'bo', markersize=8, alpha=1.0)
plt.plot(time[0:200:50], estimated_positions[0:200:50], 'ro', markersize=6)
plt.plot(time[0:200:50], true_position[0:200:50], 'go', markersize=6)

plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Position (m)', fontsize=14)
plt.title(f'Aircraft Position Tracking\nMeasurement RMSE: {measurement_rmse:.2f}m, Kalman Filter RMSE: {position_rmse:.2f}m', 
          fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=14, loc='upper left')

# Thêm văn bản để nhấn mạnh sự khác biệt
plt.annotate('Radar measurements (noisy)', xy=(150, measurements[150]+1000), 
             xytext=(150, measurements[150]+3000),
             arrowprops=dict(facecolor='blue', shrink=0.05), fontsize=12)

plt.annotate('Kalman filter estimate', xy=(250, estimated_positions[250]), 
             xytext=(250, estimated_positions[250]-3000),
             arrowprops=dict(facecolor='red', shrink=0.05), fontsize=12)

# Chỉ tập trung vào phần đầu để thấy rõ sự khác biệt
# Uncomment dòng dưới đây nếu muốn phóng to phần đầu của biểu đồ
# plt.xlim(0, 300)
# plt.ylim(29000, 45000)

plt.tight_layout()
plt.savefig("aircraft_tracking_visible_data.png", format="png", dpi=300)
plt.show()  # Hiển thị biểu đồ

print(f"Measurement RMSE: {measurement_rmse:.2f} meters")
print(f"Kalman Filter RMSE: {position_rmse:.2f} meters")