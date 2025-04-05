import re
from datetime import datetime
import json
import math
import numpy as np
import pyproj
from scipy.optimize import least_squares
from scipy.linalg import inv

def ecef_to_geodetic(x, y, z):
    """
    Chuyển đổi tọa độ ECEF sang tọa độ Geodetic (vĩ độ, kinh độ, cao độ)
    sử dụng API mới của pyproj
    """
    # Sử dụng Transformer - cách được khuyến nghị trong pyproj 2+
    transformer = pyproj.Transformer.from_crs(
        {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
        {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'},
        always_xy=True
    )
    
    # always_xy=True đảm bảo thứ tự lon, lat (không phải lat, lon)
    lon, lat, alt = transformer.transform(x, y, z)
    return lat, lon, alt

# Hàm tính toán vị trí bộ thu
def compute_receiver_position(
    sat_positions, pseudoranges, c=299792458, max_iters=1000, tol=1e-6
):
    """
    Tính toán vị trí của bộ thu GPS sử dụng Least Squares.

    Parameters:
    - sat_positions: numpy array (n x 3) chứa tọa độ (x, y, z) của n vệ tinh
    - pseudoranges: numpy array (n x 1) chứa khoảng cách giả đo R^j
    - c: tốc độ ánh sáng (m/s)
    - max_iters: số lần lặp tối đa
    - tol: ngưỡng hội tụ

    Returns:
    - receiver_position: numpy array (1 x 3) tọa độ (x, y, z) của bộ thu
    - clock_bias: sai số đồng hồ c * δt 
    """
    if len(sat_positions) < 4:
        raise ValueError("Cần ít nhất 4 vệ tinh để tính toán vị trí.")

    # Chuyển đổi đầu vào thành numpy array với kiểu float64
    sat_positions = np.array(sat_positions, dtype=np.float64)
    pseudoranges = np.array(pseudoranges, dtype=np.float64).reshape(-1, 1)
    x0 =0
    y0 =0
    z0 =0
    bu = 0
    # Khởi tạo vị trí gần đúng (tại tâm Trái Đất) với kiểu float64
    solution = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    
    for iter_num in range(max_iters):
        # Tính khoảng cách hình học dựa trên ước lượng vị trí hiện tại
        rho = np.sqrt(
            (sat_positions[:, 0] - solution[0]) ** 2 +
            (sat_positions[:, 1] - solution[1]) ** 2 +
            (sat_positions[:, 2] - solution[2]) ** 2
        )
        
        # Tránh chia cho 0
        rho[rho < 1e-10] = 1e-10
                
        # Xây dựng ma trận thiết kế
        G = np.column_stack([
            -(sat_positions[:, 0] - solution[0]) / rho,
            -(sat_positions[:, 1] - solution[1]) / rho,
            -(sat_positions[:, 2] - solution[2]) / rho,
            np.ones_like(rho)  # Cột sai số đồng hồ
        ])
        # print("size ", G)
        residuals = pseudoranges.flatten()
        # print("residual", residuals)
        # print("pseudorang", pseudoranges)
        G_transpose = G.T
        invGandGT = inv(G_transpose @ G)
        delta_rho =  residuals - rho #* rho - residuals
        # print("drho", delta_rho)


        # Tính toán hiệu chỉnh sử dụng phương trình normal
        try:
            # Đảm bảo delta là float64
            delta = np.linalg.solve(G, delta_rho).astype(np.float64) #solve Gx = y with .solve(G,y)| G.T @ G, G.T @ delta_rho
        except np.linalg.LinAlgError:
            # Sử dụng pseudoinverse nếu ma trận là singular
            delta = (np.linalg.pinv(G) @ delta_rho).astype(np.float64)
        
        # Cập nhật giải pháp (cả hai giờ đều là float64)
        solution = solution + delta.flatten()
        
        # Kiểm tra hội tụ
        if np.linalg.norm(delta) < tol:
            break
            
    # Trích xuất kết quả
    receiver_position = solution[:3]
    clock_bias = solution[3]
    
    return receiver_position, clock_bias


# Đọc tệp RINEX
file = open("OBS.rnx", "r")

# Đọc dữ liệu quỹ đạo từ tệp JSON
with open("outputOrbit.json", "r") as orbit_file:
    orbit_data = json.load(orbit_file)

# Chuyển đổi dữ liệu quỹ đạo thành từ điển để dễ tra cứu
orbit_lookup = {item["Sattelite"]: item for item in orbit_data}

sat_system_code = {}
time_of_first_obs = {}
sys_phase_shift = {}
phase_shifts = {}
epoches = {}

epoch_count = 0
receiver_positions = {}  # Lưu trữ vị trí bộ thu tại mỗi epoch

# Đọc phần đầu
for line in file:
    if "RINEX VERSION / TYPE" in line:
        ver = line[0:15].strip()
        type = line[42:55].strip()
    if "REC # / TYPE / VERS" in line:
        receiver_number = line[0:20].strip()
        receiver_type = line[20:40].strip()
        recerver_ver = line[40:60].strip()
    if "APPROX POSITION XYZ" in line:
        app_x = line[0:14].strip()  # x0
        app_y = line[14:28].strip()  # y0
        app_z = line[28:42].strip()  # z0
        print("x0: ", app_x)
        print("y0: ", app_y)
        print("z0: ", app_z)
    if "ANTENNA: DELTA H/E/N" in line:
        ant_h = line[0:14].strip()
        ant_e = line[14:28].strip()
        ant_n = line[28:42].strip()
    if "SYS / # / OBS TYPES" in line:
        sat_system_code["System code"] = line[0:1].strip()
        sat_system_code["Number of Observation"] = line[2:7].strip()
        sat_system_code["Observation types"] = line[7:55].strip()
    if "INTERVAL" in line:
        interval = line[0:11].strip()
    if "TIME OF FIRST OBS" in line:
        time_of_first_obs["Time system"] = line[48:52].strip()
        time_of_first_obs["Time of first observation"] = line[0:45].strip()

        time_parts = list(
            map(float, time_of_first_obs["Time of first observation"].split())
        )

        year, month, day, hour, minute = map(int, time_parts[:5])
        second = time_parts[5]

        time_obj = datetime(
            year, month, day, hour, minute, int(second), int((second % 1) * 1e6)
        )

        time_of_first_obs["Time of first observation"] = time_obj.strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )
    if "SYS / PHASE SHIFT" in line:
        system = line[0:1].strip()
        carrier_phase = line[2:6].strip()
        phase_shift_correction = line[7:15].strip()
        if system not in phase_shifts:
            phase_shifts[system] = {}

        phase_shifts[system][carrier_phase] = phase_shift_correction

    if "SIGNAL STRENGTH UNIT" in line:
        signal_strength_unit = line[0:5].strip()

    if "END OF HEADER" in line:
        break

# Đọc và xử lý các epoch
previous_epoch = None

for line in file:
    if line[0:1] == ">":  # epoch mới
        # Tính toán vị trí cho epoch trước đó (nếu có)
        if previous_epoch is not None:
            epoch_data = epoches[previous_epoch]
            sat_pos = []
            pseudorange_values = []

            for sattelite, data in epoch_data["Satellites"].items():
                if "Orbit Data" in data and "C1 Observable" in data:
                    try:
                        sat_pos.append([
                            float(data["Orbit Data"]["X"]),
                            float(data["Orbit Data"]["Y"]),
                            float(data["Orbit Data"]["Z"])
                        ])
                        pseudorange_values.append(float(data["C1 Observable"]))
                    except (ValueError, TypeError):
                        print(f"Bỏ qua vệ tinh {sattelite} do dữ liệu không hợp lệ")
            
            if len(sat_pos) >= 4:
                try:
                    receiver_position, clock_bias = compute_receiver_position(sat_pos, pseudorange_values)
                    receiver_positions[previous_epoch] = {
                        "ECEF": receiver_position,
                        "Clock Bias": clock_bias,
                        "Time": epoch_data["Time of epoch"]
                    }
                    
                    # Chuyển đổi sang tọa độ địa lý
                    lat, lon, alt = ecef_to_geodetic(receiver_position[0], receiver_position[1], receiver_position[2])
                    receiver_positions[previous_epoch]["Geodetic"] = {
                        "Latitude": lat,
                        "Longitude": lon,
                        "Altitude": alt
                    }
                    
                    print(f"Epoch {previous_epoch} - Vị trí: ECEF({receiver_position[0]:.2f}, {receiver_position[1]:.2f}, {receiver_position[2]:.2f}) m")
                    print(f"              Tọa độ địa lý: ({lat:.6f}°, {lon:.6f}°, {alt:.2f} m)")
                    print(f"              Sai số đồng hồ: {clock_bias:.2f} m")
                except Exception as e:
                    print(f"Lỗi khi tính toán vị trí cho epoch {previous_epoch}: {str(e)}")
            else:
                print(f"Không đủ vệ tinh hợp lệ ({len(sat_pos)}/4) cho epoch {previous_epoch}")
        
        # Bắt đầu xử lý epoch mới
        epoch_count += 1
        epoch_time = line[2:30].strip()
        epoch_time_parts = list(map(float, epoch_time.split()))

        epochYear, epochMonth, epochDay, epochHour, epochMinute = map(
            int, epoch_time_parts[:5]
        )
        epochSecond = epoch_time_parts[5]

        epoch_timeObj = datetime(
            epochYear,
            epochMonth,
            epochDay,
            epochHour,
            epochMinute,
            int(epochSecond),
            int((epochSecond % 1) * 1e6),
        )

        epoches[epoch_count] = {
            "Time of epoch": epoch_timeObj.strftime("%Y-%m-%d %H:%M:%S"),
            "Epoch flag": line[30:33].strip(),
            "Numbers of sat": line[33:36].strip(),
            "Receiver clock offset": line[36:57].strip(),
            "Satellites": {},  # Lưu trữ dữ liệu vệ tinh
        }
        
        previous_epoch = epoch_count
    else:
        sattelite = line[0:4].strip()
        if epoch_count not in epoches:
            continue

        if sattelite not in epoches[epoch_count]["Satellites"]:
            epoches[epoch_count]["Satellites"][sattelite] = {}

        epoches[epoch_count]["Satellites"][sattelite]["C1 Observable"] = line[5:18].strip()
        epoches[epoch_count]["Satellites"][sattelite]["C1 Signal Strength Indicator"] = line[18:19].strip()
        epoches[epoch_count]["Satellites"][sattelite]["L1 Observable"] = line[20:33].strip()
        epoches[epoch_count]["Satellites"][sattelite]["LLI"] = line[33:34].strip()
        epoches[epoch_count]["Satellites"][sattelite]["L1 Signal Strength Indicator"] = line[34:35].strip()

        if sattelite in orbit_lookup and all(k in orbit_lookup[sattelite] for k in ["X", "Y", "Z"]):
            epoches[epoch_count]["Satellites"][sattelite]["Orbit Data"] = {
                "X": orbit_lookup[sattelite]["X"],
                "Y": orbit_lookup[sattelite]["Y"],
                "Z": orbit_lookup[sattelite]["Z"],
            }

# Xử lý epoch cuối cùng nếu chưa được xử lý
if previous_epoch is not None and previous_epoch not in receiver_positions:
    epoch_data = epoches[previous_epoch]
    sat_pos = []
    pseudorange_values = []

    for sattelite, data in epoch_data["Satellites"].items():
        if "Orbit Data" in data and "C1 Observable" in data:
            try:
                sat_pos.append([
                    float(data["Orbit Data"]["X"]),
                    float(data["Orbit Data"]["Y"]),
                    float(data["Orbit Data"]["Z"])
                ])
                pseudorange_values.append(float(data["C1 Observable"]))
            except (ValueError, TypeError):
                print(f"Bỏ qua vệ tinh {sattelite} do dữ liệu không hợp lệ")
    
    if len(sat_pos) >= 4:
        try:
            receiver_position, clock_bias = compute_receiver_position(sat_pos, pseudorange_values)
            receiver_positions[previous_epoch] = {
                "ECEF": receiver_position,
                "Clock Bias": clock_bias,
                "Time": epoch_data["Time of epoch"]
            }
            
            # Chuyển đổi sang tọa độ địa lý
            lat, lon, alt = ecef_to_geodetic(receiver_position[0], receiver_position[1], receiver_position[2])
            receiver_positions[previous_epoch]["Geodetic"] = {
                "Latitude": lat,
                "Longitude": lon,
                "Altitude": alt
            }
            
            print(f"Epoch {previous_epoch} - Vị trí: ECEF({receiver_position[0]:.2f}, {receiver_position[1]:.2f}, {receiver_position[2]:.2f}) m")
            print(f"              Tọa độ địa lý: ({lat:.6f}°, {lon:.6f}°, {alt:.2f} m)")
            print(f"              Sai số đồng hồ: {clock_bias:.2f} m")
        except Exception as e:
            print(f"Lỗi khi tính toán vị trí cho epoch {previous_epoch}: {str(e)}")
    else:
        print(f"Không đủ vệ tinh hợp lệ ({len(sat_pos)}/4) cho epoch {previous_epoch}")

# In tổng kết
print("\n===== TÓM TẮT VỊ TRÍ THEO EPOCH =====")
print(f"Tổng số epoch đã xử lý: {epoch_count}")
print(f"Số epoch tính được vị trí: {len(receiver_positions)}")

file.close()
