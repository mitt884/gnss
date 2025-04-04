import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Các hằng số toàn cục (thay thế bằng class nếu cần)
MUY = 3.986055e14          # m^3/s^2
WDOT_RATE = 7.2921151467e-5 # Earth rotation rate (rad/s)
PI = 3.1415926535898
C = 2.99792458e8           # Tốc độ ánh sáng (m/s)
F = -4.442807633e-10

def PVTMain():
    # Cờ điều khiển
    new_data_reading = False
    use_klobuchar_model = False
    use_ionex_map = False

    # Đọc dữ liệu
    if new_data_reading:
        # Đường dẫn file (cần điều chỉnh theo hệ thống)
        rinex_obs_path = 'MobileDataSet/FixPoint/SET081U.18o'
        rinex_eph_path = 'MobileDataSet/FixPoint/brdc0810.18n'
        ionex_path = 'MobileDataSet/FixPoint/brdc0810.18n'
        
        # Giả định các hàm đọc RINEX đã được triển khai
        ObsGPS = read_rinex_obs(rinex_obs_path)
        EphsGPS = read_rinex_eph(rinex_eph_path)
        IonexGPS = read_ionex(ionex_path)
        
        # Lưu dữ liệu
        with open('EphsGPSfixp2.pkl', 'wb') as f:
            pickle.dump(EphsGPS, f)
        # ... (tương tự cho các file khác)
    else:
        # Tải dữ liệu đã lưu
        with open('ObsGPSabmf.pkl', 'rb') as f:
            ObsGPS = pickle.load(f)
        with open('EphsGPSabmf.pkl', 'rb') as f:
            EphsGPS = pickle.load(f)
        with open('IonexGPS23032018.pkl', 'rb') as f:
            IonexGPS = pickle.load(f)

    # Khởi tạo biến
    num_epochs = ObsGPS['numEpochs']
    reference_point = np.array([4646624, 1031428, 4231581])
    
    enu = np.zeros((num_epochs, 3))
    xyz = np.zeros((num_epochs, 3))
    userpos = np.zeros(4)
    computed_pos = np.zeros((num_epochs, 4))

    for epoch in range(num_epochs):
        obs_tow = ObsGPS['epochs'][epoch]['TOC']
        sats_in_view = ObsGPS['epochs'][epoch]['PRNs']
        
        nsat = len(sats_in_view)
        correct_pseudoranges = np.zeros(nsat)
        satpos = np.zeros((nsat, 3))
        
        sat_idx = 0
        for i in range(nsat):
            sat_id = sats_in_view[i]
            pseudorange = ObsGPS['epochs'][epoch]['Obs'][sat_id]['pseudorange']
            
            if pseudorange > 27e6:  # Lỗi dữ liệu
                continue
            
            t_sv = obs_tow - pseudorange / C
            
            # Tính vị trí vệ tinh (giả định hàm này đã được triển khai)
            x, y, z, clk_bias = compute_satellite_pos_EFEC(sat_id, EphsGPS, obs_tow, t_sv)
            
            if x != -1:  # Kiểm tra dữ liệu hợp lệ
                satpos[sat_idx, :] = [x, y, z]
                correct_pseudoranges[sat_idx] = pseudorange + clk_bias * C
                
                if epoch > 0:
                    userpos = computed_pos[epoch-1, :]
                    az, el, llh = az_el(satpos[sat_idx, :], userpos[:3])
                    
                    # Hiệu chỉnh iono
                    if use_klobuchar_model:
                        alpha = EphsGPS['IonoCoeficient']['Alpha']
                        beta = EphsGPS['IonoCoeficient']['Beta']
                        iono_delay = klobuchar_correction(az, el, llh[0], llh[1], alpha, beta, obs_tow)
                        correct_pseudoranges[sat_idx] -= iono_delay * C
                    
                    if use_ionex_map:
                        iono_delay = ionex_map_correction(az, el, llh[0], llh[1], IonexGPS, obs_tow)
                        if not np.isnan(iono_delay):
                            correct_pseudoranges[sat_idx] -= iono_delay
                
                sat_idx += 1
        
        if sat_idx >= 4:  # Cần ít nhất 4 vệ tinh
            if sat_idx < nsat:
                satpos = satpos[:sat_idx, :]
                correct_pseudoranges = correct_pseudoranges[:sat_idx]
            
            # Tính vị trí (giả định hàm least squares đã triển khai)
            if epoch == 0:
                rx, ry, rz, rbu = compute_PVT(correct_pseudoranges, satpos, userpos)
            else:
                rx, ry, rz, rbu = compute_PVT(correct_pseudoranges, satpos, computed_pos[epoch-1, :])
            
            xyz[epoch, :] = [rx, ry, rz]
            computed_pos[epoch, :] = [rx, ry, rz, rbu]
        
        print(f"Processed epoch {epoch+1}/{num_epochs}")

    # Tính ENU
    reference_point = np.mean(xyz, axis=0)
    for i in range(len(xyz)):
        enu[i, :] = xyz2enu(xyz[i, :], reference_point[:3])
    
    # Vẽ đồ thị
    plt.figure()
    plt.title("Ground Track")
    plt.plot(enu[:, 0], enu[:, 1], 'x')
    plt.figure()
    plt.title("E-W Position Error (m)")
    plt.plot(enu[:, 0])
    plt.show()
    
    precision = np.sqrt(np.mean((xyz - reference_point)**2, axis=0))
    print(f"Precision (RMS): {precision}")

# Các hàm phụ trợ cần triển khai
def compute_satellite_pos_EFEC(sat_id, ephs, tow, t_sv):
    # Triển khai giải thuật tính vị trí vệ tinh
    return x, y, z, clk_bias  # Giá trị mẫu

def compute_PVT(pseudoranges, satpos, initial_pos):
    # Triển khai least squares hoặc Kalman filter
    return rx, ry, rz, rbu  # Giá trị mẫu

def xyz2enu(pos, ref):
    # Chuyển đổi tọa độ
    return enu  # Giá trị mẫu

if __name__ == "__main__":
    PVTMain()