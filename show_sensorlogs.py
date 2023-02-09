import os
import pathlib

import matplotlib.collections
import numpy as np
import mat73
import datetime
import matplotlib.pyplot as plt
import scipy.optimize

from util import getType, plot_and_save, smothing
import gps_util

#<constants>
epoch = datetime.datetime.utcfromtimestamp(0)

time_format_str = "%Y_%m_%d-%H_%M_%S"
x_ending = "_x"
relevant_keys = ["BMS_SOC_UsbFlRec", "Converter_L_N_actual_UsbFlRec", "Converter_R_N_actual_UsbFlRec",
                 "Converter_L_RPM_Actual_Filtered_UsbFlRec", "Converter_R_RPM_Actual_Filtered_UsbFlRec",
                 "Converter_L_Torque_Out_UsbFlRec", "Converter_R_Torque_Out_UsbFlRec", "ECU_ACC_X_UsbFlRec",
                 "ECU_ACC_Y_UsbFlRec", "ECU_ACC_Z_UsbFlRec", "GNSS_heading_UsbFlRec", "GNSS_latitude_UsbFlRec",
                 "GNSS_longitude_UsbFlRec", "GNSS_speed_over_ground_UsbFlRec", "SWS_angle_UsbFlRec"]
cam_keys = ["cam_left", "cam_right", "cam_drone"]
# irrelevant_keys = ['APPS_L_ADC_UsbFlRec', 'APPS_L_ADC_UsbFlRec_x', 'APPS_R_ADC_UsbFlRec', 'APPS_R_ADC_UsbFlRec_x', 'APPS_rx_delta_time_UsbFlRec', 'APPS_rx_delta_time_UsbFlRec_x', 'APPS_rx_delta_time_error_UsbFlRec', 'APPS_rx_delta_time_error_UsbFlRec_x', 'BMS_error_IVTcommFail_UsbFlRec', 'BMS_error_IVTcommFail_UsbFlRec_x', 'BMS_error_VCUcommFail_UsbFlRec', 'BMS_error_VCUcommFail_UsbFlRec_x', 'BMS_error_bootFail_UsbFlRec', 'BMS_error_bootFail_UsbFlRec_x', 'BMS_error_cellOvervoltage_UsbFlRec', 'BMS_error_cellOvervoltage_UsbFlRec_x', 'BMS_error_cellTempHigh_UsbFlRec', 'BMS_error_cellTempHigh_UsbFlRec_x', 'BMS_error_cellTempLow_UsbFlRec', 'BMS_error_cellTempLow_UsbFlRec_x', 'BMS_error_cellUndervoltage_UsbFlRec', 'BMS_error_cellUndervoltage_UsbFlRec_x', 'BMS_error_communication_UsbFlRec', 'BMS_error_communication_UsbFlRec_x', 'BMS_error_contactor_UsbFlRec', 'BMS_error_contactor_UsbFlRec_x', 'BMS_error_currentCharge_UsbFlRec', 'BMS_error_currentCharge_UsbFlRec_x', 'BMS_error_currentDischarge_UsbFlRec', 'BMS_error_currentDischarge_UsbFlRec_x', 'BMS_error_eepromFail_UsbFlRec', 'BMS_error_eepromFail_UsbFlRec_x', 'BMS_error_interlockFail_UsbFlRec', 'BMS_error_interlockFail_UsbFlRec_x', 'BMS_error_isolation_UsbFlRec', 'BMS_error_isolation_UsbFlRec_x', 'BMS_error_lmmSupplyVoltage_UsbFlRec', 'BMS_error_lmmSupplyVoltage_UsbFlRec_x', 'BMS_error_openWire_UsbFlRec', 'BMS_error_openWire_UsbFlRec_x', 'BMS_error_powerLimExceeded_UsbFlRec', 'BMS_error_powerLimExceeded_UsbFlRec_x', 'BMS_error_precharge_UsbFlRec', 'BMS_error_precharge_UsbFlRec_x', 'BMS_error_relayWelded_UsbFlRec', 'BMS_error_relayWelded_UsbFlRec_x', 'BMS_error_socLow_UsbFlRec', 'BMS_error_socLow_UsbFlRec_x', 'BMS_error_systemFail_UsbFlRec', 'BMS_error_systemFail_UsbFlRec_x', 'BMS_error_voltDiff_UsbFlRec', 'BMS_error_voltDiff_UsbFlRec_x', 'BMS_max_cellAvg_UsbFlRec', 'BMS_max_cellAvg_UsbFlRec_x', 'BMS_max_cellTemp_UsbFlRec', 'BMS_max_cellTemp_UsbFlRec_x', 'BMS_max_cellVoltage_UsbFlRec', 'BMS_max_cellVoltage_UsbFlRec_x', 'BMS_min_cellTemp_UsbFlRec', 'BMS_min_cellTemp_UsbFlRec_x', 'BMS_min_cellVoltage_UsbFlRec', 'BMS_min_cellVoltage_UsbFlRec_x', 'BMS_total_battery_voltage_UsbFlRec', 'BMS_total_battery_voltage_UsbFlRec_x', 'BMS_total_discharge_current_UsbFlRec', 'BMS_total_discharge_current_UsbFlRec_x', 'BPS_F_bar_UsbFlRec', 'BPS_F_bar_UsbFlRec_x', 'BPS_F_rx_delta_time_UsbFlRec', 'BPS_F_rx_delta_time_UsbFlRec_x', 'BPS_F_rx_delta_time_error_UsbFlRec', 'BPS_F_rx_delta_time_error_UsbFlRec_x', 'BPS_R_bar_UsbFlRec', 'BPS_R_bar_UsbFlRec_x', 'BPS_R_rx_delta_time_UsbFlRec', 'BPS_R_rx_delta_time_UsbFlRec_x', 'BPS_R_rx_delta_time_error_UsbFlRec', 'BPS_R_rx_delta_time_error_UsbFlRec_x', 'Cockpit_Taster_UsbFlRec', 'Cockpit_Taster_UsbFlRec_x', 'Converter_L_Air_Temperature_UsbFlRec', 'Converter_L_Air_Temperature_UsbFlRec_x', 'Converter_L_BTB_UsbFlRec', 'Converter_L_BTB_UsbFlRec_x', 'Converter_L_Current1_Actual_UsbFlRec', 'Converter_L_Current1_Actual_UsbFlRec_x', 'Converter_L_Current2_Actual_UsbFlRec', 'Converter_L_Current2_Actual_UsbFlRec_x', 'Converter_L_Current3_Actual_UsbFlRec', 'Converter_L_Current3_Actual_UsbFlRec_x', 'Converter_L_DC_Voltage_UsbFlRec', 'Converter_L_DC_Voltage_UsbFlRec_x', 'Converter_L_Error_UsbFlRec', 'Converter_L_Error_UsbFlRec_x', 'Converter_L_I_actual_UsbFlRec', 'Converter_L_I_actual_UsbFlRec_x', 'Converter_L_I_cmd_Ramp_UsbFlRec', 'Converter_L_I_cmd_Ramp_UsbFlRec_x', 'Converter_L_I_cmd_UsbFlRec', 'Converter_L_I_cmd_UsbFlRec_x', 'Converter_L_Motor_Temperature_UsbFlRec', 'Converter_L_Motor_Temperature_UsbFlRec_x', 'Converter_L_N_cmd_Ramp_UsbFlRec', 'Converter_L_N_cmd_Ramp_UsbFlRec_x', 'Converter_L_N_cmd_UsbFlRec', 'Converter_L_N_cmd_UsbFlRec_x', 'Converter_L_PowerState_Temperature_UsbFlRec', 'Converter_L_PowerState_Temperature_UsbFlRec_x', 'Converter_L_Power_UsbFlRec', 'Converter_L_Power_UsbFlRec_x', 'Converter_L_Run_UsbFlRec', 'Converter_L_Run_UsbFlRec_x', 'Converter_L_V_out_UsbFlRec', 'Converter_L_V_out_UsbFlRec_x', 'Converter_L_Vdc_Bus_UsbFlRec', 'Converter_L_Vdc_Bus_UsbFlRec_x', 'Converter_R_Air_Temperature_UsbFlRec', 'Converter_R_Air_Temperature_UsbFlRec_x', 'Converter_R_BTB_UsbFlRec', 'Converter_R_BTB_UsbFlRec_x', 'Converter_R_Current1_Actual_UsbFlRec', 'Converter_R_Current1_Actual_UsbFlRec_x', 'Converter_R_Current2_Actual_UsbFlRec', 'Converter_R_Current2_Actual_UsbFlRec_x', 'Converter_R_Current3_Actual_UsbFlRec', 'Converter_R_Current3_Actual_UsbFlRec_x', 'Converter_R_DC_Voltage_UsbFlRec', 'Converter_R_DC_Voltage_UsbFlRec_x', 'Converter_R_Error_UsbFlRec', 'Converter_R_Error_UsbFlRec_x', 'Converter_R_I_actual_UsbFlRec', 'Converter_R_I_actual_UsbFlRec_x', 'Converter_R_I_cmd_Ramp_UsbFlRec', 'Converter_R_I_cmd_Ramp_UsbFlRec_x', 'Converter_R_I_cmd_UsbFlRec', 'Converter_R_I_cmd_UsbFlRec_x', 'Converter_R_Motor_Temperature_UsbFlRec', 'Converter_R_Motor_Temperature_UsbFlRec_x', 'Converter_R_N_cmd_Ramp_UsbFlRec', 'Converter_R_N_cmd_Ramp_UsbFlRec_x', 'Converter_R_N_cmd_UsbFlRec', 'Converter_R_N_cmd_UsbFlRec_x', 'Converter_R_PowerState_Temperature_UsbFlRec', 'Converter_R_PowerState_Temperature_UsbFlRec_x', 'Converter_R_Power_UsbFlRec', 'Converter_R_Power_UsbFlRec_x', 'Converter_R_Run_UsbFlRec', 'Converter_R_Run_UsbFlRec_x', 'Converter_R_V_out_UsbFlRec', 'Converter_R_V_out_UsbFlRec_x', 'Converter_R_Vdc_Bus_UsbFlRec', 'Converter_R_Vdc_Bus_UsbFlRec_x', 'ECU_APPS_Error_UsbFlRec', 'ECU_APPS_Error_UsbFlRec_x', 'ECU_APPS_and_BPS_simultaniously_UsbFlRec', 'ECU_APPS_and_BPS_simultaniously_UsbFlRec_x', 'ECU_BPS_Error_UsbFlRec', 'ECU_BPS_Error_UsbFlRec_x', 'ECU_InverterCAN_ON_UsbFlRec', 'ECU_InverterCAN_ON_UsbFlRec_x', 'ECU_Inverter_Enabling_UsbFlRec', 'ECU_Inverter_Enabling_UsbFlRec_x', 'ECU_air_pressure_UsbFlRec', 'ECU_air_pressure_UsbFlRec_x', 'ECU_brake_UsbFlRec', 'ECU_brake_UsbFlRec_x', 'ECU_nanoseconds_global_UsbFlRec', 'ECU_nanoseconds_global_UsbFlRec_x', 'ECU_seconds_global_UsbFlRec', 'ECU_seconds_global_UsbFlRec_x', 'ECU_speed_setpoint_UsbFlRec', 'ECU_speed_setpoint_UsbFlRec_x', 'ECU_state_UsbFlRec', 'ECU_state_UsbFlRec_x', 'ECU_torque_setpoint_UsbFlRec', 'ECU_torque_setpoint_UsbFlRec_x', 'GNSS_attitude_UsbFlRec', 'GNSS_attitude_UsbFlRec_x', 'GNSS_mode_UsbFlRec', 'GNSS_mode_UsbFlRec_x', 'GNSS_n_satelites_UsbFlRec', 'GNSS_n_satelites_UsbFlRec_x', 'GNSS_odometer_UsbFlRec', 'GNSS_odometer_UsbFlRec_x', 'GNSS_position_data_valid_UsbFlRec', 'GNSS_position_data_valid_UsbFlRec_x', 'SAS_FL_ADC_UsbFlRec', 'SAS_FL_ADC_UsbFlRec_x', 'SAS_FR_ADC_UsbFlRec', 'SAS_FR_ADC_UsbFlRec_x', 'SAS_RL_ADC_UsbFlRec', 'SAS_RL_ADC_UsbFlRec_x', 'SAS_RR_ADC_UsbFlRec', 'SAS_RR_ADC_UsbFlRec_x', 'SWS_angle_rotation_speed_UsbFlRec', 'SWS_angle_rotation_speed_UsbFlRec_x', 'TS_Schalter_UsbFlRec', 'TS_Schalter_UsbFlRec_x', 'TelemetryComment_UsbFlRec', 'TelemetryComment_UsbFlRec_x']
startTimestamp_key = "StartTimestamp"
length_key = "Length"
vis_out_path = pathlib.Path("vis_out/")
mat_files_dir = [pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/testrun_2022_12_17/sensordata/"+p) for p in ["testrun_13_30/", "testrun_14_21/", "testrun_14_41/"]]  # only dirs that contain .mat files for runs on the 2022-12-17 (yyyy-mm-dd)
csv_files_dir = pathlib.Path("merged_rundata_csv/")
# csv_files = [pathlib.Path(f) for f in os.listdir(csv_files_dir) if str(f).endswith(".csv")]
csv_files = [pathlib.Path(f) for f in os.listdir(csv_files_dir) if str(f).endswith(".csv") and str(f).split("-")[
    0] == "alldata_2022_12_17"]  # only return csv files that were recorded at the 2022-12-17 (yyyy-mm-dd)
cam_footage_dir = pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/testrun_2022_12_17/cam_footage/")
cam_sync_files = [cam_footage_dir / pathlib.Path(f) for f in
                  ["sync_info_camL0.txt", "sync_info_camR0.txt", "sync_info_camL3.txt", "sync_info_camR3.txt",
                   "sync_info_drone0.txt", "sync_info_drone3.txt"]]  # , "sync_info_drone1.txt"
poii_const_points_range = (0, 5)  # poii[0:5] = constant points
poii_carpos_points_range = (5, 9)
poii_bluecones_range = (9, 47)
poii_yellowcones_range = (47, 88)
#</constants
#<dataTypes>
# SensorDataDict should contain the keys startTimestamp_key: datetime.datetime, length_key:float, key: np.array, key+x_ending: np.array for key in relevant_keys and optionaly cam_keys and cam_keys+x_ending
SensorDataDict = dict
def assert_SensorDataDict_is_valid(sdd: SensorDataDict) -> None:
    keys = list(relevant_keys)
    if any([k in sdd.keys() for k in cam_keys]):
        keys += cam_keys
    keysp = keys + [k + x_ending for k in keys] + [startTimestamp_key, length_key]
    for k in [k for k in sdd.keys() if str(k).endswith(x_ending)]:
        if not all([sdd[k][i+1] >= sdd[k][i] for i in range(len(sdd[k])-1)]):  # time is ascending.
            print(f"time of {k} is not ascending")
            assert False
    if not set(keysp) == set([str(k) for k in sdd.keys()]):
        print("sdd.keys() = ", sdd.keys())
        print("\n keys+ = ", keysp)
        print("\nkeys that should be in sdd.keys() but arent: ", [k for k in keysp if k not in sdd.keys()])
        print("keys that should not be in sdd.keys() but are: ", [k for k in sdd.keys() if k not in keysp])
        assert False
    for key in keys:
        assert len(sdd[key]) == len(sdd[key + x_ending])
normalised_px_h = float  # in range(0, 1) height position of pixel in image / height of image
normalised_px_w = float  # in range(0, 1) width position of pixel in image / width of image
#meter_vector = (meter_north, meter_east)  # difference between two points on globe in meters, with (distance to North, distance to east)
seconds = float  # seconds since start of mesurment
plot = None
bearing_radiant = float  # in range(-pi, pi) 0 = south, -pi/2 = east, pi/2 = west, pi = north, -pi = north
cone_bounding_box = (int, normalised_px_w, normalised_px_h, normalised_px_w, normalised_px_h)  # classid (0:blue, 1:yellow), center of bounding box, size of bounding box
cone_keypoitns = [(normalised_px_w, normalised_px_h)]  # always length 7
poii_id = int  # id of cones and landmarks, consistent between cam and drone footage. see point_of_interest_numbers.jpg
som_time_seconds = float  # seconds since start of moving.
drone_frnr = int
camL_frnr = int
#</dataTypes>


#<utility methods>
def k_to_name(k) -> str:
    return str(k).replace("Converter_", "").replace("_UsbFlRec", "").replace("_actual", "").replace("_Actual", "").replace("_Filtered", "")


drone_som_frame = {0:2075, 3:2360}
camL_som_frame = {0:2610, 3:1281}
camR_som_frame = {0:2508, 3:1225}
ssd_som_seconds = {0:54, 3:118}
def drone2t(drone_frnr: drone_frnr, runid=3) -> som_time_seconds:
    return (drone_frnr - drone_som_frame[runid]) / 25
def t2drone(t: som_time_seconds, runid=3) -> drone_frnr:
    return int(t*25+drone_som_frame[runid])
def camL2t(camL_frnr: camL_frnr, runid=3) -> som_time_seconds:
    return (camL_frnr-camL_som_frame[runid])/20
def t2camL(t: som_time_seconds, runid=3) -> camL_frnr:
    return int(t*20+camL_som_frame[runid])
def ssdt2t(ssdt: seconds, runid=3) -> som_time_seconds:
    return ssdt-ssd_som_seconds[runid]
def t2ssdt(t: som_time_seconds, runid=3) -> seconds:
    return t+ssd_som_seconds[runid]
def abs_value(l):
    return np.sqrt(np.sum([t**2 for t in l]))


def remove_zeros(x, y, zero=0):
    # remove elements from x and y if the element of y is zero
    assert len(x) == len(y)
    tmp = [(a, b) for (a, b) in zip(x, y) if b != zero]
    x = np.array([a for (a, b) in tmp])
    y = np.array([b for (a, b) in tmp])
    return x, y


def to_range(x):
    while x > np.pi:
        x -= 2*np.pi
    while x < -np.pi:
        x += 2*np.pi
    return x


def get_at_time(x: [seconds], y: [float], t: seconds) -> (float, int):
    # returns the linear interpolated value of y at time t and nearest index
    # if t in x: return y[i], so that x[i]==t
    assert len(x) == len(y)
    if t < x[0] or t > x[-1]:
        print(f"warining: get_at_time: t is out of range ({x[0]}, {x[-1]}) with {t}")
        if t < x[0]:
            return y[0], 0
        if t > x[-1]:
            return y[-1], len(y)
    for i in range(0, len(x)):
        if t == x[i]:
            return y[i], i
        if t < x[i]:
            w0 = abs(t-x[i-1])
            w1 = abs(x[i]-t)
            return (w1*y[i-1]+w0*y[i])/(w0+w1), i-1 if w0 < w1 else i


def onsided_timesinc(in_time, in_value, target_time) -> ([seconds], [float]):
    # return the same as [get_at_time(value_time, value_value, t) for t in target_time] would, but is computationally faster.
    assert len(in_time) == len(in_value)
    in_index = 0
    synced_in_value = []
    for t in target_time:
        while in_index + 1 < len(in_time) and in_time[in_index + 1] < t:
            in_index += 1
        # in_time[in_index] < t < in_time[in_index+1]
        if in_index + 1 >= len(in_time):
            in_index = len(in_time) - 2
            synced_in_value.append(in_value[-1])
        elif in_index == 0:
            synced_in_value.append(in_value[0])
        else:
            # append weighted average
            w0 = abs(in_time[in_index] - t)
            w1 = abs(in_time[in_index + 1] - t)
            synced_in_value.append((w0 * in_value[in_index] + w1 * in_value[in_index + 1]) / (w0 + w1))
    synced_in_value = np.array(synced_in_value)
    assert len(target_time) == len(synced_in_value)
    return synced_in_value

def timesinc(a_x: [seconds], a_y: [float], b_x: [seconds], b_y: [float]) -> ([seconds], [float], [float]):
    # return timesteps, a_y, b_y, so that (timesteps, a_y) and timesteps, b_y) are valid time-data pairs, and that a_y and b_y are as close to the input a_y and b_y as possible
    timesteps = np.array(sorted(list(set(a_x).union(set(b_x)))))
    i0 = 0
    i1 = 1
    sy0 = []
    sy1 = []
    for t in timesteps:
        while i0 + 1 < len(a_x) and a_x[i0 + 1] < t:
            i0 += 1
        while i1+1 < len(b_x) and b_x[i1 + 1] < t:
            i1 += 1
        # x0[i0] < t < x0[i0+1]
        if i0 + 1 >= len(a_x):
            i0 = len(a_x) - 2
            sy0.append(a_y[-1])
        elif i0 == 0:
            sy0.append(a_y[0])
        else:
            # append weighted average
            w0 = abs(a_x[i0] - t)
            w1 = abs(a_x[i0 + 1] - t)
            sy0.append((w0 * a_y[i0] + w1 * a_y[i0 + 1]) / (w0 + w1))
        if i1 + 1 >= len(b_x):
            i1 = len(b_x) - 2
            sy1.append(b_y[-1])
        elif i1 == 0:
            sy1.append(b_y[0])
        else:
            # append weighted average
            w0 = abs(b_x[i1] - t)
            w1 = abs(b_x[i1 + 1] - t)
            sy1.append((w0 * b_y[i1] + w1 * b_y[i1 + 1]) / (w0 + w1))
    sy0 = np.array(sy0)
    sy1 = np.array(sy1)
    assert len(timesteps) == len(sy0) == len(sy1)
    return timesteps, sy0, sy1


def fit_poly_fun_and_print(in_x, out_true, name, exponents=[-1, 0, 1, 2]):
    # returns fun, parameters, so that fun(in_x, parameters) = out_true
    assert len(in_x) == len(out_true)
    in_x = np.array(in_x)
    out_true = np.array(out_true)
    #fo = np.array([1, 0, 0, 0])
    fo = np.array([(1 if i == 1 else 0) for i in exponents])

    def fun(x, t):
        return np.sum([t[i]*x**v for (i, v) in enumerate(exponents)])
        #return t[0]*x + t[1]*x**2 + t[2]/x + t[3]
    def loss(t):
        return np.sum((fun(in_x, t) - out_true) ** 2)
    res = scipy.optimize.minimize(loss, fo)
    if res.success:
        #return fun, res.x
        parameters = res.x
    else:
        print(f"fitting {name} failed")
        if res.x is not None:
            #return fun, res.x
            parameters = res.x
        else:
            raise Exception("could not succesfully fit data from", in_x, "to", out_true)
    est_angle = fun(in_x, parameters)
    print(f"{name}.est_from ", in_x)
    print(f"{name}.fun(est_from) = ", est_angle)
    print(f"{name}.out_true  = ", out_true)
    print(f"parameters[{name}] = {parameters}")
    print(f"diff_{name} = {np.sqrt(np.sum((est_angle-out_true)**2))}\n")
    return fun, parameters


def avg_pxprom_from_conekeypoints(keypoints: [(normalised_px_w, normalised_px_h)], bounding_box: cone_bounding_box) -> float:
    # keypoints = position of keypoints on image of cone, with (0, 0) = upper left corner, (1, 0) = lower left corner, (0, 1) = upper right corner
    # bounding_box = (class, position of center of bounding_box, position of center of bounding_box, height of bounding box/image_height, width of bounding box/image_width)
    # returns avg(pixel dist between two keypoints / meter dist between two objectpoints)
    # objectPoints = position of keypoints on real cone in cone-centerd coordinate system (0, 0) = center of bottom of cone in meter

    #cone_height = 0.325  # [m]
    #cone_diamter_top = 0.046  # (3/21)*np.sqrt(2*0.228**2)  # [m]
    #cone_diameter_bottom = 0.169  # (11/21)*np.sqrt(2*0.228**2)  # [m]
    #cone_diamter_dif = cone_diameter_bottom-cone_diamter_top
    #objectPoints = np.array([[0, cone_height], [cone_diamter_top+(1/3)*cone_diamter_dif, (2/3)*cone_height], [cone_diamter_top+(2/3)*cone_diamter_dif, (1/3)*cone_height], [cone_diamter_top+(3/3)*cone_diamter_dif, (0/3)*cone_height], [-cone_diamter_top-(1/3)*cone_diamter_dif, (2/3)*cone_height], [-cone_diamter_top-(2/3)*cone_diamter_dif, (1/3)*cone_height], [-cone_diamter_top-(3/3)*cone_diamter_dif, (0/3)*cone_height]])  # [[left, height, deepth]]
    #objectPoints = np.array([(0,0.325), (0.087,0.21666667), (0.128,0.10833333), (0.169,0), (-0.087,0.21666667), (-0.128,0.10833333), (-0.169,0)])
    obj_Distm = {(0, 1): 0.09931057723962547, (0, 2): 0.18581319480106973, (0, 3): 0.2893731266180254, (1, 2): 0.08783333333333333, (1, 3): 0.18816666666666668, (2, 3): 0.1, (1, 5): 0.126375, (1, 6): 0.20548663807186746, (2, 6): 0.156, (1, 4): 0.058826410187308546, (2, 5): 0.08119296009798978, (3, 6): 0.1159014116265954, (0, 4): 0.09931057723962547, (0, 5): 0.18581319480106973, (0, 6): 0.2893731266180254, (2, 4): 0.126375, (3, 4): 0.20548663807186746, (3, 5): 0.156, (4, 5): 0.08783333333333333, (4, 6): 0.18816666666666668, (5, 6): 0.1}

    assert len(keypoints) == 7
    cls, posw, posh, sizew, sizeh = bounding_box
    imgsize_h, imgsize_w = (1200, 1920)
    #keypoints_pxpos_in_camImage_from_relativepxpos_in_coneimg = np.array([((posw-0.5*sizew+w*sizew)*imgsize_w, (posh-0.5*sizeh+h*sizeh)*imgsize_h) for (w, h) in keypoints])
    keypoints = np.array([(w*sizew*imgsize_w, h*sizeh*imgsize_h) for (w, h) in keypoints])  # transform keypoints from relative position in coneimage to relative position in camera image (as only differences between keypoints are observed, the offset can be ignored)
    #avg = 0
    meadian = []
    indexe = [(i, j) for i in range(6) for j in range(i+1, 7)]
    for (i, j) in indexe:
        # avg += meter dist between points on real cone / pixel dist between points on cone image
        #avg += obj_Distm[(i, j)]/abs_value(keypoints[i]-keypoints[j])
        meadian.append(obj_Distm[(i, j)]/abs_value(keypoints[i]-keypoints[j]))
    #r = avg/len(indexe)
    r = np.median(meadian)
    # distance to object [m] = real object size(m) * focal length (mm) / object height in frame (mm)
    #  with object height in frame (mm) = sensor height (mm) * object height (px) / sensor height (px)
    # distance to object [m] = real object size(m)/ object height (px) * constant
    #  with constant = focal length (mm) * sensor height (px) / sensor height (mm)
    # this returns average of
    #  real object size(m) / (object size (px) / sensor height (px))
    return r
#</utility methods>


def plot_colorgradientline(name: str, lat_pos: [gps_util.lattitude], long_pos: [gps_util.longitude], time: [seconds] = None) -> plot:
    # https://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line-in-matplotlib
    # https://matplotlib.org/2.0.2/examples/axes_grid/demo_parasite_axes2.html
    """
    assumes x_pos, y_pos are positions of car during run and plots it and cones.
    """
    assert len(lat_pos) == len(long_pos)
    fig, (ax0) = plt.subplots(1, 1)
    cmap = plt.get_cmap('jet')

    #plot path
    points = np.array([long_pos, lat_pos]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = matplotlib.collections.LineCollection(segments, array=np.linspace(0, 1, len(lat_pos)), cmap=cmap, norm=plt.Normalize(0.0, 1.0), linewidth=2, alpha=1)
    ax0.add_collection(lc)

    # add time to points on path
    if time is not None:
        assert len(lat_pos) == len(time)
        t = time[0]
        ax0.text(long_pos[0], lat_pos[0], str(int(time[0])) + "s", c=cmap(0))
        for i in range(len(lat_pos)):
            dt = 10
            if time[i] > t + dt:  # plot only one number per dt seconds.
                ax0.text(long_pos[i], lat_pos[i], str(int(time[i])) + "s", c=cmap(i / len(lat_pos)))
                t += dt
        ax0.text(long_pos[-1], lat_pos[-1], str(int(time[-1])) + "s", c=cmap(1))
    ax0.set_title(f"Path of {name}")
    ax0.scatter(long_pos, lat_pos, s=10, c=cmap(np.linspace(0, 1, len(lat_pos))), alpha=0.5)

    # plot cones
    poi_true_gpspos, carpos = true_pos_from_droneimg_pxpos()
    for ycp in poi_true_gpspos[poii_yellowcones_range[0]:poii_yellowcones_range[1]]:
        ax0.scatter(ycp[1], ycp[0], c="yellow")
    for bcp in poi_true_gpspos[poii_bluecones_range[0]:poii_bluecones_range[1]]:
        ax0.scatter(bcp[1], bcp[0], c="blue")

    # plot carpos (car positions from drone view)
    for k in carpos.keys():
        for i in range(len(carpos[k])):
            ax0.scatter(carpos[k][i][1], carpos[k][i][0], s=5, color=cmap((k/25+27.92)/max(time)), alpha=1)

    # add labels and meter scaling on both axies  # TODO meter labels on axis may be inaccurate, use gps_utils
    #lat_pos, long_pos = gps2meter(lat_pos, long_pos, lat_pos[0], long_pos[0])
    tmp = 111320 * np.cos(np.average(lat_pos) * np.pi / 180)
    long_avg = np.average(long_pos)
    axright = ax0.secondary_yaxis('right', functions=(lambda y: (y - long_avg) * tmp, lambda y: long_avg + y / tmp))
    axright.set_xlabel("m North")
    lat_avg = np.average(lat_pos)
    axtop = ax0.secondary_xaxis("top", functions=(lambda x: (x - np.average(lat_pos)) * 111320, lambda x: lat_avg + x / 111320))
    axtop.set_ylabel('m East')


    ax0.set_xlabel("Longitude")
    ax0.set_ylabel("Lattitude")
    ax0.grid()
    fig.savefig(vis_out_path / f"{name}__path.png")
    plt.show()


def timestr2datetime(timestr: str) -> datetime.datetime:
    """
    :param timestr:
    mm/dd/yyyy hh:mm:ss LL
    with LL either AM or PM
    :return:
    datetime object containing the same datetime as the string.
    """
    # I dont know if the defualt parsing with format string supports PM/AM, and dont want to look it up.
    data, time, halfday = timestr.split(" ")
    month, day, year = [int(t) for t in data.split("/")]
    hour, minute, second = [int(t) for t in time.split(":")]
    if halfday == "PM":
        hour += 12
    elif halfday == "AM":
        hour += 0
    else:
        print("Unexpected timestring: ", timestr)
        print("halfday should be AM or PM, but is ", halfday)
        exit(0)
    return datetime.datetime(year, month, day, hour=hour, minute=minute, second=second, microsecond=0, tzinfo=None, fold=0)


def read_mat(fname: os.path) -> SensorDataDict:
    print("load file ", fname)
    my_dict = {}
    description_key = "Description"
    data_dict = mat73.loadmat(fname)
    assert len(data_dict.keys()) == 1
    data_dict = data_dict[list(data_dict.keys())[0]]  # top-level has only one entry
    # Datetime +Length*minute = Datetaime der nächsten File sollte gelten
    # print("data_dict[Y]", [entry["Raster"] for entry in data_dict["Y"]])
    # print("data_dict[X]", [entry for entry in data_dict["X"]])

    my_dict[startTimestamp_key] = timestr2datetime(data_dict[description_key]["General"]["DateTime"])
    my_dict[length_key] = float(data_dict[description_key]["Measurement"][length_key])
    for (entry_time, entry_value) in zip(data_dict["X"], data_dict["Y"]):
        # x_entry and entry are np-arrays of seconds_since_starttime, data_of_x_entry["Raster"]_sensor
        assert entry_time["Raster"] == entry_value["Raster"]
        name = entry_value["Raster"]
        if name in relevant_keys:
            name_x = name + x_ending
            tmp = [(time, value) for (time, value) in zip(entry_time["Data"], entry_value["Data"]) if not np.isnan(time) and not np.isnan(value)]
            data_time = np.array([time for (time, value) in tmp])
            data_value = np.array([value for (time, value) in tmp])
            assert all([data_time[i+1] >= data_time[i] for i in range(len(data_time) - 1)])  # x should be ascending
            assert len(data_value) == len(data_time)
            my_dict[name] = data_value
            my_dict[name_x] = data_time
    assert_SensorDataDict_is_valid(my_dict)
    return my_dict


def merge_matsdata(dict_a: SensorDataDict, dict_b: SensorDataDict) -> SensorDataDict:
    assert_SensorDataDict_is_valid(dict_a)
    assert_SensorDataDict_is_valid(dict_b)
    # assert dict_a[startTimestamp_key]+dict_a[length_key] == dict_b[startTimestamp_key]
    merged_dict = {startTimestamp_key: dict_a[startTimestamp_key], length_key: dict_a[length_key] + dict_b[length_key]}
    data_keys = [k for k in dict_a.keys() if
                 not str(k).endswith(x_ending) and str(k) != startTimestamp_key and str(k) != length_key]
    for name in data_keys:
        name_x = name + x_ending
        new_data = np.concatenate(
            (np.reshape(dict_a[name], len(dict_a[name])), np.reshape(dict_b[name], len(dict_b[name]))))
        new_datax = np.concatenate((np.reshape(dict_a[name_x], len(dict_a[name_x])), np.array([t + dict_a[length_key] for t in dict_b[name_x]])))
        assert len(new_data) == len(new_datax)
        merged_dict[name] = new_data
        merged_dict[name_x] = new_datax
    assert_SensorDataDict_is_valid(merged_dict)
    return merged_dict


def read_mats(path: os.path) -> [SensorDataDict]:
    if type(path) == list:
        files = []
        for p in path:
            files += [p / f for f in os.listdir(p) if str(f).endswith(".mat")]
    else:
        files = [path / f for f in os.listdir(path) if str(f).endswith(".mat")]
    files.sort(key=lambda x: str(x))
    mes_dicts = []
    # read all files
    for fname in files:
        mes_dicts.append(read_mat(fname))
    print(f"read {len(mes_dicts)} .mat files")
    # sort by time
    mes_dicts.sort(key=lambda x: x[startTimestamp_key])
    print("keys = ", mes_dicts[0].keys())
    # mearge consequtive dicts (starttime[i]+length[i] == starttime[i+1] -> merge i and i+1
    merged_dicts = []
    tmp = None
    time_tolerace = datetime.timedelta(seconds=60)  # most runs where half an hour apart, chance of falsy appending wrong should be minimal
    for mes_dict in mes_dicts:
        print("mes_dict: StartTimestamp = ", mes_dict[startTimestamp_key], "; length = ", mes_dict[length_key])
        if tmp is None:
            tmp = mes_dict
        else:
            print("tmp[StartTimestampe] = ", tmp[startTimestamp_key], "; length = ", tmp[length_key])
            print("diff = ", abs(tmp[startTimestamp_key] + datetime.timedelta(seconds=int(tmp[length_key])) - mes_dict[
                startTimestamp_key]))
            if abs(tmp[startTimestamp_key] + datetime.timedelta(seconds=int(tmp[length_key])) - mes_dict[
                startTimestamp_key]) < time_tolerace:
                tmp = merge_matsdata(tmp, mes_dict)
            else:
                merged_dicts.append(tmp)
                print("added dict from ", tmp[startTimestamp_key], " with length ", tmp[length_key])
                tmp = None
    if tmp is not None:
        merged_dicts.append(tmp)
    # add cam footag (timestamps and framenumbers) to data
    cam_footages = []
    name = "name"
    for cam_name in cam_sync_files:
        st, fps, n_frames, frames_dir = get_laptoptimes_for_camFrames(cam_name)
        cam_footages.append({startTimestamp_key: st, "fps": fps, "n_frames": n_frames, name: frames_dir})
    print("cam_footages: ", [cf["name"] for cf in cam_footages])
    for sensor_data in merged_dicts:
        for cam_footage in cam_footages:
            cf_sts = (cam_footage[startTimestamp_key] - epoch).total_seconds()
            cf_ls = cam_footage["n_frames"] / cam_footage["fps"]
            sd_sts = (sensor_data[startTimestamp_key] - epoch).total_seconds()
            sd_ls = sensor_data[length_key]
            if sd_sts < cf_sts < sd_sts + sd_ls or sd_sts < cf_sts + cf_ls < sd_sts + sd_ls or (
                    cf_sts < sd_sts and cf_sts + cf_ls > sd_sts + sd_ls) or (
                    sd_sts < cf_sts and sd_sts + sd_ls > cf_sts + cf_ls):
                print(
                    f"add cam_footage {cam_footage[startTimestamp_key]} with length {int(cf_ls // 60)}:{cf_ls % 60} to sensor_data from {sensor_data[startTimestamp_key]} with length {int(sd_ls // 60)}:{sd_ls % 60}")
                # start or end of ov cam_footage are inside of sensor_data or sensor_data is inside of cam_footage or cam_footage is inside sensor_data
                # there exist an overlap of the time where cam_footage was recorded and sensor_data was recorded
                if str(cam_footage[name]).split("\\")[-1].startswith("left_cam"):
                    cam_key = "cam_left"
                elif str(cam_footage[name]).split("\\")[-1].startswith("right_cam"):
                    cam_key = "cam_right"
                elif str(cam_footage[name]).split("\\")[-1].startswith("DJI_"):
                    cam_key = "cam_drone"
                else:
                    raise Exception(f"invalid start of onboard camara frames dir: {cam_footage[name]}")
                startframe = 0
                if cam_footage[startTimestamp_key] < sensor_data[startTimestamp_key]:
                    dts = (sensor_data[startTimestamp_key] - cam_footage[startTimestamp_key]).total_seconds()
                    startframe = 1 + int(cam_footage["fps"] * dts)
                sensor_data[cam_key + x_ending] = [
                    cam_footage[startTimestamp_key] + datetime.timedelta(seconds=frame_nr / cam_footage["fps"]) for
                    frame_nr in range(startframe, cam_footage["n_frames"])]
                sensor_data[cam_key + x_ending] = [(t - sensor_data[startTimestamp_key]).total_seconds() for t in sensor_data[cam_key+x_ending]]  # transform from datetime to float seconds since sensor_date[startTimestamp_key
                sensor_data[cam_key] = [str(cam_footage[name] / f"frame_{i}.jpg") for i in range(startframe, cam_footage["n_frames"])]
        for key in cam_keys:
            if not key in sensor_data.keys():
                sensor_data[key] = np.array([])
                sensor_data[key + x_ending] = np.array([])
    for sdd in merged_dicts:
        assert_SensorDataDict_is_valid(sdd)
    return merged_dicts


read_csv_files={}
def read_csv(csv_file: os.path) -> SensorDataDict:
    if str(csv_file) in read_csv_files.keys():
        return read_csv_files[str(csv_file)]
    # format:
    #  first line: only first cell, datetime of start and end of mesurment
    #  second line: names/keys
    my_dict = {}
    with open(csv_file) as f:
        lines = f.readlines()
        # f.write("mesurment from "+str(data[startTimestamp_key])+" for "+str(data[length_key])+" seconds\n")
        print("read " + lines[0])
        tmp = lines[0].split(" ")
        date = tmp[2].split("-")
        time = tmp[3].split(":")
        startTimestamp = datetime.datetime(year=int(date[0]), month=int(date[1]), day=int(date[2]), hour=int(time[0]),
                                           minute=int(time[1]), second=int(time[2]))
        length = float(tmp[5])
        labels = lines[1].replace("\n", "").split(",")
        print("read labels = ", len(labels), " : ", labels)
        for label in labels:
            if not label in ["time"]:
                my_dict[label] = []
                my_dict[label + x_ending] = []
        for line in lines[2:]:
            linesplit = line.split(",")
            time = float(linesplit[0])
            for i, v in enumerate(linesplit):  # time,cam_left,cam_right,cam_Drone,BMS_SOC_UsbFlRec,
                if not v.replace(" ", "").replace("\n", "") == "":
                    if labels[i] == "time":
                        pass
                    elif labels[i] in cam_keys:
                        my_dict[labels[i]].append(str(v))
                        my_dict[labels[i] + x_ending].append(time)
                    else:
                        my_dict[labels[i]].append(float(v))
                        my_dict[labels[i] + x_ending].append(time)
    for k in my_dict.keys():
        my_dict[k] = np.array(my_dict[k])
    my_dict[startTimestamp_key] = startTimestamp
    my_dict[length_key] = length
    assert_SensorDataDict_is_valid(my_dict)
    read_csv_files[str(csv_file)] = my_dict
    return my_dict


def write_csv(filename: os.path, data: SensorDataDict):
    assert_SensorDataDict_is_valid(data)
    # print("getType(data)", getType(data))
    print("\nwrite file ", filename)
    # save as .csv
    times = set()
    write_keys = list(relevant_keys)
    print("relevant_keys = ", relevant_keys)
    print("cam_keys = ", cam_keys)
    if any([len(data[k]) > 0 for k in cam_keys]):
        write_keys += cam_keys
    print("write_keys = ", write_keys)
    for k in [k for k in write_keys]:
        times |= set(data[k + x_ending])
    times = list(times)
    times.sort()
    print("len(times) = ", len(times), "; max[len(k)] = ", max([len(data[k + x_ending]) for k in write_keys]))
    print("lengths = ", [str(k) + ":" + str(len(data[k])) for k in write_keys])
    with open(filename, "w") as f:
        f.write("mesurment from " + str(data[startTimestamp_key]) + " for " + str(data[length_key]) + " seconds\n")
        f.write("time," + ",".join(write_keys) + "\n")
        indexe = {}
        for k in write_keys:
            indexe[k] = 0
        for t in times:
            tmp_d = [str(t)]
            for k in write_keys:
                kx = k + x_ending
                if indexe[k] < len(data[kx]) and t == data[kx][indexe[k]]:
                    tmp_d.append(str(data[k][indexe[k]]))
                    indexe[k] += 1
                else:
                    tmp_d.append("")
            f.write(",".join(tmp_d) + "\n")
    return


def visualise_data(data: SensorDataDict) -> plot:
    assert_SensorDataDict_is_valid(data)
    name = data[startTimestamp_key].strftime(time_format_str)
    print("\nvisualise data ", name)
    keys_where_zero_is_nan = ["BMS_SOC_UsbFlRec", "GNSS_latitude_UsbFlRec", "GNSS_longitude_UsbFlRec"]
    print("empty keys =", [key for key in relevant_keys if
                               not str(key) == startTimestamp_key and not str(key) == length_key and len(
                                   data[key]) == 0])
    length = min([len(data[key]) for key in relevant_keys])

    all_data_same_length = all([len(data[key]) == length for key in relevant_keys])
    print("all data has same length =", all_data_same_length)
    if not all_data_same_length:
        print("length = ", [str(key) + ": " + str(len(data[key])) for key in relevant_keys])
    error_keys = [k for k in data.keys() if ("error" in str(k).split("_")) and not k.endswith(x_ending)]
    for k in error_keys:
        if any([v != 0 for v in data[k]]):
            ranges = []
            tmp = (0, 0)
            b = False
            for i, e in enumerate([v != 0 for v in data[k]]):
                if e and not b:
                    tmp = (i, 0)
                    b = True
                if not e and b:
                    tmp = (tmp[0], i)
                    ranges.append(tmp)
                    tmp = (0, 0)
                    b = False
            if b:
                ranges.append((tmp[0], len(data[k])))
            print("ERROR: ", k, " is true in ranges", ranges)
    # BMS_SOC_UsbFlRec StateOfCharge of HV battary
    # Converter_L_N_actual_UsbFlRec is rotations/seconds of Left back wheel
    # 16bit_int.MAX_VALUE = 32767
    U_I_converter_max = 42.426
    motorkonstante = 0.83
    # true_Torque = Converter_L_Torque_Out_UsbFlRec*U_I_converter_max/(16bit_int.MAX_VALUE*np.sqrt(2))*motorkonstante
    #data["Converter_L_Torque_Out_UsbFlRec"] = np.array(data["Converter_L_Torque_Out_UsbFlRec"])*U_I_converter_max*motorkonstante/(32767*np.sqrt(2))
    #data["Converter_L_Torque_Out_UsbFlRec"] = np.array(data["Converter_L_Torque_Out_UsbFlRec"]) * U_I_converter_max * motorkonstante / (32767 * np.sqrt(2))
    for i, k in enumerate(relevant_keys):
        print("")
        if k in cam_keys:
            print("unreachable code was reached")
            continue
        if len(data[k]) == 0:
            print(f"{k} is empty")
        else:
            min_elem = min(data[k])
            max_elem = max(data[k])
            if min_elem == max_elem:
                print(f"{k} is constantly {min_elem} for {len(data[k])} elements")
            else:
                tmp = [t for t in data[k] if t != 0]
                print(f"k = {k}, type(data[k]) = {getType(data[k])}")
                print(
                    f"{k}: ({len(data[k])} elements) on average = {sum(data[k]) / len(data[k])}, nonzero_average = {sum(tmp) / len(tmp)}, (min, max) = ({min_elem}, {max_elem})")
                # plot
                x, y = data[k + x_ending], data[k]
                if k in keys_where_zero_is_nan:
                    x, y = remove_zeros(x, y)  # might be unnesary cause of outlier filtering in plot_and_save
                plot_and_save(str(k), x, [y], f"vis_out/{name}__{str(k)}.png")

    # show gnss path
    timesteps, xp, yp = get_path(data)
    plot_colorgradientline(name, xp, yp, time=timesteps)
    plt.show()


def get_laptoptimes_for_camFrames(timesyncfile: os.path) -> (datetime.datetime, float, float, str):
    """
    :param timesyncfile:
    file with format:
    first two lines: info to print
    following lines: framenumber, any of ["b2w", "w2b", datetime in format "%Y-%m-%d %H:%M:%S.%f", with milliseconds optional]
     b2w means framenumber is white and the frame before was black
     w2b means framenumber is black and the frame before was white
     datetime is the datetime shown on that frame
    :return:
    datetime of first frame, as average of all frames where datetime was given
    frames per second
    """
    print("\nread timesyncfile ", timesyncfile)
    sync_data = {}  # sync_data[frame_number] = time on laptop screen on that frame
    b2worw2b = []  # list of framenumbers where screen has changed color with respect to previus frame
    lfw = None
    frames_dir = None
    with open(timesyncfile) as f:
        lines = f.readlines()
        print("lines[0] = ", lines[0].strip())
        for t in lines[0].split(","):
            if t.startswith("frames_dir="):
                frames_dir = pathlib.Path(t.replace("frames_dir=", ""))
        if lines[1].startswith("res="):
            # lines[1] = f"res=starttime:{starttime},fps:{fps},number_of_frames:{number_of_frames}\n"
            res = lines[1].replace("res=", "").split(",")
            starttime = datetime.datetime.strptime(res[0].replace("starttime:", ""), "%Y-%m-%d %H:%M:%S.%f")
            fps = float(res[1].replace("fps:", ""))
            number_of_frames = int(res[2].replace("number_of_frames:", ""))
            return starttime, fps, number_of_frames, frames_dir
        for line in lines[2:]:
            framenr, info = line.strip().split(", ")
            framenr = int(framenr)
            if len(info) == 3:
                if lfw is None:
                    lfw = info.endswith("w")
                else:
                    assert lfw == info.startswith("w")
                    lfw = info.endswith("w")
                # change from black to white or other way round
                b2worw2b.append(framenr)
            else:
                if "." in info:
                    timestr, ms = info.split(".")
                    ms = ms + "0" * (6 - len(ms))
                    timestr = timestr + "." + ms
                else:
                    timestr = info + ".000000"
                frametime = datetime.datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S.%f")  # read with milliseconds
                sync_data[framenr] = frametime
    # print("all number of frames per seconds: ", [seconds[i+1]-seconds[i] for i in range(len(seconds)-1)])
    fps = (b2worw2b[-1] - b2worw2b[0]) / (len(b2worw2b) - 1)  # frames at last change-frames at first change / (number of changes==secends-1)
    framnr_keys = list(sync_data.keys())
    framnr_keys.sort()
    fps = (framnr_keys[-1]-framnr_keys[0]) / (sync_data[framnr_keys[-1]]-sync_data[framnr_keys[0]]).total_seconds()  # total frames of recording / total secends of recording
    spf = 1 / fps
    starttimes = [(ts - datetime.timedelta(seconds=fnr * spf)) for (fnr, ts) in sync_data.items()]
    print("starttimes = ", [t.strftime('%H:%M:%S.%f') for t in starttimes])
    tmp_time = starttimes[0]
    starttime = tmp_time + sum([date - tmp_time for date in starttimes], datetime.timedelta()) / len(starttimes)
    one_second = datetime.timedelta(seconds=0.5)
    for fnr, ts in sync_data.items():
        if abs((ts-datetime.timedelta(seconds=fnr*spf)) - starttime) > one_second:
            print(f"wrong start time: frame_nr = {fnr}, timestamp = {ts}")
            exit(0)
    number_of_frames = len([f for f in os.listdir(cam_footage_dir / frames_dir)])
    ls = number_of_frames * spf
    print("number of frames = ", number_of_frames, ", fps = ", fps, "video length = ", ls)
    print(
        f"cam {frames_dir} recors from {starttime} to {starttime + datetime.timedelta(seconds=ls)} (length={int(ls // 60)}:{ls % 60})")
    with open(timesyncfile, "w") as f:
        lines[1] = f"res=starttime:{starttime}, fps:{fps}, number_of_frames:{number_of_frames}\n"
        for line in lines:
            f.write(line)
    return starttime, fps, number_of_frames, frames_dir


def get_car_moves_starttime_from_sensors(my_dicts: SensorDataDict):
    car_started_moving_threshhold = 1
    car_stoped_moving_threshhold = 0.3
    for data in my_dicts:
        st = data[startTimestamp_key]
        print(
            f"\nsensor recording from {st} to {st + datetime.timedelta(seconds=data[length_key])} (length={data[length_key]})")

        for k, f in [("Converter_L_N_actual_UsbFlRec", 886.7828258624362),
                     ("Converter_R_N_actual_UsbFlRec", 153.78931167554344), ("GNSS_speed_over_ground_UsbFlRec", 1)]:
            k_name = k_to_name(k)
            # print(f"\n\tmoving time using {n}")
            # print(f"\tmaxm({n}_x, {n}) = ({max(data[k+x_ending])}, {max([abs(t) for t in data[k]])})")
            # print(f"\tstarted_moving_thresh = {f*car_started_moving_threshhold}, car_stoped_moving_thresh = {f*car_stoped_moving_threshhold}")
            t_start = None
            end_was_true = False
            t_end = None
            for time, value in zip(data["Converter_L_N_actual_UsbFlRec" + x_ending],
                                   data["Converter_L_N_actual_UsbFlRec"]):
                if abs(value) > f * car_started_moving_threshhold and t_start is None:
                    t_start = time
                # car is currently standing and has been moving before and has not been standing the frame before -> only true at first frame after stopped moving
                if abs(value) < f * car_stoped_moving_threshhold and t_start is not None and not end_was_true:
                    t_end = time
                end_was_true = abs(value) < f * car_stoped_moving_threshhold
            print(f"\t{k_name}: car moves from {t_start} to {t_end}")
            if t_start is not None and t_end is not None:
                print(
                    f"\taccording to {k_name} car is moving from {st + datetime.timedelta(seconds=t_start)} to {st + datetime.timedelta(seconds=t_end)} (length={t_end - t_start})")


def averegae_diff(data: SensorDataDict, k0, k1, nonzero_treshhold=(1, 1), quotents_are_same=0.0001):
    k0_name = k_to_name(k0)
    k1_name = k_to_name(k1)
    y0 = data[k0]
    y1 = data[k1]
    x0 = data[k0 + x_ending]
    x1 = data[k1 + x_ending]
    syncd = []
    xs = sorted(list(x0) + list(x1))
    min_x = max(x0[0], x1[0])
    max_x = min(x0[-1], x1[-1])
    u, o = 0, len(xs) - 1
    while xs[u] < min_x:
        u += 1
    while xs[o] > max_x:
        o -= 1
    # print("(u, o) = ", (u, o))
    xs = xs[u:o]
    i0 = 0
    i1 = 0
    for x in xs:
        while i0 + 2 < len(x0) and x0[i0 + 1] < x:
            i0 += 1
        while i1 + 2 < len(x1) and x1[i1 + 1] < x:
            i1 += 1
        # yx[ix] < y <= yx[ix+1] or ix+1 == len(yx)
        if (x0[i0 + 1] - x0[i0]) > 0 and (x1[i1 + 1] - x1[i1]) > 0:
            w1 = (x0[i0 + 1] - x)
            w2 = (x - x0[i0])
            y0t = float((y0[i0] * w1 + y0[i0 + 1] * w2) / (w1 + w2))
            w1 = (x1[i1 + 1] - x)
            w2 = (x - x1[i1])
            y1t = float((y1[i1] * w1 + y1[i1 + 1] * w2) / (w1 + w2))
            syncd.append((y0t, y1t))
    # k0/k1:
    tmp = np.array([t0 / t1 for (t0, t1) in syncd if abs(t1) > nonzero_treshhold[1]])
    k01 = None
    if len(tmp) > 0:
        k01 = np.average(tmp)
        print(f"avg({k0_name}/{k1_name}) = {k01}")
    else:
        print(f"no valid elements in {k0_name}/{k1_name}.")
    # k1/k0
    tmp = np.array([t1 / t0 for (t0, t1) in syncd if abs(t0) > nonzero_treshhold[0]])
    k10 = None
    if len(tmp) > 0:
        k10 = np.average(tmp)
        print(f"avg({k1_name}/{k0_name}) = {k10}")
    else:
        print(f"no valid elements in {k1_name}/{k0_name}.")
    # print(f"averegae_diff {k0}/{k1}: = {k01}\n{k1}/{k0} = {k10}")
    # fin
    if k01 is not None and k10 is not None:
        while abs(k01 - 1 / k10) > quotents_are_same or abs(k10 - 1 / k01) > quotents_are_same:
            k01, k10 = k01 * 0.5 + 0.5 / k10, k10 * 0.5 + 0.5 / k01
        print(f"averegae_diff: \n# {k0_name} / {k1_name} = {k01}\n# {k1_name} / {k0_name} = {k10}")


def true_pos_from_droneimg_pxpos(point_file=None) -> ([gps_util.gps_pos_radiant], [gps_util.gps_pos_radiant], [gps_util.gps_pos_radiant], {int: [gps_util.gps_pos_radiant]}):
    #poi_true_gps_positions_radiants, carpos = true_pos_from_droneimg_pxpos()
    #point_file = "C:/Users/Idefix/PycharmProjects/datasets/keypoints/droneview_annotations.csv"
    gully_u_gmpos = np.array(gps_util.degree_to_radiants((51.4641141, 6.7389449)))
    gully_o_gmpos = np.array(gps_util.degree_to_radiants((51.4644476, 6.7388095)))
    well_gmpos = np.array(gps_util.degree_to_radiants((51.4643793, 6.7390844)))
    dach_u_gmpos = np.array(gps_util.degree_to_radiants((51.4644667, 6.7389267)))
    dach_o_gmpos = np.array(gps_util.degree_to_radiants((51.46446872061805, 6.7389091136932295)))
    print("fixpoints =", [gully_u_gmpos, gully_o_gmpos, well_gmpos, dach_u_gmpos, dach_o_gmpos])
    gps_pos_optimised = []
    bearing = {}  # franenumber -> bearing
    carpos = {}  # framenumber -> [(gps_lat_frontlefttire, gps_long_frontlefttire), ... frontright, rearleft, rearright]
    if point_file is None:
        # use stored values instead of optimising from raw data
        # old gps functions: gps_pos_optimised = [gps_util.degree_to_radiants(gps) for gps in [(51.46411414940673,6.738944977588712), (51.46444738631447,6.738809863549967), (51.464376920188684,6.739083154724557), (51.46446718888097,6.738927768062857), (51.46447029093418,6.738909215184113), (51.46433948081566,6.738858198040306), (51.464340634576544,6.738867556464109), (51.464332301312695,6.738860518358256), (51.46433352256563,6.7388693023431845), (51.46425560352164,6.738978243689988), (51.46427108152239,6.738956349310485), (51.46430420716902,6.738933864796242), (51.46434835826103,6.738916791956062), (51.46439114717442,6.738898908652391), (51.464435010009225,6.738881909437699), (51.46447785743073,6.738871411732899), (51.46450255330567,6.738853593635997), (51.464523350787786,6.738819787999573), (51.464533504173595,6.738776268215176), (51.46453037892502,6.738730218636211), (51.464517197636866,6.7386905132803125), (51.46449659091481,6.738663015861728), (51.46446663771645,6.738648698012101), (51.464438661684675,6.738653402331532), (51.46441130495734,6.738677405727721), (51.46439626718731,6.73871541407409), (51.464389944382496,6.7387554556266025), (51.46438436411264,6.738785735746747), (51.46437249221701,6.738802412049573), (51.46433474605532,6.738820935341453), (51.46429112213464,6.738838149987502), (51.46425084032891,6.738857744036035), (51.46420852749894,6.73887419245192), (51.46416856291396,6.738895720484548), (51.46412900038989,6.738912834604563), (51.46410496482645,6.738923655505124), (51.464079595253736,6.738958957232722), (51.464067119195114,6.738998752044335), (51.464066961660166,6.739049464522678), (51.46407733183846,6.739086333417893), (51.46410152721491,6.739123619222857), (51.46413179871855,6.739137612689789), (51.46415989439572,6.7391333201885475), (51.46419621559581,6.739098942910411), (51.464209958613,6.739064370135348), (51.464218071281536,6.739045220236359), (51.464230863835915,6.739022540553707), (51.464235195017565,6.73894986483897), (51.46425254304922,6.738924631810385), (51.464298461923505,6.738896452978141), (51.4643428654891,6.7388789814292425), (51.46438668534625,6.738860624573461), (51.46442779156127,6.7388451840515105), (51.46447030037779,6.7388281064257844), (51.46448691323747,6.738816478338111), (51.46449890894766,6.738795981253502), (51.46450483971843,6.738769736369862), (51.4645028126984,6.738742394083106), (51.464494201192664,6.738718220083149), (51.464480374977576,6.738700955868589), (51.46446375293953,6.738694369123524), (51.464446342496,6.7386975953166015), (51.46443094576782,6.738710948324009), (51.46442003090998,6.738732241930608), (51.46441452623774,6.738758460709004), (51.4644094765204,6.738799210265234), (51.464398695574275,6.738827028726462), (51.464380796405976,6.738840788713605), (51.4643624111585,6.738848593652515), (51.4643380159601,6.738858309555241), (51.46429508893946,6.738875725158482), (51.46425547308892,6.738894756649162), (51.464215777303274,6.738910776637766), (51.46417554371516,6.7389327114558935), (51.46413319131306,6.738958079522211), (51.46411580753114,6.73896635691615), (51.46410299161643,6.738986446365815), (51.46409511800807,6.7390122939678285), (51.46409565819885,6.739041007544318), (51.46410386007197,6.739066872605536), (51.4641179539995,6.739085351021696), (51.46413588370013,6.739092211889237), (51.464154482508064,6.739087866698141), (51.464172734232235,6.739071377317364), (51.464183024539835,6.739047157912321), (51.46419395425206,6.739020047700429), (51.46420877083953,6.738992929967574), (51.46422267511233,6.738971629815222)]]
        # looks correct, but was obtained before gps_util was written.
        #gps_pos_optimised = [gps_util.degree_to_radiants(gps) for gps in [(51.46411414940673,6.738944977588712), (51.46444738631447,6.738809863549967), (51.464376920188684,6.739083154724557), (51.46446718888097,6.738927768062857), (51.46447029093418,6.738909215184113), (51.46433948081566,6.738858198040306), (51.464340634576544,6.738867556464109), (51.464332301312695,6.738860518358256), (51.46433352256563,6.7388693023431845), (51.46425560352164,6.738978243689988), (51.46427108152239,6.738956349310485), (51.46430420716902,6.738933864796242), (51.46434835826103,6.738916791956062), (51.46439114717442,6.738898908652391), (51.464435010009225,6.738881909437699), (51.46447785743073,6.738871411732899), (51.46450255330567,6.738853593635997), (51.464523350787786,6.738819787999573), (51.464533504173595,6.738776268215176), (51.46453037892502,6.738730218636211), (51.464517197636866,6.7386905132803125), (51.46449659091481,6.738663015861728), (51.46446663771645,6.738648698012101), (51.464438661684675,6.738653402331532), (51.46441130495734,6.738677405727721), (51.46439626718731,6.73871541407409), (51.464389944382496,6.7387554556266025), (51.46438436411264,6.738785735746747), (51.46437249221701,6.738802412049573), (51.46433474605532,6.738820935341453), (51.46429112213464,6.738838149987502), (51.46425084032891,6.738857744036035), (51.46420852749894,6.73887419245192), (51.46416856291396,6.738895720484548), (51.46412900038989,6.738912834604563), (51.46410496482645,6.738923655505124), (51.464079595253736,6.738958957232722), (51.464067119195114,6.738998752044335), (51.464066961660166,6.739049464522678), (51.46407733183846,6.739086333417893), (51.46410152721491,6.739123619222857), (51.46413179871855,6.739137612689789), (51.46415989439572,6.7391333201885475), (51.46419621559581,6.739098942910411), (51.464209958613,6.739064370135348), (51.464218071281536,6.739045220236359), (51.464230863835915,6.739022540553707), (51.464235195017565,6.73894986483897), (51.46425254304922,6.738924631810385), (51.464298461923505,6.738896452978141), (51.4643428654891,6.7388789814292425), (51.46438668534625,6.738860624573461), (51.46442779156127,6.7388451840515105), (51.46447030037779,6.7388281064257844), (51.46448691323747,6.738816478338111), (51.46449890894766,6.738795981253502), (51.46450483971843,6.738769736369862), (51.4645028126984,6.738742394083106), (51.464494201192664,6.738718220083149), (51.464480374977576,6.738700955868589), (51.46446375293953,6.738694369123524), (51.464446342496,6.7386975953166015), (51.46443094576782,6.738710948324009), (51.46442003090998,6.738732241930608), (51.46441452623774,6.738758460709004), (51.4644094765204,6.738799210265234), (51.464398695574275,6.738827028726462), (51.464380796405976,6.738840788713605), (51.4643624111585,6.738848593652515), (51.4643380159601,6.738858309555241), (51.46429508893946,6.738875725158482), (51.46425547308892,6.738894756649162), (51.464215777303274,6.738910776637766), (51.46417554371516,6.7389327114558935), (51.46413319131306,6.738958079522211), (51.46411580753114,6.73896635691615), (51.46410299161643,6.738986446365815), (51.46409511800807,6.7390122939678285), (51.46409565819885,6.739041007544318), (51.46410386007197,6.739066872605536), (51.4641179539995,6.739085351021696), (51.46413588370013,6.739092211889237), (51.464154482508064,6.739087866698141), (51.464172734232235,6.739071377317364), (51.464183024539835,6.739047157912321), (51.46419395425206,6.739020047700429), (51.46420877083953,6.738992929967574), (51.46422267511233,6.738971629815222)]]
        #gps_pos_optimised = [(0.898218219196664,0.11761677384927945), (0.8982240263633601,0.11761440048511256), (0.8982228208377531,0.11761917699687374), (0.8982243851864168,0.11761646624489688), (0.8982244265640591,0.11761613035106883), (0.8982218946550489,0.11761559814002326), (0.8982218938811084,0.11761570098767488), (0.8982218149295176,0.11761558849952686), (0.8982218156214595,0.11761568281976631), (0.898220710983085,0.11761735052795479), (0.898220978494443,0.11761696934924652), (0.8982215456554087,0.1176165589296821), (0.8982223091257496,0.11761626454942113), (0.8982230487531687,0.11761596120613443), (0.8982238102598032,0.11761566273739824), (0.8982245598301507,0.11761548097132278), (0.8982249870659813,0.11761516890051886), (0.8982253377320112,0.11761457351741167), (0.8982255063560869,0.1176138166202877), (0.8982254417874138,0.11761300752690491), (0.8982252066406282,0.11761231329768941), (0.8982248452265666,0.11761182984846542), (0.8982243198428156,0.11761158278914603), (0.8982238335827609,0.1176116613067907), (0.898223361984392,0.11761209061621343), (0.8982231161606045,0.11761272297037856), (0.8982230191214551,0.11761342217100187), (0.898222915051808,0.1176139797115508), (0.8982227126497527,0.11761427495956295), (0.8982220583954695,0.11761459772454162), (0.8982212939134352,0.11761489442892041), (0.8982206047210792,0.11761524097804477), (0.898219869335219,0.1176155300751091), (0.898219178245082,0.11761590604955019), (0.8982184915495137,0.11761620408389882), (0.8982180777163289,0.1176163952015395), (0.8982176416208973,0.11761701177458551), (0.898217430957465,0.1176177163612042), (0.8982174374194676,0.11761859771040431), (0.8982176254689792,0.11761924547658863), (0.8982180576473158,0.11761989621541709), (0.898218589784458,0.11762013926499308), (0.8982190816576179,0.11762006481849195), (0.8982196962977222,0.1176194612651902), (0.8982199340976174,0.11761885410670762), (0.898220071904213,0.11761851946248372), (0.8982202888353831,0.11761812866047128), (0.8982203507713146,0.11761685217122334), (0.8982206459024316,0.1176164149578317), (0.8982214284084625,0.11761591559656409), (0.8982222029890452,0.11761561188460036), (0.898222967750002,0.1176152993564572), (0.8982236824628241,0.11761502403366299), (0.8982244192117744,0.11761472238430348), (0.8982247045071223,0.11761451225122162), (0.8982249096590265,0.11761414911534866), (0.8982250072298149,0.11761369099046028), (0.8982249614609337,0.11761321716279867), (0.8982248072729057,0.11761280105367879), (0.8982245669728158,0.1176124952791572), (0.8982242740079597,0.11761237300886834), (0.8982239703596181,0.11761243096935899), (0.8982237053549882,0.11761267268776333), (0.8982235223021843,0.11761305154077652), (0.8982234265960101,0.11761350536866697), (0.8982233409236279,0.1176142456812405), (0.8982231742148179,0.11761470490248802), (0.8982228598869461,0.11761494843930137), (0.8982225364027688,0.11761508542012415), (0.8982221200565467,0.11761524341376464), (0.8982213755722425,0.11761554807805627), (0.8982206936194123,0.11761589106571084), (0.8982200051432438,0.11761616782657454), (0.8982193083875797,0.11761655419450856), (0.8982185724863053,0.117616995783369), (0.8982182705560944,0.11761714058233069), (0.898218049845183,0.11761748925418344), (0.8982179184728051,0.1176179427819382), (0.898217935170833,0.11761844320110212), (0.898218080845213,0.11761889524618208), (0.8982183272257223,0.11761921950259445), (0.8982186438165033,0.11761934968664887), (0.898218967330849,0.11761927794497112), (0.8982192838198799,0.11761896785619438), (0.8982194631694919,0.11761854184059344), (0.8982196467080553,0.11761808439862703), (0.8982199006040388,0.11761760611102237), (0.8982201370314852,0.11761723655621349)]  # with new gps_util method
        gps_pos_optimised = [(0.8982182234878937,0.11761677401554493), (0.8982240283230206,0.11761440317336193), (0.8982228170818565,0.11761917319413157), (0.8982243857590018,0.11761646770990754), (0.8982244271621759,0.11761613484277073), (0.8982213844585475,0.11761598941181839), (0.8982213848761311,0.11761599186563093), (0.8982213939332642,0.11761598320634509), (0.8982213904563322,0.11761597398023396), (0.8982207127624222,0.11761734744161406), (0.8982209760115857,0.1176169686056584), (0.8982215439275895,0.11761655930264585), (0.8982223077464857,0.11761626572893127), (0.8982230483868425,0.11761596725188847), (0.8982238096321328,0.11761566637849627), (0.8982245596483993,0.11761548864916314), (0.8982249819730231,0.11761517199604024), (0.8982253368833978,0.11761458077777534), (0.8982255030900722,0.11761382123124949), (0.898225447183784,0.11761301742240739), (0.8982252110144567,0.11761232810533767), (0.8982248519130996,0.1176118462628097), (0.8982243287436013,0.11761159688349265), (0.8982238387564737,0.1176116845319551), (0.898223368869313,0.11761210044032144), (0.8982231223058281,0.11761274062642617), (0.8982230209395806,0.11761343434028235), (0.8982229206983123,0.11761399348142915), (0.898222717627581,0.1176142835697097), (0.8982220622722421,0.11761461084675427), (0.8982212955292291,0.11761490598513104), (0.8982206112901114,0.1176152522660883), (0.8982198769854388,0.11761553425899654), (0.898219185159113,0.11761591243628926), (0.8982185020043061,0.11761621676535045), (0.898218084570002,0.11761640056576247), (0.8982176482977221,0.11761701278046427), (0.8982174345520092,0.11761770589005656), (0.8982174382508502,0.11761859101897006), (0.8982176249481852,0.11761923622323167), (0.8982180536751105,0.11761987791794339), (0.8982185841039327,0.11762012320428758), (0.8982190702389945,0.11762004854004705), (0.8982196929279495,0.11761944804410492), (0.8982199298134613,0.11761885078516811), (0.8982200712609417,0.11761851223896709), (0.8982202860806936,0.11761812295330196), (0.8982203511382697,0.11761685524391803), (0.8982206475933272,0.11761641718416244), (0.8982214282158968,0.1176159161961673), (0.8982222010192281,0.11761561620903986), (0.898222967630039,0.11761530553184169), (0.898223684375342,0.11761502589625013), (0.8982244161379449,0.11761472666226179), (0.8982247022392744,0.11761451990482338), (0.8982249082879257,0.11761415662866992), (0.8982250039797648,0.11761370110567296), (0.8982249651030323,0.11761322820271117), (0.8982248115235022,0.11761281265239247), (0.8982245726066329,0.11761250887310425), (0.8982242811683868,0.11761239019909261), (0.8982239768391174,0.11761244807650233), (0.8982237116270445,0.11761268035194303), (0.8982235273770167,0.11761305898677114), (0.8982234335652678,0.11761351516990351), (0.8982233422254021,0.11761425236867185), (0.8982231750738204,0.1176147088004039), (0.8982228621181694,0.11761495421716618), (0.8982225397447375,0.11761509300978924), (0.8982221228161205,0.1176152556984028), (0.8982213782338375,0.11761555457679386), (0.8982206959150201,0.11761589637706767), (0.8982200096870406,0.117616173004985), (0.8982193118618405,0.1176165583212593), (0.898218576444067,0.11761700026184876), (0.8982182741018131,0.11761714295644894), (0.8982180548910497,0.11761749339765243), (0.8982179222761699,0.11761794225651936), (0.8982179386997172,0.11761844043930811), (0.8982180819073882,0.1176188885780413), (0.8982183270825627,0.11761921019622713), (0.8982186434717904,0.11761933576345569), (0.8982189654237478,0.11761926128737678), (0.8982192809565324,0.11761895895116418), (0.8982194603685004,0.11761853365189182), (0.8982196452487042,0.11761807291030295), (0.8982198979389193,0.11761760291706577), (0.8982201360624948,0.117617227105226)]
        # carpos = {donre3_framenr: [front_left_tire, front_right_tire, rear_left_tire, rear_right_tire]:[gps_util.gps_pos_radiants]
        # maybe have this in a file or something. its 180 lines long.
        carpos = {}
        with open("C:/Users/Idefix/PycharmProjects/datasets/testrun_2022_12_17/processed_data/dronefrnr_carposes.txt") as f:
            for line in f.readlines():
                dronefrnr = int(line.split("_")[0])
                carposes = [np.array([float(tmp.split("#")[0]), float(tmp.split("#")[1])]) for tmp in line.split("_")[1].split(",")]
                carpos[dronefrnr] = carposes
    else:
        # fit function from pixel-positions in drone image to gps-positions on earth
        # calculate positions from data
        # fixed points, where coordinates in google maps and pixel frame are known
        def loss_factory(a_pxpos, b_pxpos, c_pxpos, d_pxpos, e_pxpos, a_gmpos=gully_u_gmpos, b_gmpos=gully_o_gmpos, c_gmpos=well_gmpos, d_gmpos=dach_u_gmpos, e_gmpos=dach_o_gmpos):
            def loss(t):
                t_mat = np.array([[t[0], t[1]], [t[2], t[3]]])
                offset = np.array([t[4], t[5]])
                # a_pxpos: (rel_pxpos_w, rel_pspos_h), np.array
                # a_gmppos: gps_point_radiant, np.array
                # why look up how to properly set tolerace when the loss function can be scaled?
                f = 10
                return np.sum(((np.matmul(a_pxpos, t_mat)+offset)*f - a_gmpos*f) ** 2) + \
                       np.sum(((np.matmul(b_pxpos, t_mat)+offset)*f - b_gmpos*f) ** 2) + \
                       np.sum(((np.matmul(c_pxpos, t_mat)+offset)*f - c_gmpos*f) ** 2) + \
                       np.sum(((np.matmul(d_pxpos, t_mat)+offset)*f - d_gmpos*f) ** 2) + \
                       np.sum(((np.matmul(e_pxpos, t_mat)+offset)*f - e_gmpos*f) ** 2)
            return loss  # scipy.opimize.minimize: (success, fails) = (100, 86)
        frnr_losses = []
        with open(point_file) as f:
            lines = f.readlines()
            #line[0] = C:\Users\Idefix\PycharmProjects\tmpProject\cam_footage\DJI_0456_frames\frame_2075.jpg,0.0984375#0.41015625,0.9765625#0.4296875,0.5078125#0.8583984375,0.48125#0.6181640625,0.47291666666666665#0.568359375,0.5151041666666667#0.6025390625,0.5083333333333333#0.556640625,0.5140625#0.5654296875
            for line in lines:
                framenr = int(line.split(",")[0].split("_")[-1].replace(".jpg", ""))
                points_pxpos = [np.array((float(tup.split("#")[0])-0.5, float(tup.split("#")[1])-0.5)) for tup in line.split(",")[1:]]
                gully_u_pxpos, gully_o_pxpos, well_pxpos, dach_u_pxpos, dach_o_pxpos = points_pxpos[:5]
                frnr_losses.append((framenr, loss_factory(a_pxpos=gully_u_pxpos, b_pxpos=gully_o_pxpos, c_pxpos=well_pxpos, d_pxpos=dach_u_pxpos, e_pxpos=dach_o_pxpos), points_pxpos))

        # fit liner transformation t pixelPosition -> googlemapsPosition, so that t(objp)-objg for (objp, objg) in [gully_u, gully_o, well] is minimised
        #t_flat = np.array([7.313843509836915e-06,  9.065006222537148e-07, 8.031148742488077e-06, 1.4172729793492748e-06, 0.8982205375970675, 0.11761627101674235])
        t_flat = np.array([8.788436009204054e-06, -4.954819897553732e-06, 1.8398831516808416e-06, 7.693741050805764e-06, 0.8982214991470836, 0.11761587519893994])

        gps_pos_all = {}  # framenumber -> [(gps_lat, gps_long)], index look at point_of_interest_numbers.jpg
        all_res = []
        success = 0
        fails = 0
        for (frnr, loss, pxposs) in frnr_losses:
            print("frnr = ", frnr)
            res = scipy.optimize.minimize(loss, t_flat)  # tol = 1m
            if res.success:
                success += 1
                t = res.x
                print("res = ", t)
                all_res.append(t)
                t_mat = np.array([[t[0], t[1]], [t[2], t[3]]])  # rotation+scale
                offset = np.array([t[4], t[5]])  # gps-position of center of image
                gps_pos = [np.matmul(cp, t_mat)+offset for cp in pxposs]
                gps_pos_all[frnr] = gps_pos

                carpos[frnr] = gps_pos[poii_carpos_points_range[0]:poii_carpos_points_range[1]]
                print("dist(fixpoint, gps_pos) = ", [gps_util.gps_to_dist(fp, gps_pos_p) for (fp, gps_pos_p) in [(gully_u_gmpos, gps_pos[0]), (gully_o_gmpos, gps_pos[1]), (well_gmpos, gps_pos[2]), (dach_u_gmpos, gps_pos[3]), (dach_o_gmpos, gps_pos[4])]])
            else:
                fails += 1
                print("minimize failed, res = ", res)
        print(f"scipy.opimize.minimize: (success, fails) = ({success}, {fails})")
        all_res_avg = [0]*len(all_res[0])
        for i in range(len(all_res[0])):
            all_res_avg[i] = np.average([all_res[j][i] for j in range(len(all_res))])
        print("optimize t_flat average =", all_res_avg)
        print("carpos =", str(carpos).replace("array(", "np.array(").replace("])], ", "])], \n"))
        assert len(gps_pos_all[list(gps_pos_all.keys())[0]]) == 88
        gps_pos_avg = [(0.9, 0.12)]*88
        for i in range(len(gps_pos_avg)):
            gps_pos_avg[i] = (np.average([gps_pos_all[k][i][0] for k in gps_pos_all.keys()]), np.average([gps_pos_all[k][i][1] for k in gps_pos_all.keys()]))
        gps_pos_avg = np.array(gps_pos_avg)
        print("gps_pos_avg = [", ", ".join([f"({gp[0]},{gp[1]})" for gp in gps_pos_avg])+"]")
        # fit cone positions over known distances between cones
        cw = 0.228  # width of cone baseplate in m
        l = 5+cw
        kw = 3+cw
        dists_m_const = {(0, 1): 38.28, (0, 2): 30.86, (0, 3):39.18, (0, 4):39.5, (1, 2): 20.43, (1, 3): 8.38, (1, 4): 7.25, (2, 3): 14.9, (2, 4): 15.86, (3, 4): 1.1}
        dists_m_car = {(5, 6): 0.65, (7, 8): 0.65, (5, 7): 0.8, (6, 8): 0.8}
        dists_m_cones = {(11, 12): l, (12, 13): l, (13, 14): l, (30, 29): l, (67, 68): 2, (49, 50): l, (50, 51): l, (49, 51):2*l, (67, 69):l, (69, 70):l,
                   (74, 75): 2, (75, 76): 2, (76, 77): 2, (77, 78): 2, (78, 79): 2, (79, 80): 2, (80, 81): 2, (81, 82): 2, (82, 83): 2, (83, 84): 2,
                   (36, 76): kw, (37, 77): kw, (38, 78): kw, (39, 79): kw, (40, 80): kw, (41, 81): kw, (42, 82): kw, (43, 83): kw, (44, 84): kw,
                   (11, 49): 2.5+cw, (12, 50): 2.5+cw, (13, 51): 2.5+cw, (14, 52): 2.5+cw,
                   (53, 54): 2, (54, 55): 2, (55, 56): 2, (56, 57): 2, (57, 58): 2, (58, 59): 2, (60, 61): 2, (61, 62): 2, (62, 63): 2, (63, 64): 2,
                   (17, 55): kw, (18, 56): kw, (19, 57): kw, (20, 58): kw, (21, 59): kw, (22, 60): kw, (23, 61): kw, (24, 62): kw, (25, 64): kw, (26, 65): kw}
        dists_m = {**dists_m_cones, **dists_m_const}  # replace with dist_m = dist_m_cones | dist_m_const when python version >= 3.9
        #for (k0, k1) in dists_m.keys():
        #    print(f"dist_m[{k0}][{k1}] = {dists_m[(k0,k1)]} = {abs_meter_diff(gps_pos[k0], gps_pos[k1])}")

        def loss(flat_gps_pos):
            dists_loss = np.sum(np.array([gps_util.gps_to_dist((flat_gps_pos[2*k0], flat_gps_pos[2*k0+1]), (flat_gps_pos[2*k1], flat_gps_pos[2*k1+1]))-dists_m[(k0, k1)] for (k0, k1) in dists_m.keys()])**2)
            mespos_loss = np.sum(np.array([gps_util.gps_to_dist(gps_pos_avg[i], (flat_gps_pos[2*i], flat_gps_pos[2*i+1])) for i in range(len(gps_pos_avg)) if i not in [5, 6, 7, 8]])**2)
            return dists_loss + mespos_loss

        flat_gpspos = np.array([gps_pos_avg[int(i//2)][int(i%2)] for i in range(2*len(gps_pos_avg))])
        print("prae_loss = ", loss(flat_gpspos))

        res = scipy.optimize.minimize(loss, flat_gpspos)

        res_flat_gpspos = res.x
        print("post_loss = ", loss(res_flat_gpspos))
        gps_pos_optimised = [(res_flat_gpspos[2*i], res_flat_gpspos[2*i+1]) for i in range(len(gps_pos_avg))]
        #print("success = ", res.success)
        print("gps_pos_optimised = [", ", ".join([f"({gp[0]},{gp[1]})" for gp in gps_pos_optimised])+"]")
        print("dist(gps_pos, gps_pos_optimised) = ", [gps_util.gps_to_distazimuth(gps, opt_gps) for (gps, opt_gps) in zip(gps_pos_avg, gps_pos_optimised)])

    return gps_pos_optimised, carpos


def plot_on_googlemaps(points: gps_util.gps_pos_radiant):
    from plot_on_googlemaps import CustomGoogleMapPlotter
    gmap = CustomGoogleMapPlotter(51.4639933,  6.73884552, 25, map_type='satellite')  # making 25 larger -> smaler map cutout shown by default
    #gmap.draw("map.html")
    r2d = 180/np.pi
    fixpoitns_gpspos = points[poii_const_points_range[0]:poii_const_points_range[1]]
    carpos = points[poii_carpos_points_range[0]:poii_carpos_points_range[1]]
    blue_cone_gpspos = points[poii_bluecones_range[0]:poii_bluecones_range[1]]
    yellow_cone_gpspos = points[poii_yellowcones_range[0]:poii_yellowcones_range[1]]
    gmap.color_scatter([x*r2d for (x, y) in fixpoitns_gpspos], [y*r2d for (x, y) in fixpoitns_gpspos], size=0.228, c="black")
    gmap.color_scatter([x*r2d for (x, y) in carpos], [y*r2d for (x, y) in carpos], size=0.228, c="purple")
    gmap.color_scatter([x*r2d for (x, y) in blue_cone_gpspos], [y*r2d for (x, y) in blue_cone_gpspos], size=0.228, c="blue")
    gmap.color_scatter([x*r2d for (x, y) in yellow_cone_gpspos], [y*r2d for (x, y) in yellow_cone_gpspos], size=0.228, c="yellow")
    gmap.draw('map.html')


def get_boundingboxes_keypoints_poii(cam: str, framenr: int) -> [(cone_bounding_box, cone_keypoitns, poii_id)]:
    bbs = get_boundingboxes(cam, framenr)
    return [(bb, get_cone_keypoints(cam, framenr, i), get_poii(cam, framenr, i)) for (i, bb) in enumerate(bbs)]


def get_boundingboxes(cam: str, framenr: int) -> [cone_bounding_box]:
    bounding_box_dir = pathlib.Path(f"./vp_labels/{cam}_bb/")
    bbfiles = [str(f) for f in os.listdir(bounding_box_dir)]
    filename = f"{cam}_frame_{framenr}.txt"
    if filename in bbfiles:
        bounding_box = []
        with open(bounding_box_dir/filename) as f:
            for line in f.readlines():
                tmp = line.split(" ")
                bounding_box.append((int(tmp[0]), float(tmp[1]), float(tmp[2]), float(tmp[3]), float(tmp[4])))
        return bounding_box
    else:
        print(f"no bounding box file found for: {filename}")
        return []


def get_cone_keypoints(cam, framenr, cone) -> [(normalised_px_w, normalised_px_h)]:
    cone_keypoints_file = pathlib.Path("./vp_labels/cone_annotations.csv")
    filename = f"/cones/{cam}_frame_{framenr}.jpg_cone_{cone}.jpg"
    with open(cone_keypoints_file) as f:
        for line in f.readlines():
            #/cones/camL3_frame_2642.jpg_cone_8.jpg
            if line.split(",")[0] == filename:
                keypoints = [(float(strtup.split("#")[0]), float(strtup.split("#")[1])) for strtup in line.split(",")[1:]]
                assert len(keypoints) == 7
                return keypoints
    return None


poii_pxpos = {}  # ("drone3", framenr) -> [pixel_position of keypoints]
def get_poii_positions(cam, framenr) -> [(normalised_px_w, normalised_px_h)]:
    # returns poii positions
    if len(poii_pxpos.keys()) == 0:
        file = pathlib.Path("./vp_labels/droneview_annotations.csv")
        with open(file) as f:
            for line in f.readlines():
                #C:\Users\Idefix\PycharmProjects\datasets\keypoints\droneview\d3_frame_3285.jpg	0.1527777777777778#0.39	0.7877777777777778#0.48777777777777775	0.5544444444444444#0.9588888888888889	0.7744444444444445#0.76	0.7911111111111111#0.72	0.6566666666666666#0.42	0.6572222222222223#0.45444444444444443	0.63#0.42	0.63#0.45666666666666667	0.38666666666666666#0.6155555555555555	0.4211111111111111#0.5877777777777777	0.49277777777777776#0.5811111111111111	0.5744444444444444#0.5955555555555555	0.6533333333333333#0.6066666666666667	0.735#0.6222222222222222	0.8161111111111111#0.6488888888888888	0.865#0.6422222222222222	0.9155555555555556#0.5988888888888889	0.9511111111111111#0.5222222222222223	0.9638888888888889#0.4266666666666667	0.9561111111111111#0.3322222222222222	0.9316666666666666#0.2544444444444444	0.8861111111111111#0.19333333333333333	0.8355555555555556#0.17	0.7788888888888889#0.18444444444444444	0.7361111111111112#0.23777777777777778	0.7083333333333334#0.30777777777777776	0.6872222222222222#0.37444444444444447	0.6605555555555556#0.39444444444444443	0.5861111111111111#0.3877777777777778	0.5061111111111111#0.37333333333333335	0.42777777777777776#0.36666666666666664	0.3461111111111111#0.3522222222222222	0.26722222222222225#0.3511111111111111	0.19333333333333333#0.34	0.14722222222222223#0.33444444444444443	0.08888888888888889#0.38	0.049444444444444444#0.44222222222222224	0.029444444444444443#0.5455555555555556	0.03166666666666667#0.6311111111111111	0.059444444444444446#0.7344444444444445	0.10777777777777778#0.7966666666666666	0.15611111111111112#0.8166666666666667	0.23444444444444446#0.7922222222222223	0.2727777777777778#0.7377777777777778	0.29388888888888887#0.71	0.32611111111111113#0.6755555555555556	0.3611111111111111#0.5366666666666666	0.4033333333333333#0.5066666666666667	0.49777777777777776#0.5022222222222222	0.5783333333333334#0.5166666666666667	0.6566666666666666#0.5255555555555556	0.74#0.5466666666666666	0.8194444444444444#0.5577777777777778	0.8538888888888889#0.55	0.8833333333333333#0.5277777777777778	0.9038888888888889#0.47444444444444445	0.9105555555555556#0.42	0.9066666666666666#0.35777777777777775	0.8916666666666667#0.31333333333333335	0.8616666666666667#0.2788888888888889	0.8288888888888889#0.26666666666666666	0.7972222222222223#0.27666666666666667	0.7716666666666666#0.3088888888888889	0.7522222222222222#0.36	0.7266666666666667#0.44	0.6961111111111111#0.4722222222222222	0.6572222222222223#0.48	0.6222222222222222#0.47444444444444445	0.5788888888888889#0.4688888888888889	0.5011111111111111#0.45555555555555555	0.42055555555555557#0.44666666666666666	0.3438888888888889#0.43555555555555553	0.26666666666666666#0.43444444444444447	0.18222222222222223#0.43555555555555553	0.14777777777777779#0.43444444444444447	0.11888888888888889#0.4588888888888889	0.095#0.5055555555555555	0.08444444444444445#0.56	0.08666666666666667#0.62	0.10277777777777777#0.6744444444444444	0.13055555555555556#0.7088888888888889	0.16444444444444445#0.7255555555555555	0.20722222222222222#0.7088888888888889	0.23277777777777778#0.67	0.2633333333333333#0.6288888888888889	0.29888888888888887#0.5911111111111111	0.33166666666666667#0.5666666666666667
                tmp = line.split(",")
                read_cam = tmp[0].split("_")[0]
                read_framenr = tmp[0].split("_")[2].replace(".jpg", "")
                pxpos = [(float(t.split("#")[0]), float(t.split("#")[1])) for t in tmp[1:]]
                assert len(pxpos) == 88
                poii_pxpos[(read_cam, read_framenr)] = pxpos
    if (cam, framenr) in poii_pxpos.keys():
        return poii_pxpos[(cam, framenr)]
    else:
        return None


def get_poii(cam: str, framenr: int, conenr: int) -> poii_id:
    file = pathlib.Path(f"./vp_labels/bbi_poii/{cam}_frame_{framenr}.txt")
    with open(file) as f:
        lines = f.readlines()
        return int(lines[conenr])


def custom_pnp_find_parameters():
    #run 3 (14:46)
    all_true_dist = []
    all_estfrom_dist = []
    all_true_angle = []
    all_estfrom_angle = []
    bbi_poii_label_dir = "C:/Users/Idefix/PycharmProjects/datasets/testrun_2022_12_17/bbi_poii/"
    image_frnr = [int(str(f).replace(".txt", "").split("_")[-1]) for f in os.listdir(bbi_poii_label_dir)]
    poi_true_gpspos, carpos = true_pos_from_droneimg_pxpos()
    print("image_frnr = ", [(camL3_frnr, get_synced_frame(camL3_frnr)) for camL3_frnr in image_frnr])
    for camL3_frnr in image_frnr:
        drone3_frnr = get_synced_frame(camL3_frnr)
        print("(camL3_frnr, drone3_frnr) =", (camL3_frnr, drone3_frnr))
        # load data from files
        car_gpsposes = carpos[drone3_frnr]
        name = f"drone3frame_{drone3_frnr}_camL3_{camL3_frnr}"
        bb_kp_poii = get_boundingboxes_keypoints_poii("camL3", camL3_frnr)
        bb_kp_poii = [(bb, kp, poii) for (bb, kp, poii) in bb_kp_poii if kp is not None]

        # car_gpspos and _bearing from drone image:
        car_gpspos = car_gpsposes[3]  # gps is about at location of rear right wheel
        car_bearing = gps_util.carposs_to_heading(car_gpsposes)  # bearing of vector from center of rear axis to center of front axis

        tmp = [gps_util.gps_to_distazimuth(poi_true_gpspos[poii], car_gpspos) for (bb, kp, poii) in bb_kp_poii]
        true_dist = [dist for (dist, azimuth) in tmp]
        true_angle = [to_range(azimuth-car_bearing) for (dist, azimuth) in tmp]

        all_true_dist += list(true_dist)
        all_true_angle += list(true_angle)

        estfrom_dist = np.array([avg_pxprom_from_conekeypoints(keypoints=kp, bounding_box=bb) for (bb, kp, poii) in bb_kp_poii])
        all_estfrom_dist += list(estfrom_dist)
        #fit_linpp_and_print(in_x=estfrom_dist, out_true=true_dist, name=f"dist_{name}")

        estfrom_heading = np.array([bb[1]-0.5*bb[3]+bb[3]*np.average(np.array([kp[i][0] for i in range(7)])) for (bb, kp, poii) in bb_kp_poii])
        all_estfrom_angle += list(estfrom_heading)
        #fit_linpp_and_print(in_x=estfrom_heading, out_true=true_angle, name=f"heading_{name}")

        # TODO test cv2.solvePnPRansac
        #cameraMatrix = np.array([[1.55902258e+03, 0, 1.03564443e+03], [0, 1.49628271e+03, 6.89322561e+02], [0, 0, 1]])
        #Basler_dist = np.array([[-2.42797289e-01, 9.77514487e-02, -8.00761502e-05, 5.61321688e-03, 1.08419697e-02]])
        #objectPoints = np.array([(0,0.325), (0.087,0.21666667), (0.128,0.10833333), (0.169,0), (-0.087,0.21666667), (-0.128,0.10833333), (-0.169,0)])
        #objectPoints = np.array([(x, y, 0) for (x, y) in objectPoints])  # height, width, depth
        #imgsize_h, imgsize_w = (1200, 1920)
        #for (bb, kp, poii) in bb_kp_poii:
        #    cls, posw, posh, sizew, sizeh = bb
        #    keypoints = np.array([((posw-0.5*sizew+w*sizew)*imgsize_w, (posh-0.5*sizeh+h*sizeh)*imgsize_h) for (w, h) in kp])
        #    retval, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=objectPoints, imagePoints=keypoints, cameraMatrix=cameraMatrix, distCoeffs=Basler_dist)
        #    print("\ntvec = ", tvec)  # no correlation between tvec and true meterpos detected
        #    cone_meterpos = gps_util.gps_to_meter(poi_true_gpspos[poii], gps_base=car_gpspos)
        #    print("true_dist = ", cone_meterpos)
        #    print(f"dists = {abs_value(tvec)} = {abs_value(cone_meterpos)}")

    all_true_dist = np.array(all_true_dist)
    all_estfrom_dist = np.array(all_estfrom_dist)
    fit_poly_fun_and_print(in_x=all_estfrom_dist, out_true=all_true_dist, name="dist_total", exponents=[0, 1, 2])
    # parameters[dist_total] = [-1.90217367e+03  3.05375783e+05 -9.73326783e-03  1.12787139e+01]
    # diff_dist_total =  1.0695141605801541

    all_true_angle = np.array(all_true_angle)
    all_estfrom_angle = np.array(all_estfrom_angle)
    fit_poly_fun_and_print(in_x=all_estfrom_angle, out_true=all_true_angle, name="bearing_total", exponents=[0, 1, 2])
    # parameters[bearing_total] = [ 0.96307476 -0.10296335 -0.00138886 -0.76054734]
    # diff_bearing_total =  0.0662130889486984


def custom_pnp_validate():
    labeld_frames = [(2632, 4029), (2032, 3279), (1282, 2341), (1287, 2348), (1292, 2354), (1297, 2360), (1302, 2366), (1307, 2373)]  # (camL3_framnr, drone3_framenr)]
    cls_est_true_pos = []
    poi_true_gpspos, dronefrmrn_carposes = true_pos_from_droneimg_pxpos()
    new_cetp_startindex = 0
    for (camL3_framenr, drone3_framenr) in labeld_frames:
        car_poses = dronefrmrn_carposes[drone3_framenr]
        gps_base = car_poses[3]
        car_posm = (0, 0)  # gps_util.gps_to_meter(dronefrmrn_carposes[drone3_framenr][3], gps_base)

        car_bearing = gps_util.carposs_to_heading(car_poses)
    # < visual pipeline modified
        bounding_boxes = get_boundingboxes("camL3", camL3_framenr)
        if len(bounding_boxes) == 0:
            print(f"visual_pipeline: no cone detected in camL3_frame_{camL3_framenr}")

        #print(f"\nvisual_pipeline: framenumber = {framenr}, car_bearing={car_bearing}, car_pos = {car_pos}")
        for i, bb in enumerate(bounding_boxes):
            keypoints = get_cone_keypoints("camL3", camL3_framenr, i)
            poii = get_poii("camL3", camL3_framenr, i)
            if keypoints is not None:
                dist, heading = custom_PnP(keypoints, bb)
                #heading = bearing_from_gps_points(cone_gpspos, car_gpspos)-car_bearing
                #print(f"{i}: cone {bb[0]} detected at (dist={dist}, heading={heading}")
                (north_m, east_m) = gps_util.distazimuth_to_meter((dist, to_range(car_bearing+heading)))
                est_pos = np.array([car_posm[0]+north_m, car_posm[1]+east_m])
                true_pos = np.array(gps_util.gps_to_meter(poi_true_gpspos[poii], gps_base=gps_base))
                cls_est_true_pos += [(bb[0], est_pos, true_pos)]  # (cls, est_pos, true_pos)

        # plot estimated cone positions and true cone positions in same image
        fig, (axe) = plt.subplots(1)
        axe.set_title(f"true_conepos-est_conepos of camL3 frame {camL3_framenr}")
        axe.set_xlabel("meter east")
        axe.set_xlabel("meter north")
        axe.grid()
    # >
        # plot all true cone positions
        # true positions x, estimated position o, unseen cones black, seen cones blue/yellow
        tmp = [gps_util.gps_to_meter(yc, gps_base=gps_base) for yc in poi_true_gpspos[poii_yellowcones_range[0]:poii_yellowcones_range[1]] if gps_util.gps_to_dist(yc, gps_base) < 20]
        axe.scatter(x=[t[1] for t in tmp], y=[t[0] for t in tmp], c="black", marker='x', label="true_pos_yellow_unseen")
        tmp = [gps_util.gps_to_meter(bc, gps_base=gps_base) for bc in poi_true_gpspos[poii_bluecones_range[0]:poii_bluecones_range[1]] if gps_util.gps_to_dist(bc, gps_base) < 20]
        axe.scatter(x=[t[1] for t in tmp], y=[t[0] for t in tmp], c="black", marker='x', label="true_pos_blue_unseen")
        # plot vision cone
        car_poses_m = [gps_util.gps_to_meter(cp, gps_base) for cp in [car_poses[0], car_poses[1], car_poses[3], car_poses[2], car_poses[0], 0.5*np.array(car_poses[0])+0.5*np.array(car_poses[1]), 0.5*np.array(car_poses[2])+0.5*np.array(car_poses[3])]]
        axe.plot([t[1] for t in car_poses_m], [t[0] for t in car_poses_m], color="red")
        for (cls, est_pos, true_pos) in cls_est_true_pos[new_cetp_startindex:]:
            color = "blue" if cls==0 else "yellow"
            axe.scatter(x=[est_pos[1]], y=[est_pos[0]], c=color, marker='o', label="true_pos_seen")
            axe.scatter(x=[true_pos[1]], y=[true_pos[0]], c=color, marker='x', label="est_pos")
            axe.plot([est_pos[1], true_pos[1]], [est_pos[0], true_pos[0]], color=color)
            axe.text(x=0.5*est_pos[1]+0.5*true_pos[1], y=0.5*est_pos[0]+0.5*true_pos[0], s=f"{gps_util.meter_meter_to_dist(true_pos, est_pos)}"[:4])
        #axe.legend()
        fig.savefig(vis_out_path / f"true_est_conepos_camL3_frame_{camL3_framenr}.png")
        fig.show()
        new_cetp_startindex = len(cls_est_true_pos)

    # plot (distance between detected cone and true cone position) against true distance
    fig, (axe) = plt.subplots(1)
    axe.set_title(f"accuracy of visual pipeline over distance ({len(cls_est_true_pos)} datapoints from {len(labeld_frames)} frames")
    axe.scatter(x=[gps_util.meter_to_dist(true_pos) for (cls, est_pos, true_pos) in cls_est_true_pos], y=[gps_util.meter_to_dist(est_pos-true_pos) for (cls, est_pos, true_pos) in cls_est_true_pos], c=["blue" if cls==0 else "yellow" for (cls, est_pos, true_pos) in cls_est_true_pos])
    axe.set_xlabel("dist(car, true_cone_pos)")
    axe.set_ylabel("dist(estimated_cone_pos, true_cone_pos")
    axe.grid()
    fig.savefig(vis_out_path / "visual_pipeline_accuracy.png")
    fig.show()


def custom_PnP(keypoints: [(normalised_px_w, normalised_px_h)], bounding_box) -> (gps_util.meter, gps_util.heading_radiants):
    # custom_pnp_get_parameters: linear+-1 function, meadian
    #  parameters[dist_total] = [1.82951011e+03 2.78068401e+01 3.42363868e-04 6.50204337e-02]
    #  diff_dist_total =  5.572888817075419
    #  parameters[bearing_total] = [ 1.88694861 -0.56362367  0.02771185 -1.29870971]
    #  diff_bearing_total =  0.4052837611048084
    cls, posw, posh, sizew, sizeh = bounding_box
    mpropx = avg_pxprom_from_conekeypoints(keypoints=keypoints, bounding_box=bounding_box)  # avg of (meter dist on object / pixel dist in image)
    #dist = 1.86613152e+03*mpropx + 9.63512329e-02  # parameters using pos and bearing from drone image
    dist = 1.82951011e+03*mpropx + 2.78068401e+01*mpropx**2 + 3.42363868e-04/mpropx + 6.50204337e-02
    w = posw-0.5*sizew+sizew*np.average(np.array([keypoints[i][0] for i in range(7)]))  # avg width position of pixels
    #angle = 1.02105188*w - -0.97115736  # parameters using pos and bearing from drone image (camara seems to be rotated by -24.78 degree and has a field of view of 63 degree)
    angle = 1.88694861*w - 0.56362367*w**2 + 0.02771185/w - 1.29870971
    # angle = 0 -> in front of car.
    # anlge < 0 -> left of car.
    return dist, angle


def get_path(data: SensorDataDict):
    print("len(xx, xd, yx, yd) = ", len(data["GNSS_latitude_UsbFlRec_x"]), len(data["GNSS_latitude_UsbFlRec"]),
          len(data["GNSS_longitude_UsbFlRec_x"]), len(data["GNSS_longitude_UsbFlRec"]))
    xx, xd = remove_zeros(data["GNSS_latitude_UsbFlRec_x"], data["GNSS_latitude_UsbFlRec"])
    yx, yd = remove_zeros(data["GNSS_longitude_UsbFlRec_x"], data["GNSS_longitude_UsbFlRec"])
    print("len(xx, xd, yx, yd) = ", len(xx), len(xd), len(yx), len(yd))
    assert len(xx) == len(xd)
    assert len(yx) == len(yd)
    if all([len(t) == 0 for t in [xx, yx]]):
        print("cant visualise all 0 path of TODO")
        return
    #timesteps, xp, yp = timesinc(xx, xd, yx, yd)  # TODO maybe instead of reast of function
    #return timesteps, xp, yp
    if any([a != b for (a, b) in zip(xx, yx)]):
        print("times of xd and yd different, do timesinc")
        # join xd and yd over xx and xy
        timesteps = sorted(list(xx) + list(yx))
        ix = 0
        iy = 0
        xp = []
        yp = []
        # for each timestep in xx or xy:
        # add the weigthed average of xd to xp and yd to yp
        for ts in timesteps:
            # update ix anx iy to point to biggest time in xx or xy that is smaler than ts
            while ix + 1 < len(xx) and xx[ix + 1] < ts:
                ix += 1
            while iy + 1 < len(yx) and yx[iy + 1] < ts:
                iy += 1
            if ix + 1 >= len(xd):
                ix = len(xd) - 2
            if iy + 1 >= len(yd):
                iy = len(yd) - 2
            # average(a, wa, b, wb) = (wa*a+wb*b)/(wa+wb)
            xt = (abs(xx[ix + 1] - ts) * xd[ix] + abs(xx[ix] - ts) * xd[ix + 1]) / (abs(xx[ix + 1] - ts) + abs(xx[ix] - ts))
            xp.append(xt)
            yt = (abs(yx[iy + 1] - ts) * yd[iy] + abs(yx[iy] - ts) * yd[iy + 1]) / (abs(yx[iy + 1] - ts) + abs(yx[iy] - ts))
            yp.append(yt)
    else:
        print("times of xd and yd identical")
        l = min(len(xd), len(yd))
        xp = xd[:l]
        yp = yd[:l]
        timesteps = xx
    print(f"len(x) = {len(xp)}; len(yp) = {len(yp)}; len(t) = {len(timesteps)}")
    return timesteps, xp, yp


def plot_from_pos_and_sensor():
    # GNSS_heading_UsbFlRec*np.pi/180 = [gps_util.gps_to_azimuth(path[i+1], path[i]) for i in range(len(path))]
    #  but GNSS_heading_UsbFlRec*np.pi/180 is in range(0, 2pi), gps_to_azimuth is in range(-pi, pi)
    #  so GNSS_heading_UsbFlRec = 0: car is moving north
    run = read_csv(csv_files_dir / csv_files[0])
    print("show_sensorlogs.plot_from_pos_and_sensor: show heading for run ")
    heading_y = np.array(run["GNSS_heading_UsbFlRec"])*np.pi/180
    heading_x = run["GNSS_heading_UsbFlRec"+x_ending]
    speed_y = run["GNSS_speed_over_ground_UsbFlRec"]  # value
    speed_x = run["GNSS_speed_over_ground_UsbFlRec_x"]  # timestamps
    timesteps, xp, yp = get_path(run)
    # first 40 seconds doesnt contain anything usefull -> cut
    for i, v in enumerate(heading_x):
        if v > 40:
            heading_x = heading_x[i:]
            heading_y = heading_y[i:]
            print(f"heading_x cutout = {i}")
            break
    for i, v in enumerate(speed_x):
        if v > 40:
            speed_x = speed_x[i:]
            speed_y = speed_y[i:]
            print(f"speed_x cutout = {i}")
            break
    for i, v in enumerate(timesteps):
        if v > 40:
            timesteps = timesteps[i:]
            xp = xp[i:]
            yp = yp[i:]
            print(f"position_x cutout = {i}")
            break
    speed_from_pos_x = []
    speed_from_pos_y = []
    heading_from_pos_x = []
    heading_from_pos_y = []
    xp, yp = np.pi/180*np.array(xp), np.pi/180*np.array(yp)  # convert positions to radiant
    dt = 1000  # take heading and speed over 1s intervals to avoid noise  # 1000 entries per Seconds in sensordata
    print("dt = ", dt)
    print("path[0] =", (xp[0], yp[0]), (xp[dt], yp[dt]))
    print("range(",len(xp)-dt,")")
    for i in range(len(xp)-dt):
        # speed: average over 3 seconds
        d_t = timesteps[i+dt]-timesteps[i]
        dist, heading = gps_util.gps_to_distazimuth((xp[i+dt], yp[i+dt]), (xp[i], yp[i]))
        if d_t > 0:
            #time
            speed_from_pos_x.append(0.5*timesteps[i+dt]+0.5*timesteps[i])
            #speed
            speed_from_pos_y.append(dist/d_t)
        # heading: average over 1 second
        if d_t > 0 and dist > 0:
            #heading
            heading_from_pos_x.append(0.5*timesteps[i+dt]+0.5*timesteps[i])
            #x_pos: lattitude position
            #y_pos: longitude position
            heading_from_pos_y.append(heading)
    speed_from_pos_x = np.array(speed_from_pos_x)
    speed_from_pos_y = np.array(speed_from_pos_y)
    heading_from_pos_x = np.array(heading_from_pos_x)
    heading_from_pos_y = np.array(heading_from_pos_y)
    # heading_from_pos_y in range(-pi, pi) with 0 = south, -pi/2 = east
    # heading_y
    #plot

    fig, (axe) = plt.subplots(1)
    axe.set_title("speed")
    #axe.plot(from_pos_x, speed_from_pos_y, label="speedf")
    axe.plot(speed_x, speed_y, label="speedf")
    #padded_y = np.array([speed_from_pos_y[0]] * (n // 2) + list(speed_from_pos_y) + [speed_from_pos_y[-1]] * (n - n // 2 - 1))
    #y_avg = np.convolve(padded_y, np.ones(n) / n, mode='valid')
    axe.plot(speed_from_pos_x, speed_from_pos_y, "--", color="green", linewidth=0.5, label="average speedf s")
    axe.legend()
    axe.grid()
    plt.show()

    fig, (axe) = plt.subplots(1)
    axe.set_title("heading")
    print("len(heading_x, heading_y, from_pos_x, heading_from_pos_y) = ", len(heading_x), len(heading_y), len(heading_from_pos_x), len(heading_from_pos_y))
    axe.plot(heading_from_pos_x, heading_from_pos_y, label="heading_from_pos")
    axe.plot(heading_x, heading_y, label="gps_heading*pi/180")
    axe.legend()
    axe.grid()
    plt.show()


def plot_cones(car_pos: [gps_util.gps_pos_radiant], yellow_cones: [gps_util.gps_pos_radiant], blue_cones: [gps_util.gps_pos_radiant], name: str=None, blue_cones_labes: [str]=None, yellow_cones_labes: [str]=None, true_conepos: [gps_util.gps_pos_radiant]=None, save_dir: str=None) -> plot:
    # may be gps_pos_degree or (meter_north, meter_east), as long as all inputs have the same type
    fig, (axe) = plt.subplots(1)
    if name is None:
        name = "cone positions"
    axe.set_title(name)
    if car_pos is not None:
        axe.plot([long for (lat, long) in car_pos], [lat for (lat, long) in car_pos], color="black")
        axe.scatter([long for (lat, long) in car_pos], [lat for (lat, long) in car_pos], color="green", s=1)
    if true_conepos is not None:
        axe.scatter([long for (lat, long) in true_conepos], [lat for (lat, long) in true_conepos], color="gray", marker='x', s=2)
    if len(car_pos) > 1:
        dx, dy = car_pos[1][1]-car_pos[0][1], car_pos[1][0]-car_pos[0][0]
        axe.arrow(x=car_pos[0][1], y=car_pos[0][0], dx=dx, dy=dy, width=0.01*np.sqrt(dx**2+dy**2))  # arrow from first to second
    if len(car_pos) > 2:
        dx, dy = car_pos[-1][1]-car_pos[-2][1], car_pos[-1][0]-car_pos[-2][0]
        axe.arrow(x=car_pos[-1][1], y=car_pos[-1][0], dx=dx, dy=dy, width=0.01*np.sqrt(dx**2+dy**2))  # arrow from first to second
    axe.scatter(x=[long for (lat, long) in yellow_cones], y=[lat for (lat, long) in yellow_cones], color="yellow")
    axe.scatter(x=[long for (lat, long) in blue_cones], y=[lat for (lat, long) in blue_cones], color="blue")
    if yellow_cones_labes is not None:
        assert len(yellow_cones) == len(yellow_cones_labes)
        for i in range(len(blue_cones)):
            axe.text(yellow_cones[i][1], yellow_cones[i][0], yellow_cones_labes[i], c="black")
    if blue_cones_labes is not None:
        assert len(blue_cones) == len(blue_cones_labes)
        for i in range(len(blue_cones)):
            axe.text(blue_cones[i][1], blue_cones[i][0], blue_cones_labes[i], c="black")
    axe.grid()
    axe.set_xlabel("long")
    axe.set_ylabel("lat")
    if save_dir is not None:
        fig.savefig("vis_out/"+save_dir)
    fig.show()


def show_sensorlogs():
    """
     method for showing data from testruns.
     if only the .mat files recorded at that day exist, merge the different files that contain data from the same testrun and save the merged data in .csv files.
     else load from these .csv files
     visualisation is saved to vis_out_path
    """
    # real_motor_torque = 0.83*my_dict["Converter_L_Torque"]?*efficency  # 0.83 = motor_konstante from https://uni-duisburg-essen.sciebo.de/apps/files/?dir=/eTeam%20-%20Technik/06_A40-04/10_Datenbl%C3%A4tter&fileid=1101973513#pdfviewer
    # ax = real_motor_torque/(radius*mass)
    if len(csv_files) > 0:  # check if csv files already exist
        print("read data from csv files ", csv_files)
        my_dicts = [read_csv(csv_files_dir / csv_file) for csv_file in csv_files[2:]]  # TODO remove [2:] to read all csv files
        # my_dicts = [read_csv(csv_files_dir+csv_files[0])]
    else:
        print("read and merge data from mat files")
        my_dicts = read_mats(mat_files_dir)
        print("\n\n\n")
        print("save data to csv files")
        counter = 0
        for data in my_dicts:
            write_csv(f"{csv_files_dir}/alldata_{data[startTimestamp_key].strftime(time_format_str)}_id{counter}.csv",
                      data)
            counter += 1
    print("\n\n\n")
    print("merged data to ", len(my_dicts), " runs of length ", [data[length_key] for data in my_dicts])
    print("\n\n\n")

    for data in my_dicts:
        visualise_data(data)


# averegae_diff of 13:31:
# L_N / R_N = -5.83243368615474
# R_N / L_N = -0.17145501419856335
# GNSS_speed_over_ground / L_N = -0.001127671826231047
# L_N / GNSS_speed_over_ground = -886.7828258624362
# GNSS_speed_over_ground / R_N = 0.006502402469364128
# R_N / GNSS_speed_over_ground = 153.78931167554344

drone3_frame = int
def get_synced_frame(camL3_frame) -> drone3_frame:
    camL3_starttime = datetime.datetime.strptime("2022-12-17 14:44:56.48", "%Y-%m-%d %H:%M:%S.%f")
    drone3_starttime = datetime.datetime.strptime("2022-12-17 14:44:26.92", "%Y-%m-%d %H:%M:%S.%f")
    return round(25*(camL3_starttime+datetime.timedelta(seconds=camL3_frame/20)-drone3_starttime).total_seconds())


def print_synced_frame(camL3_frame):
    camL3_starttime = datetime.datetime.strptime("2022-12-17 14:44:56.48", "%Y-%m-%d %H:%M:%S.%f")
    camL3_somframe = 1281
    camL3_eomframe = 2643
    drone3_starttime = datetime.datetime.strptime("2022-12-17 14:44:26.92", "%Y-%m-%d %H:%M:%S.%f")
    drone3_somframe = 2360
    drone3_eomframe = 4039
    # camL3_starttime+camL3_frame/20 == drone3_starttime+drone3_frame/25
    # drone3_fram = 25*(camL3_starttime+camL3_frame/20-drone3_starttime)
    from_starttime = 25*(camL3_starttime+datetime.timedelta(seconds=camL3_frame/20)-drone3_starttime).total_seconds()
    from_som = drone3_somframe + 25*(camL3_frame-camL3_somframe)/20
    from_eom = drone3_eomframe - 25*(camL3_eomframe-camL3_frame)/20
    print(f"\ndrone3 start from from camL3_frame={camL3_frame}")
    print("time = ", camL3_starttime+datetime.timedelta(seconds=camL3_frame/20))
    print("from starttime: ", from_starttime)
    print("from som: ", from_som)
    print("from eom: ", from_eom)
    l = [from_starttime, from_som, from_eom]
    print("avg: ", np.average(l))
    print("meadian: ", sorted(l)[1])


def print_synced_pos_bearing_from_drone_and_sensors():
    poi_true_gpspos, carpos = true_pos_from_droneimg_pxpos()
    sensor_data = read_csv(csv_files_dir / "alldata_2022_12_17-14_43_59_id3.csv")
    td = 27.92  # seconds that sensor_data started recording before drone3
    for k in carpos.keys():
        #"GNSS_heading_UsbFlRec", "GNSS_latitude_UsbFlRec", "GNSS_longitude_UsbFlRec"
        gps_heading, i = get_at_time(sensor_data["GNSS_heading_UsbFlRec"+x_ending], sensor_data["GNSS_heading_UsbFlRec"], k/25+td)
        droneview_heading = gps_util.carposs_to_heading(carpos[k])
        print(f"gps_heading.at({k/25+td}={i}) = {gps_heading} = {(droneview_heading+np.pi)*180/np.pi} = bearing[{k}]")
    for k in carpos.keys():
        gps_lat, i = get_at_time(sensor_data["GNSS_latitude_UsbFlRec"+x_ending], sensor_data["GNSS_latitude_UsbFlRec"], k/25+td)
        gps_long = sensor_data["GNSS_longitude_UsbFlRec"][i]  # should be the same as line below
        #gps_long, i = get_at_time(sensor_data["GNSS_longitude_UsbFlRec"+x_ending], sensor_data["GNSS_longitude_UsbFlRec"], k/25+td)
        carposition = (np.average([lat for (lat, lon) in carpos[k]]), np.average([lon for (lat, lon) in carpos[k]]))
        #print(f"sensor_data.gps[{k/25+td}] = {(gps_lat, gps_long)}, \tcarpos[k] = {carposition}")
        print(f"abs meter diff between sensor_data.gps[{k/25+td}] and carpos[{k}] = ", gps_util.gps_to_dist((gps_lat, gps_long), carposition))


def visual_pipeline(cam="camL3", framenr=2632, car_bearing=0.0, car_pos: (gps_util.meter_north, gps_util.meter_east)=(0,0)) -> ([(gps_util.meter_north, gps_util.meter_east)], [(gps_util.meter_north, gps_util.meter_east)]):
    bounding_boxes = get_boundingboxes(cam, framenr)  # TODO replace with call to yolov5 NN
    if len(bounding_boxes) == 0:
        print(f"visual_pipeline: no cone detected in {cam}_frame_{framenr}")

    # get absolute cone positions from current car position, bearing and image.
    #  "image" means files containing the bounding boxes and keypoints.
    detections = []
    #print(f"\nvisual_pipeline: framenumber = {framenr}, car_bearing={car_bearing}, car_pos = {car_pos}")
    for i, bb in enumerate(bounding_boxes):
        keypoints = get_cone_keypoints(cam, framenr, i)  # TODO replace with call to keypointregression NN
        if keypoints is not None:
            dist, heading = custom_PnP(keypoints, bb)
            #heading = bearing_from_gps_points(cone_gpspos, car_gpspos)-car_bearing
            #print(f"{i}: cone {bb[0]} detected at (dist={dist}, heading={heading}")
            (north_m, east_m) = gps_util.distazimuth_to_meter((dist, to_range(car_bearing+heading)))
            detections.append((bb[0], north_m, east_m))
    #print("car_pos =", car_pos)
    blue_cone_pos = [(car_pos[0]+north_m, car_pos[1]+east_m) for (cls, north_m, east_m) in detections if cls == 0]
    yellow_cone_pos = [(car_pos[0]+north_m, car_pos[1]+east_m) for (cls, north_m, east_m) in detections if cls == 1]
    return blue_cone_pos, yellow_cone_pos


def get_nearst_gpspoint(gps_lat: [gps_util.lattitude], gps_long: [gps_util.longitude], pos: gps_util.gps_pos_radiant) -> (int, gps_util.meter):
    # returns i, dist, so that dist(pos, (gps_lat[i], gps_long[i]) is minimized and min_dist is that dist
    assert len(gps_lat) == len(gps_long)
    min_dist = gps_util.gps_to_dist((gps_lat[0], gps_long[0]), pos)
    min_i = 0
    for i in range(1, len(gps_lat)):
        dist = gps_util.gps_to_dist((gps_lat[i], gps_long[i]), pos)
        if dist < min_dist:
            min_dist = dist
            min_i = i
    return min_i, min_dist


def plot_gps_vs_droneview():
    sensordatadict = read_csv("merged_rundata_csv/alldata_2022_12_17-14_43_59_id3.csv")
    gps_lat_x = sensordatadict["GNSS_latitude_UsbFlRec"+x_ending]
    gps_lat_y = sensordatadict["GNSS_latitude_UsbFlRec"]*np.pi/180
    gps_long_x = sensordatadict["GNSS_longitude_UsbFlRec"+x_ending]
    gps_long_y = sensordatadict["GNSS_longitude_UsbFlRec"]*np.pi/180
    gps_heading_x = sensordatadict["GNSS_heading_UsbFlRec"+x_ending]
    gps_heading_y = sensordatadict["GNSS_heading_UsbFlRec"]*np.pi/180
    start_of_moving_index_gps = 0
    while gps_lat_x[start_of_moving_index_gps] < 120:
        start_of_moving_index_gps += 1
    print("start_of_moving_index_gps =", start_of_moving_index_gps)

    #plot_cones(car_pos=[gps_util.degree_to_radiants(cp) for cp in carpos], yellow_cones=gps_pos_optimised[poii_yelowcones_range[0]:], blue_cones=gps_pos_optimised[9:47], name="true_gps_pos", blue_cones_labes=[str(i) for i in range(9, 47)], yellow_cones_labes=[str(i) for i in range(47, 88)])
    #custom_pnp_find_parameters()
    poi_true_gpspos, carpos = true_pos_from_droneimg_pxpos()  # true_pos_from_droneimg_pxpos("C:/Users/Idefix/PycharmProjects/datasets/keypoints/droneview_annotations.csv")
    # carpos[frnr] = [np.array([lat, long]), np.array([lat, long]), np.array([lat, long]), np.array([lat, long])

    tdiff_dronesensor = 26.5  # framenumber_of_droneview3/25+tdiff = time of gps mesurment
    t_carpos = [(frnr/25+tdiff_dronesensor, carpos[frnr]) for frnr in carpos.keys()]
    t_carpos.sort(key=lambda x: x[0])
    print(f"t_carpos: from (framenr, t) {(min(carpos.keys()), t_carpos[0][0])} to {(max(carpos.keys()), t_carpos[-1][0])}")
    t = np.array([t for (t, y) in t_carpos])

    # plot path of all wheels from droneview and carpos from gps
    fig, (axe) = plt.subplots(1)
    axe.set_title("path of all wheels")
    name = ["FL", "FR", "RL", "RR"]
    for i in [0, 1, 2, 3]:
        axe.plot(np.array([y[i][1] for (t, y) in t_carpos]), np.array([y[i][0] for (t, y) in t_carpos]), label=f"carpos_{name[i]}")
    axe.plot(gps_long_y[start_of_moving_index_gps:], gps_lat_y[start_of_moving_index_gps:], label="gps_carpos")
    blue_cone_gpspos = poi_true_gpspos[poii_bluecones_range[0]:poii_bluecones_range[1]]
    yellow_cone_gpspos = poi_true_gpspos[poii_yellowcones_range[0]:poii_yellowcones_range[1]]
    axe.scatter(np.array([long for (lat, long) in blue_cone_gpspos]), np.array([lat for (lat, long) in blue_cone_gpspos]), color="blue")
    axe.scatter(np.array([long for (lat, long) in yellow_cone_gpspos]), np.array([lat for (lat, long) in yellow_cone_gpspos]), color="yellow")
    axe.legend()
    axe.grid()
    fig.show()

    # plot lattiutude and longitude from gps and droneview.RR against each other.
    name = ["lattitude", "longitude"]
    for i in [0, 1]:
        fig, (axe) = plt.subplots(1)
        axe.set_title(name[i])
        axe.plot(t, np.array([y[3][i] for (t, y) in t_carpos]), label="car_pos_rearright_long")
        if i == 0:
            axe.plot(gps_lat_x[start_of_moving_index_gps:], gps_lat_y[start_of_moving_index_gps:], label="gps_lat")
        else:
            axe.plot(gps_long_x[start_of_moving_index_gps:], gps_long_y[start_of_moving_index_gps:], label="gps_long")
        axe.legend()
        axe.grid()
        fig.show()

    # plot heading from gps and droneview against each other
    fig, (axe) = plt.subplots(1)
    axe.set_title("heading")
    axe.plot([t for (t, pos) in t_carpos], [gps_util.gps_to_azimuth(0.5*pos[0]+0.5*pos[1], 0.5*pos[2]+0.5*pos[3]) for (t, pos) in t_carpos], label="heading from drone")
    axe.plot(gps_heading_x[start_of_moving_index_gps:], gps_heading_y[start_of_moving_index_gps:], label="heading from gps")
    axe.legend()
    axe.grid()
    fig.show()

    # plot dist between droneview.RR and gps
    fig, (axe) = plt.subplots(1)
    axe.set_title(f"dist(gps_pos, drone_pos/25+{tdiff_dronesensor}")
    tmp = []
    for (t, pos) in t_carpos:
        t_lat, i_lat = get_at_time(x=gps_lat_x, y=gps_lat_y, t=t)
        t_long = gps_long_y[i_lat]
        tmp.append((t, gps_util.gps_to_dist(pos[3], (t_lat, t_long))))
    axe.plot([t for (t, d) in tmp], [d for (t, d) in tmp])
    axe.grid()
    fig.show()


class SLAM():
    def measurment(self, blue_cones_meterpos, yellow_cones_meterpos, car_mpos):
        # blue and yellow_cones_meterpos = output of visual pipeline.
        pass
    def localisation(self):
        # return current meter position of car
        return (0, 0)
    def get_map(self):
        # return map, as list of all cones in meterpos
        return [], []


class trackingSLAM(SLAM):
    def __init__(self):
        self.bc = []
        self.yc = []
        self.cp = (0, 0)
        self.seeing_cones = []  # (color:int, [meterpos], added:bool)
        self.seen_cones = []


    def measurment(self, blue_cones_meterpos, yellow_cones_meterpos, car_mpos):
        self.bc += list(blue_cones_meterpos)
        self.yc += list(yellow_cones_meterpos)
        self.cp = car_mpos
        for (cls, conepos) in [(0, bc) for bc in blue_cones_meterpos]+[(1, yc) for yc in yellow_cones_meterpos]:
            f = True
            for sc in self.seeing_cones:
                if sc[0] == cls and (conepos[0]-sc[1][-1][0])**2 + (conepos[1]-sc[1][-1][1])**2 < 1:  #  same color and less then 1 meter appart -> same cone
                    if sc[2]:
                        print("error: sc = ", sc)
                    sc[1].append(conepos)
                    sc[2] = True
                    f = False
                    break
            if f:
                self.seeing_cones.append([cls, [conepos], True])
        [self.seen_cones.append((sc[0], (np.average([lat for (lat, long) in sc[1]]), np.average([long for (lat, long) in sc[1]])))) for sc in self.seeing_cones if not sc[2] and len(sc[1]) > 3]
        self.seeing_cones = [sc for sc in self.seeing_cones if sc[2]]
        self.seeing_cones = [[cls, t, False] for (cls, t, _) in self.seeing_cones]
    def get_map(self) -> (gps_util.meter_pos, [gps_util.meter_pos], [gps_util.meter_pos]):
        # return (car position, map), as list of all cones in meterpos
        return self.cp, [conepos for (cls, conepos) in self.seen_cones if cls == 0], [conepos for (cls, conepos) in self.seen_cones if cls == 1]


def test_Slam():
    sensordatadict = read_csv("merged_rundata_csv/alldata_2022_12_17-14_43_59_id3.csv")
    gps_lat_x = sensordatadict["GNSS_latitude_UsbFlRec"+x_ending]
    gps_lat_y = sensordatadict["GNSS_latitude_UsbFlRec"]*np.pi/180
    gps_long_x = sensordatadict["GNSS_longitude_UsbFlRec"+x_ending]
    gps_long_y = sensordatadict["GNSS_longitude_UsbFlRec"]*np.pi/180
    gps_heading_x = sensordatadict["GNSS_heading_UsbFlRec"+x_ending]
    gps_heading_y = sensordatadict["GNSS_heading_UsbFlRec"]*np.pi/180
    start_of_moving_index_gps = 0
    while gps_lat_x[start_of_moving_index_gps] < 120:
        start_of_moving_index_gps += 1
    print("start_of_moving_index_gps =", start_of_moving_index_gps)

    #plot_cones(car_pos=[gps_util.degree_to_radiants(cp) for cp in carpos], yellow_cones=gps_pos_optimised[poii_yelowcones_range[0]:], blue_cones=gps_pos_optimised[9:47], name="true_gps_pos", blue_cones_labes=[str(i) for i in range(9, 47)], yellow_cones_labes=[str(i) for i in range(47, 88)])
    #custom_pnp_find_parameters()
    poi_true_gpspos, carpos = true_pos_from_droneimg_pxpos()  # true_pos_from_droneimg_pxpos("C:/Users/Idefix/PycharmProjects/datasets/keypoints/droneview_annotations.csv")
    # carpos[frnr] = [np.array([lat, long]), np.array([lat, long]), np.array([lat, long]), np.array([lat, long])

    tdiff_dronesensor = 25.75  # framenumber_of_droneview3/25+tdiff_dronesensor = time of gps mesurment
    t_carpos = [(frnr/25+tdiff_dronesensor, carpos[frnr]) for frnr in carpos.keys()]
    t_carpos.sort(key=lambda x: x[0])

    base_gpspos = (gps_lat_y[0], gps_long_y[0])
    tdiff_caml3sensor = 56.8
    all_blue_cones = []
    all_yellow_cones = []
    print("gps_lat_x = ", min(gps_lat_x), ", ", max(gps_lat_x))
    print("gps_long_x = ", min(gps_long_x), ", ", max(gps_long_x))
    print(f"visual_pipeline from (framenr, time) (1282, {1282/20+tdiff_caml3sensor}) to (2643, {2643/20+tdiff_caml3sensor})")
    car_pos = []

    use_gpspos = False
    noslam = trackingSLAM()
    for frnr in range(1282, 2643, 5):  # frnr of camL3
        t = frnr/20+tdiff_caml3sensor
        if use_gpspos:
            t_long, i_long = get_at_time(x=gps_long_x, y=gps_long_y, t=t)
            t_lat = gps_lat_y[i_long]
            t_head = gps_heading_y[i_long]
            car_gps_pos = (t_lat, t_long)
        else:
            carposes, _ = get_at_time(x=[t for (t, car_gps_pos) in t_carpos], y=[np.array(car_gps_pos) for (t, car_gps_pos) in t_carpos], t=t)
            car_gps_pos = carposes[3]
            t_head = gps_util.carposs_to_heading(carposes)
        car_mpos = gps_util.gps_to_meter(car_gps_pos, base_gpspos)
        #car_mpos += gps_util.distazimuth_to_meter(1, car_bearing)  # TODOconvert base_gpspos from rear right wheel to center of car
        blue_cones_meterpos, yellow_cones_meterpos = visual_pipeline(cam="camL3", framenr=frnr, car_pos=car_mpos, car_bearing=t_head)

        noslam.measurment(blue_cones_meterpos, yellow_cones_meterpos, car_mpos)

        car_pos.append(car_mpos)
        all_blue_cones += [bc for bc in blue_cones_meterpos if gps_util.meter_meter_to_dist(bc, car_mpos) < 10]
        all_yellow_cones += [yc for yc in yellow_cones_meterpos if gps_util.meter_meter_to_dist(yc, car_mpos) < 10]

    #plot cones
    true_cone_meterpos = [gps_util.gps_to_meter(conepos, base_gpspos) for conepos in poi_true_gpspos[poii_bluecones_range[0]:]]
    cp, detected_blue_cones, detected_yellow_cones = noslam.get_map()
    plot_cones(car_pos=car_pos, blue_cones=detected_blue_cones, yellow_cones=detected_yellow_cones, name="cone_position from visual_pipeline_droneparams tracking filter", true_conepos=true_cone_meterpos, save_dir="trackingSLAM5.png")
    plot_cones(car_pos=car_pos, blue_cones=all_blue_cones, yellow_cones=all_yellow_cones, name="cone_position from visual_pipeline_droneparams only detections < 10 meter from car", true_conepos=true_cone_meterpos, save_dir="noSLAM5.png")

    # get average distance between slam-result cones and true pos cones, and number of non-detected cones, number of color misclassification, number of detected cones that doesnt exist.
    falsenegatives = 0  # number of true cones that were not detected
    falsepositives = 0  # number of detected cones that were not true
    truepositives = 0
    tot_dist = 0
    for (true_pos, detected_pos) in [(poi_true_gpspos[poii_bluecones_range[0]:poii_bluecones_range[1]], detected_blue_cones), (poi_true_gpspos[poii_yellowcones_range[0]:poii_yellowcones_range[1]], detected_yellow_cones)]:
        tmp = [1 for _ in range(len(detected_pos))]
        for pos in true_pos:
            # get nearest detected blue cone
            nd_bc = None
            mindist = 1
            for (i, bc) in enumerate(detected_pos):
                dist = gps_util.meter_meter_to_dist(gps_util.gps_to_meter(pos, gps_base=base_gpspos), bc)
                if dist < mindist:
                    nd_bc = i
                    mindist = dist
            if nd_bc is None:
                falsenegatives += 1
            else:
                tot_dist += mindist
                truepositives += 1
                tmp[nd_bc] = 0
        falsepositives += sum(tmp)
        print(f"len(detected_pos) = {len(detected_pos)}, len(true_pos) = {len(true_pos)}")
        print(f"truepositives = {truepositives}, falsenegatives = {falsenegatives}, falsepositives = {falsepositives}, tot_dist = {tot_dist}")
    print(f"nonSlam5 & {tot_dist/truepositives} & {truepositives} & {falsenegatives} & {falsepositives} & ? \\\\")


def get_true_carstate():
    sdd = read_csv("merged_rundata_csv/alldata_2022_12_17-14_43_59_id3.csv")
    #relevant_keys = ["BMS_SOC_UsbFlRec", "Converter_L_N_actual_UsbFlRec", "Converter_R_N_actual_UsbFlRec",
    #"Converter_L_RPM_Actual_Filtered_UsbFlRec", "Converter_R_RPM_Actual_Filtered_UsbFlRec",
    #"Converter_L_Torque_Out_UsbFlRec", "Converter_R_Torque_Out_UsbFlRec",
    # "ECU_ACC_X_UsbFlRec", "ECU_ACC_Y_UsbFlRec", "ECU_ACC_Z_UsbFlRec",
    # "GNSS_heading_UsbFlRec", "GNSS_latitude_UsbFlRec", "GNSS_longitude_UsbFlRec", "GNSS_speed_over_ground_UsbFlRec",
    # "SWS_angle_UsbFlRec"]
    vabs_gnss_time = sdd["GNSS_speed_over_ground_UsbFlRec"+x_ending]
    vabs_gnss_value = sdd["GNSS_speed_over_ground_UsbFlRec"]
    heading_gnss_time = sdd["GNSS_heading_UsbFlRec"+x_ending]
    heading_gnss_value = sdd["GNSS_heading_UsbFlRec"]*np.pi/180
    ax_imu_time = sdd["ECU_ACC_X_UsbFlRec"+x_ending]
    ax_imu_value = sdd["ECU_ACC_X_UsbFlRec"]
    ay_imu_time = sdd["ECU_ACC_Y_UsbFlRec"+x_ending]
    ay_imu_value = sdd["ECU_ACC_Y_UsbFlRec"]
    az_imu_time = sdd["ECU_ACC_Z_UsbFlRec"+x_ending]
    az_imu_value = sdd["ECU_ACC_Z_UsbFlRec"]

    poi_true_gps_positions_radiants, carpos = true_pos_from_droneimg_pxpos()

    # get absolute velocity from droneview carpositions
    carposkeys = list(carpos.keys())
    carposkeys.sort()
    abs_v_droneview_time = [0.0 for _ in range(len(carposkeys))]
    abs_v_droneview_value = [0.0 for _ in range(len(carposkeys))]
    abs_v_droneview_time[0] = t2ssdt(drone2t(carposkeys[0]))-0.001
    for i in range(1, len(carposkeys), 1):
        t0 = t2ssdt(drone2t(carposkeys[i-1]))
        t1 = t2ssdt(drone2t(carposkeys[i]))
        abs_v_droneview_time[i] = 0.5*t0+0.5*t1
        abs_v_droneview_value[i] = gps_util.gps_to_dist(gps_util.average(carpos[carposkeys[i-1]]), gps_util.average(carpos[carposkeys[i]]))/(t1-t0)
    print("abs_v_droneview_time =", abs_v_droneview_time[:10])
    print("abs_v_droneview_value =", abs_v_droneview_value[:10])
    vabs_time, sincd_vabs_droneview_value, sincd_vabs_gps_value = timesinc(abs_v_droneview_time, abs_v_droneview_value, vabs_gnss_time, vabs_gnss_value)
    plot_and_save("vabs from gps and drone", x_in=vabs_time, ys=[smothing(vabs_time, sincd_vabs_droneview_value, 0.5), sincd_vabs_gps_value], names=["droneview", "gps"])

    # get heading from droneview carpositions
    heading_droneview_time = [0.0 for _ in range(len(carposkeys))]
    heading_droneview_value = [0.0 for _ in range(len(carposkeys))]
    for i in range(len(carposkeys)):
        heading_droneview_time[i] = t2ssdt(drone2t(carposkeys[i]))
        heading_droneview_value[i] = gps_util.carposs_to_heading(carpos[carposkeys[i]])
    heading_time, sincd_heading_droneview_value, sincd_heading_gps_value = timesinc(heading_droneview_time, heading_droneview_value, heading_gnss_time, heading_gnss_value)
    plot_and_save("heading from gps and drone", x_in=heading_time, ys=[sincd_heading_droneview_value, np.array([to_range(x) for x in sincd_heading_gps_value])], names=["droneview", "gps"], avgs=False)

    # derivitate of zip(gnss_vabs_time, gnss_vabs_value)
    smothed_gpsv = smothing(time=vabs_gnss_time, values=vabs_gnss_value, t=1)
    ax_time = np.array([(0.5*vabs_gnss_time[i+1]+0.5*vabs_gnss_time[i]) for i in range(len(smothed_gpsv)-1)])
    ax_gnss_value = [(vabs_gnss_value[i+1]-vabs_gnss_value[i])/(vabs_gnss_time[i+1]-vabs_gnss_time[i]) for i in range(len(vabs_gnss_value)-1)]
    imu_ax_value_meadian = np.median(ax_imu_value)
    imu_ay_value_meadian = np.median(ay_imu_value)
    imu_az_value_meadian = np.median(az_imu_value)
    imu_aabs_value = [np.sqrt((ax_imu_value[i]-imu_ax_value_meadian)**2+(ay_imu_value[i]-imu_ay_value_meadian)**2+(az_imu_value[i]-imu_az_value_meadian)**2) for i in range(len(ax_imu_time))]
    imu_aabs_value = smothing(ax_imu_time, imu_aabs_value, 3)
    syncd_ax_time, synced_imu_aabs_vale, synced_aabs_gnss_value = timesinc(ax_imu_time, imu_aabs_value, ax_time, ax_gnss_value)
    plot_and_save("aabs from gnss_vabs", x_in=ax_time, ys=[smothing(ax_time, ax_gnss_value, 3), synced_imu_aabs_vale], names=["aabs_gnss_value", "aabs_imu_value"], avgs=False)

    # derivitate of zip(gnss_heading_time, gnss_heading_value)
    smothed_gps_heading = smothing(time=heading_gnss_time, values=heading_gnss_value, t=1)
    yawrate_time = [(0.5*heading_gnss_time[i+1]+0.5*heading_gnss_time[i]) for i in range(len(smothed_gps_heading)-1)]
    yawrate_gps_value = [(smothed_gps_heading[i+1]-smothed_gps_heading[i])/(heading_gnss_time[i+1]-heading_gnss_time[i]) for i in range(len(heading_gnss_time)-1)]
    smothed_dv_heading = smothing(heading_droneview_time, heading_droneview_value, t=1)
    yawrate_dv_value = [(smothed_dv_heading[i+1]-smothed_dv_heading[i])/(heading_droneview_time[i+1]-heading_droneview_time[i]) for i in range(len(heading_droneview_time)-1)]
    plot_and_save("yawrate", x_in=yawrate_time, ys=[yawrate_gps_value, yawrate_dv_value], names=["yawrate_value", "yawrate_dv_value"], avgs=False)

    true_v_value = smothing(vabs_gnss_time, vabs_gnss_value, 0.5)
    true_v_time = vabs_gnss_time
    true_ax_value = smothing(ax_time, ax_gnss_value, 3)
    true_ax_time = ax_time
    for k in ["Converter_L_Torque_Out_UsbFlRec", "Converter_R_Torque_Out_UsbFlRec"]:
        k_name = k_to_name(k)
        print(f"\nname = {k_name}\n")

        k_time = sdd[k+x_ending]
        time, syncd_true_v_value, kv = timesinc(true_v_time, true_v_value, k_time, smothing(k_time, sdd[k], 5))
        offset = syncd_true_v_value[0]-kv[0]
        fac = np.sum(syncd_true_v_value-offset)/np.sum(kv)
        #fun, parameters = fit_poly_fun_and_print(fac*kv+offset, tv, f"{k} to v", exponents=[0, 1])
        fig, axe = plt.subplots()
        axe.set_title(f"fit linear from {k_name} to vx")
        axe.plot(time, syncd_true_v_value, label="true_vabs")
        #axe.plot(time, np.array([fun(x, parameters) for x in kv]), label="fun(kv)")
        axe.plot(time, fac*kv+offset, label=f"{fac:.2E}*{k_name}+{offset:.2E}")
        axe.legend()
        axe.grid()
        fig.show()

        time, syncd_true_ax_value, ka = timesinc(true_ax_time, true_ax_value, sdd[k+x_ending], smothing(k_time, sdd[k], 5))
        offset = syncd_true_ax_value[0]-ka[0]
        fac = np.sum(syncd_true_ax_value-offset)/np.sum(ka)
        #fun, parameters = fit_poly_fun_and_print(ka, ta, f"{k} to ax", exponents=[0, 1])
        fig, axe = plt.subplots()
        axe.set_title(f"fit linear from {k_name} to ax")
        axe.plot(time, syncd_true_ax_value, label="true_aabs")
        axe.plot(time, fac*ka+offset, label=f"{fac:.2E}*{k_name}+{offset:.2E}")
        #axe.plot(time, np.array([fun(x, parameters) for x in ka]), label="fun(ka)")
        axe.legend()
        axe.grid()
        fig.show()


def main():
    get_true_carstate()


if __name__ == "__main__":
    # TODO write framenumbers into merged_rundata_csv/alldata_....csv
    main()


def all_functions():
    # just a list of all functions. should never be called.
    exit(1)
    # namespace util:
    assert_SensorDataDict_is_valid()
    abs_value()
    remove_zeros()
    to_range()
    get_at_time()
    timesinc()
    fit_poly_fun_and_print()
    avg_pxprom_from_conekeypoints()
    timestr2datetime()
    get_path()
    get_nearst_gpspoint()
    # namespace io:
    read_mat()
    merge_matsdata()
    read_mats()
    read_csv()
    write_csv()
    get_boundingboxes()
    get_cone_keypoints()
    get_poii_positions()
    # namespace plot:
    plot_colorgradientline()
    plot_and_save()
    visualise_data()
    show_sensorlogs()
    plot_from_pos_and_sensor()
    plot_cones()
    plot_on_googlemaps()
    # manuel
    get_laptoptimes_for_camFrames()
    get_car_moves_starttime_from_sensors()
    averegae_diff()
    true_pos_from_droneimg_pxpos()
    custom_pnp_find_parameters()
    custom_pnp_validate()
    print_synced_frame()
    get_synced_frame()
    print_synced_pos_bearing_from_drone_and_sensors()
    # testing for cpp implementation
    custom_PnP()
    visual_pipeline()
    test_Slam()
    main()

# 16bit_int.MAX_VALUE = 32767
# U_I_converter_max = 42.46
# motorkonstante = 0.83
# Trq = Converter_L_Torque_Out_UsbFlRec*U_I_converter_max/(16bit_int.MAX_VALUE*np.sqrt(2))*motorkonstante
# RPM = Converter_L_RPM_Actual_Filtered_UsbFlRec*6000/32767