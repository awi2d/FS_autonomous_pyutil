import os
import pathlib
import sys

import cv2
import matplotlib.collections
import numpy as np
import mat73
import datetime
import matplotlib.pyplot as plt
import scipy.optimize

from plot_on_googlemaps import CustomGoogleMapPlotter
from util import getType

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
# mat_files_dir = [pathlib.Path(p) for p in ["testrun_0/", "testrun_1/", "testrun_2/", "testrun_3/", "testrun_6/", "testrun_7/", "testrun_8/", "testrun_unknown/"]]  # all dirs that contain .mat files
mat_files_dir = [pathlib.Path(p) for p in ["testrun_13_30/", "testrun_14_21/",
                                           "testrun_14_41/"]]  # only dirs that contain .mat files for runs on the 2022-12-17 (yyyy-mm-dd)
csv_files_dir = pathlib.Path("merged_rundata_csv/")
# csv_files = [pathlib.Path(f) for f in os.listdir(csv_files_dir) if str(f).endswith(".csv")]
csv_files = [pathlib.Path(f) for f in os.listdir(csv_files_dir) if str(f).endswith(".csv") and str(f).split("-")[
    0] == "alldata_2022_12_17"]  # only return csv files that were recorded at the 2022-12-17 (yyyy-mm-dd)
cam_footage_dir = pathlib.Path("cam_footage")
cam_sync_files = [cam_footage_dir / pathlib.Path(f) for f in
                  ["sync_info_camL0.txt", "sync_info_camR0.txt", "sync_info_camL3.txt", "sync_info_camR3.txt",
                   "sync_info_drone0.txt", "sync_info_drone3.txt"]]  # , "sync_info_drone1.txt"
#</constants
#<dataTypes>

#</dataTypes
# SensorDataDict should contain the keys startTimestamp_key: datetime.datetime, length_key:float, key: np.array, key+x_ending: np.array for key in relevant_keys and optionaly cam_keys and cam_keys+x_ending
SensorDataDict = dict
def assert_SensorDataDict_is_valid(sdd: SensorDataDict) -> None:
    keys = list(relevant_keys)
    if any([k in sdd.keys() for k in cam_keys]):
        keys += cam_keys
    keysp = keys + [k + x_ending for k in keys] + [startTimestamp_key, length_key]
    if not set(keysp) == set([str(k) for k in sdd.keys()]):
        print("sdd.keys() = ", sdd.keys())
        print("\n keys+ = ", keysp)
        print("\nkeys that should be in sdd.keys() but arent: ", [k for k in keysp if k not in sdd.keys()])
        print("keys that should not be in sdd.keys() but are: ", [k for k in sdd.keys() if k not in keysp])
        assert False
    for key in keys:
        assert len(sdd[key]) == len(sdd[key + x_ending])
degree_lattitude = float  # how for north from the equator, e.g. 51.46411476026706
degree_longitude = float  # how far east from greenwitch, e.g. 6.738945631812034
gps_pos = (degree_lattitude, degree_longitude)  # (lattitude, longitude), e.g. (51.46411476026706, 6.738945631812034)
meter_north = float  # distance between two points in north/south direction, in meters
meter_east = float  # distance between two points in east/west direction, in meters
meter = float
normalised_px_h = float  # in range(0, 1) height position of pixel in image / height of image
normalised_px_w = float  # in range(0, 1) width position of pixel in image / width of image
#meter_vector = (meter_north, meter_east)  # difference between two points on globe in meters, with (distance to North, distance to east)
seconds = float  # seconds since start of mesurment
plot = None
bearing_radiant = float  # in range(-pi, pi) 0 = south, -pi/2 = east, pi/2 = west, pi = north, -pi = north
#</dataTypes>


#<utility methods>
def gps2meter(lats: [degree_lattitude], longs: [degree_longitude], lat_base: degree_lattitude, long_base: degree_longitude) -> ([meter_north], [meter_east]):
    # return meter vector from (lat_base, long_base) to (lats[i], longs[i]), assuming both are gps coordinate (langitude, lattitude)
    # if lats and longs or no list but singe elements, the output will also be.
    northm_pos = (longs-long_base)*(111320*np.cos(lat_base*np.pi/180))
    eastm_pos = (lats-lat_base)*111320
    return northm_pos, eastm_pos


def abs_value(l):
    return np.sqrt(np.sum([t**2 for t in l]))


def abs_meter_diff(p0: (degree_lattitude, degree_longitude), p1: (degree_lattitude, degree_longitude)) -> meter:
    # returns the distance in meter between two gps points
    northm_pos = (p0[1]-p1[1])*(111320*np.cos(p1[0]*np.pi/180))
    eastm_pos = (p0[0]-p1[0])*111320
    return np.sqrt(northm_pos**2+eastm_pos**2)


def remove_zeros(x, y, zero=0):
    # remove elements from x and y if the element of y is zero
    assert len(x) == len(y)
    tmp = [(a, b) for (a, b) in zip(x, y) if b != zero]
    x = np.array([a for (a, b) in tmp])
    y = np.array([b for (a, b) in tmp])
    return x, y


def bearing_from_gps_points(a: (degree_lattitude, degree_longitude), b: (degree_lattitude, degree_longitude)) -> bearing_radiant:
    # a = (lat, long), b = (lat, long)
    # returns the angle between north direction and (b-a) direction in range [-pi, pi].
    # with
    #   north = 3.115177539082554 ~= pi or -pi
    #   east = -1.560228347472096 ~= -pi/2
    #   west = 1.665346370084264 ~= pi/2
    #   south = 0.008986389136445639 ~= 0
    dL = b[1]-a[1]
    X = np.cos(b[0]) * np.sin(dL)
    Y = np.cos(a[0]) * np.sin(b[0]) - np.sin(a[0])*np.cos(b[0])*np.cos(dL)
    return np.arctan2(X, Y)


def get_at_time(x: [seconds], y: [float], t: seconds) -> float:
    # returns the linear interpolated value of y at time t.
    # if t in x: return y[i], so that x[i]==t
    assert len(x) == len(y)
    if t < x[0] or t > x[-1]:
        print(f"warining: get_at_time: t is out of range ({x[0]}, {x[-1]}) with {t}")
        if t < x[0]:
            return y[0]
        if t > x[-1]:
            return y[-1]
    for i in range(0, len(x)):
        if t == x[i]:
            return y[i]
        if t < x[i]:
            w0 = abs(t-x[i-1])
            w1 = abs(x[i]-t)
            return (w1*y[i-1]+w0*y[i])/(w0+w1)


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


def fit_linear(in_x, out_true):
    # returns fun, parameters, so that fun(in_x, parameters) = out_true
    assert len(in_x) == len(out_true)
    in_x = np.array(in_x)
    out_true = np.array(out_true)
    fo = np.array([1, 0, 0, 0])

    def fun(x, t):
        return t[0]*x + t[1]*x**2 + t[2]/x + t[3]
    def loss(t):
        return np.sum((fun(in_x, t) - out_true) ** 2)
    res = scipy.optimize.minimize(loss, fo)
    if res.success:
        return fun, res.x
    else:
        print("failed_res = ", res)
        if res.x is not None:
            print("warining: fit_linear failed")
            return fun, res.x
        raise Exception("could not succesfully fit data from", in_x, "to", out_true)


def avg_pxprom_from_conekeypoints(keypoints:[(normalised_px_w, normalised_px_h)], bounding_box: (int, normalised_px_w, normalised_px_h, normalised_px_w, normalised_px_h)) -> float:
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
    obj_Distm = {(0, 1): 0.09931057723962547, (0, 2): 0.18581319480106973, (0, 3): 0.2893731266180254, (1, 2): 0.08783333333333333, (1, 3): 0.18816666666666668, (2, 3): 0.1, (1, 5): 0.126375, (1, 6): 0.20548663807186746, (2, 6): 0.156, (1, 4): 0.058826410187308546, (2, 5): 0.08119296009798978, (3, 6): 0.1159014116265954}

    assert len(keypoints) == 7
    cls, posw, posh, sizew, sizeh = bounding_box
    imgsize_h, imgsize_w = (1200, 1920)
    #keypoints_pxpos_in_camImage_from_relativepxpos_in_coneimg = np.array([((posw-0.5*sizew+w*sizew)*imgsize_w, (posh-0.5*sizeh+h*sizeh)*imgsize_h) for (w, h) in keypoints])
    keypoints = np.array([(w*sizew*imgsize_w, h*sizeh*imgsize_h) for (w, h) in keypoints])  # transform keypoints from relative position in coneimage to relative position in camera image (as only differences between keypoints are observed, the offset can be ignored)
    avg = 0
    indexe = [(i, j) for i in range(6) for j in range(i+1, 7)]
    print("\n")
    for (i, j) in indexe:
        # avg += meter dist between points on real cone / pixel dist between points on cone image
        (i, j) = (min(i, j), max(i, j))
        if i == 0 and j > 3:
            (i, j) = (i, j-3)
        if i > 3 and j > 3:
            (i, j) = (i-3, j-3)
        if (i, j) == (2, 4):
            (i, j) = (1, 5)
        if (i, j) == (3, 4):
            (i, j) = (1, 6)
        if (i, j) == (3, 5):
            (i, j) = (2, 6)
        avg += obj_Distm[(i, j)]/abs_value(keypoints[i]-keypoints[j])
    avg /= len(indexe)
    # distance to object [m] = real object size(m) * focal length (mm) / object height in frame (mm)
    #  with object height in frame (mm) = sensor height (mm) * object height (px) / sensor height (px)
    # distance to object [m] = real object size(m)/ object height (px) * constant
    #  with constant = focal length (mm) * sensor height (px) / sensor height (mm)
    # this returns average of
    #  real object size(m) / (object size (px) / sensor height (px))
    return avg


#</utility methods>


def plot_colorgradientline(name: str, lat_pos: [degree_lattitude], long_pos: [degree_longitude], time: [seconds] = None) -> plot:
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
    fixpoitns_gpspos, blue_cone_gpspos, yellow_cone_gpspos, carpos, bearing = true_pos_from_droneimg_pxpos()
    for ycp in yellow_cone_gpspos:
        ax0.scatter(ycp[1], ycp[0], c="yellow")
    for bcp in blue_cone_gpspos:
        ax0.scatter(bcp[1], bcp[0], c="blue")

    # plot carpos (car positions from drone view)
    for k in carpos.keys():
        for i in range(len(carpos[k])):
            ax0.scatter(carpos[k][i][1], carpos[k][i][0], s=5, color=cmap((k/25+27.92)/max(time)), alpha=1)

    # add labels and meter scaling on both axies
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


def plot_and_save(name: str, x: [seconds], y, save_dir=None) -> plot:
    """
    plots y (dependent variable) against x. with 5-second average of y, mean an total average of y and saves the plot to save_dir.
    removes outliers from y and adds name as name of plot and label for y data and label for y-axis
    """
    assert len(x) == len(y)
    print(f"plot_and_save(name = {name}, save_dir={save_dir}): ")
    # if any value in y is more than ?double? the distance to mean than the max of the lower ?90?%, remove them
    mean = sorted(y)[len(y) // 2]
    dist = [abs(t - mean) for t in y]
    dist.sort()
    dist = 2 * dist[int(0.9 * len(dist))]
    print("mean = ", mean, ", dist = ", dist)
    if dist != 0:
        old_length = len(x)
        tmp = [(tx, ty) for (tx, ty) in zip(x, y) if abs(ty - mean) < dist]
        print("removing outliers changed number of data points from ", old_length, " to ", len(tmp), "(",
              100 * len(tmp) / old_length, "% of original)")
        x = [tx for (tx, ty) in tmp]
        y = [ty for (tx, ty) in tmp]
        del tmp
    fig, (axe) = plt.subplots(1)
    axe.set_title(name)
    # plot unfiltered data
    axe.plot(x, y, label=name)
    # plot exponentianal average
    # y_movingaverage = [0]*len(x)
    # y_movingaverage[0] = y[0]
    # for i in range(1, len(x)):
    #    y_movingaverage[i] = 0.99*y_movingaverage[i-1]+0.01*y[i]
    # axe.plot(x, y_movingaverage, color='red', linewidth=72/fig.dpi, label="average exp0.99")
    s = 5
    n = int(s * len(x) / max(x))  # number of timestamps/length in seconds = number of timestamps per seconds, assuming timestamps are evenly spaced.
    # doing average over {s} seconds checking for unevenly spaced timestamps was to time-consuimg to run on this pc.
    padded_y = np.array([y[0]] * (n // 2) + list(y) + [y[-1]] * (n - n // 2 - 1))
    y_avg = np.convolve(padded_y, np.ones(n) / n, mode='valid')
    axe.plot(x, y_avg, "--", color="green", linewidth=0.5, label="average " + str(s) + "s")
    # plot average
    axe.plot(x, [np.average(y)] * len(y), "-.", color="black", linewidth=0.5, label="total average")
    axe.plot(x, [mean] * len(y), "--", color="black", linewidth=0.5, label="mean")

    axe.set_xlabel("time in seconds")
    axe.set_ylabel(name)
    axe.legend()
    axe.grid()
    if save_dir is None:
        save_dir = name
    fig.savefig(save_dir)


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
    data_dict = data_dict[list(data_dict.keys())[0]]  # top-level has only one entry
    # Datetime +Length*minute = Datetaime der nächsten File sollte gelten
    # print("data_dict[Y]", [entry["Raster"] for entry in data_dict["Y"]])
    # print("data_dict[X]", [entry for entry in data_dict["X"]])

    my_dict[startTimestamp_key] = timestr2datetime(data_dict[description_key]["General"]["DateTime"])
    my_dict[length_key] = float(data_dict[description_key]["Measurement"][length_key])
    for (x_entry, entry) in zip(data_dict["X"], data_dict["Y"]):
        # x_entry and entry are np-arrays of seconds_since_starttime, data_of_x_entry["Raster"]_sensor
        assert x_entry["Raster"] == entry["Raster"]
        name = entry["Raster"]
        if name in relevant_keys:
            name_x = name + x_ending
            tmp = [(x, d) for (x, d) in zip(x_entry["Data"], entry["Data"]) if not np.isnan(x) and not np.isnan(d)]
            data = np.array([d for (x, d) in tmp])
            data_x = np.array([x for (x, d) in tmp])
            assert all([data_x[i + 1] >= data_x[i] for i in range(len(data_x) - 1)])  # x should be ascending
            assert len(data) == len(data_x)
            my_dict[name] = data
            my_dict[name_x] = data_x
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
        new_datax = np.concatenate((np.reshape(dict_a[name_x], len(dict_a[name_x])),
                                    np.array([t + dict_a[length_key] for t in dict_b[name_x]])))
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
        files = [path + f for f in os.listdir(path) if str(f).endswith(".mat")]
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
    for cam_name in cam_sync_files:
        st, fps, n_frames, frames_dir = get_laptoptimes_for_camFrames(cam_name)
        cam_footages.append({startTimestamp_key: st, "fps": fps, "n_frames": n_frames, "name": frames_dir})
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
                if str(cam_footage["name"]).split("\\")[-1].startswith("left_cam"):
                    cam_key = "cam_left"
                elif str(cam_footage["name"]).split("\\")[-1].startswith("right_cam"):
                    cam_key = "cam_right"
                else:
                    print("invalid start of onboard camara frames dir: ", cam_footage["name"])
                    exit(1)
                startframe = 0
                if cam_footage[startTimestamp_key] < sensor_data[startTimestamp_key]:
                    dts = (sensor_data[startTimestamp_key] - cam_footage[startTimestamp_key]).total_seconds()
                    startframe = 1 + int(cam_footage["fps"] * dts)
                sensor_data[cam_key + x_ending] = [
                    cam_footage[startTimestamp_key] + datetime.timedelta(seconds=frame_nr / cam_footage["fps"]) for
                    frame_nr in range(startframe, cam_footage["n_frames"])]
                sensor_data[cam_key + x_ending] = [(t - sensor_data[startTimestamp_key]).total_seconds() for t in
                                                   sensor_data[
                                                       cam_key + x_ending]]  # transform from datetime to float seconds since sensor_date[startTimestamp_key
                sensor_data[cam_key] = [cam_footage_dir / str(cam_footage["name"] / f"frame_{i}.jpg") for i in
                                        range(startframe, cam_footage["n_frames"])]
        for key in cam_keys:
            if not key in sensor_data.keys():
                sensor_data[key] = np.array([])
                sensor_data[key + x_ending] = np.array([])
    for sdd in merged_dicts:
        assert_SensorDataDict_is_valid(sdd)
    return merged_dicts


def read_csv(csv_file: os.path) -> SensorDataDict:
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


def visualise_data(data: SensorDataDict):
    assert_SensorDataDict_is_valid(data)
    name = data[startTimestamp_key].strftime(time_format_str)
    print("\nvisualise data ", name)
    keys_where_zero_is_nan = ["BMS_SOC_UsbFlRec", "GNSS_latitude_UsbFlRec", "GNSS_longitude_UsbFlRec"]
    print("empty keys in = ", [key for key in relevant_keys if
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
    # Converter_L_N_actual_UsbFlRec is rotations/sekonds of Left back wheel

    for i, k in enumerate(relevant_keys):
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
                plot_and_save(str(k), x, y, f"vis_out/{name}__{str(k)}.png")

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
    #TODO use b2worw2b
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
            n = str(k).replace("_UsbFlRec", "").replace("Converter_", "").replace("_actual", "")
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
            print(f"\t{n}: car moves from {t_start} to {t_end}")
            if t_start is not None and t_end is not None:
                print(
                    f"\taccording to {n} car is moving from {st + datetime.timedelta(seconds=t_start)} to {st + datetime.timedelta(seconds=t_end)} (length={t_end - t_start})")


def averegae_diff(data: SensorDataDict, k0, k1, nonzero_treshhold=(1, 1), t0=0.0001):
    k0_name = str(k0).replace("_UsbFlRec", "").replace("Converter_", "").replace("_actual", "")
    k1_name = str(k1).replace("_UsbFlRec", "").replace("Converter_", "").replace("_actual", "")
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
    else:
        print(f"no valid elements in {k0_name}/{k1_name}.")
    # k1/k0
    tmp = np.array([t1 / t0 for (t0, t1) in syncd if abs(t0) > nonzero_treshhold[0]])
    k10 = None
    if len(tmp) > 0:
        k10 = np.average(tmp)
    else:
        print(f"no valid elements in {k1_name}/{k0_name}.")
    # print(f"averegae_diff {k0}/{k1}: = {k01}\n{k1}/{k0} = {k10}")
    # fin
    if k01 is not None and k10 is not None:
        while abs(k01 - 1 / k10) > t0 or abs(k10 - 1 / k01) > t0:
            k01, k10 = k01 * 0.5 + 0.5 / k10, k10 * 0.5 + 0.5 / k01
        print(f"averegae_diff: \n# {k0_name} / {k1_name} = {k01}\n# {k1_name} / {k0_name} = {k10}")


def true_pos_from_droneimg_pxpos(point_file=None):
    gully_u_gmpos = (51.4641141, 6.7389449)
    gully_o_gmpos = (51.4644476, 6.7388095)
    well_gmpos = (51.4643793, 6.7390844)
    dach_u_gmpos = (51.4644667, 6.7389267)
    dach_o_gmpos = (51.46446872061805, 6.7389091136932295)
    gps_pos_optimised = []
    bearing = {}  # franenumber -> bearing
    carpos = {}  # framenumber -> [(gps_lat_frontlefttire, gps_long_frontlefttire), ... frontright, rearleft, rearright]
    if point_file is None:
        # use stored values instead of optimising from stratch
        gps_pos_optimised = [(51.46411414940673,6.738944977588712), (51.46444738631447,6.738809863549967), (51.464376920188684,6.739083154724557), (51.46446718888097,6.738927768062857), (51.46447029093418,6.738909215184113), (51.46433948081566,6.738858198040306), (51.464340634576544,6.738867556464109), (51.464332301312695,6.738860518358256), (51.46433352256563,6.7388693023431845), (51.46425560352164,6.738978243689988), (51.46427108152239,6.738956349310485), (51.46430420716902,6.738933864796242), (51.46434835826103,6.738916791956062), (51.46439114717442,6.738898908652391), (51.464435010009225,6.738881909437699), (51.46447785743073,6.738871411732899), (51.46450255330567,6.738853593635997), (51.464523350787786,6.738819787999573), (51.464533504173595,6.738776268215176), (51.46453037892502,6.738730218636211), (51.464517197636866,6.7386905132803125), (51.46449659091481,6.738663015861728), (51.46446663771645,6.738648698012101), (51.464438661684675,6.738653402331532), (51.46441130495734,6.738677405727721), (51.46439626718731,6.73871541407409), (51.464389944382496,6.7387554556266025), (51.46438436411264,6.738785735746747), (51.46437249221701,6.738802412049573), (51.46433474605532,6.738820935341453), (51.46429112213464,6.738838149987502), (51.46425084032891,6.738857744036035), (51.46420852749894,6.73887419245192), (51.46416856291396,6.738895720484548), (51.46412900038989,6.738912834604563), (51.46410496482645,6.738923655505124), (51.464079595253736,6.738958957232722), (51.464067119195114,6.738998752044335), (51.464066961660166,6.739049464522678), (51.46407733183846,6.739086333417893), (51.46410152721491,6.739123619222857), (51.46413179871855,6.739137612689789), (51.46415989439572,6.7391333201885475), (51.46419621559581,6.739098942910411), (51.464209958613,6.739064370135348), (51.464218071281536,6.739045220236359), (51.464230863835915,6.739022540553707), (51.464235195017565,6.73894986483897), (51.46425254304922,6.738924631810385), (51.464298461923505,6.738896452978141), (51.4643428654891,6.7388789814292425), (51.46438668534625,6.738860624573461), (51.46442779156127,6.7388451840515105), (51.46447030037779,6.7388281064257844), (51.46448691323747,6.738816478338111), (51.46449890894766,6.738795981253502), (51.46450483971843,6.738769736369862), (51.4645028126984,6.738742394083106), (51.464494201192664,6.738718220083149), (51.464480374977576,6.738700955868589), (51.46446375293953,6.738694369123524), (51.464446342496,6.7386975953166015), (51.46443094576782,6.738710948324009), (51.46442003090998,6.738732241930608), (51.46441452623774,6.738758460709004), (51.4644094765204,6.738799210265234), (51.464398695574275,6.738827028726462), (51.464380796405976,6.738840788713605), (51.4643624111585,6.738848593652515), (51.4643380159601,6.738858309555241), (51.46429508893946,6.738875725158482), (51.46425547308892,6.738894756649162), (51.464215777303274,6.738910776637766), (51.46417554371516,6.7389327114558935), (51.46413319131306,6.738958079522211), (51.46411580753114,6.73896635691615), (51.46410299161643,6.738986446365815), (51.46409511800807,6.7390122939678285), (51.46409565819885,6.739041007544318), (51.46410386007197,6.739066872605536), (51.4641179539995,6.739085351021696), (51.46413588370013,6.739092211889237), (51.464154482508064,6.739087866698141), (51.464172734232235,6.739071377317364), (51.464183024539835,6.739047157912321), (51.46419395425206,6.739020047700429), (51.46420877083953,6.738992929967574), (51.46422267511233,6.738971629815222)]
        bearing = {}  # franenumber -> bearing
        carpos = {}  # framenumber -> [(gps_lat_frontlefttire, gps_long_frontlefttire), ... frontright, rearleft, rearright]

        bearing[3277] = 2.968961427232848
        carpos[3277] = [np.array([51.46436552,  6.73881859]), np.array([51.46437026,  6.73883536]), np.array([51.46435247,  6.73882542]), np.array([51.46435753,  6.73884089])]
        bearing[3278] = 2.9602779464288203
        carpos[3278] = [np.array([51.46436772,  6.73881634]), np.array([51.46437299,  6.73883346]), np.array([51.4643548 ,  6.73882423]), np.array([51.4643592 ,  6.73883904])]
        bearing[3279] = 2.946124835315508
        carpos[3279] = [np.array([51.46436821,  6.73881693]), np.array([51.46437184,  6.73883334]), np.array([51.46435367,  6.73882403]), np.array([51.46435905,  6.73884112])]
        bearing[3280] = 2.9530146109797033
        carpos[3280] = [np.array([51.46436834,  6.73881453]), np.array([51.46437313,  6.73883309]), np.array([51.4643551 ,  6.73882314]), np.array([51.46435889,  6.7388389 ])]
        bearing[3281] = 2.946952586292381
        carpos[3281] = [np.array([51.46436901,  6.73881662]), np.array([51.46437424,  6.73883318]), np.array([51.46435623,  6.73882438]), np.array([51.46436026,  6.73883994])]

        bearing[3285] = 2.9237076768264165
        carpos[3285] = [np.array([51.46437197,  6.73881453]), np.array([51.4643759,  6.7388296]), np.array([51.46435852,  6.73882217]), np.array([51.46436241,  6.73883838])]
        bearing[3286] = 2.8897816229067455
        carpos[3286] = [np.array([51.46437782,  6.73881437]), np.array([51.46438212,  6.73883028]), np.array([51.46436522,  6.73882389]), np.array([51.46436875,  6.73883914])]
        bearing[3287] = 2.89842527838584
        carpos[3287] = [np.array([51.46437393,  6.73881364]), np.array([51.46437597,  6.73883027]), np.array([51.46436096,  6.73882209]), np.array([51.46436508,  6.7388381 ])]
        bearing[3288] = 2.9173079070782926
        carpos[3288] = [np.array([51.46437669,  6.73881493]), np.array([51.46438063,  6.73883012]), np.array([51.46436379,  6.73882252]), np.array([51.4643668 ,  6.73883932])]
        bearing[3289] = 2.893820474703203
        carpos[3289] = [np.array([51.46437515,  6.73881413]), np.array([51.46437816,  6.73882848]), np.array([51.46436231,  6.73882185]), np.array([51.4643669 ,  6.73883753])]

        bearing[3285] = 2.8929463838029723
        carpos[3285] = [np.array([51.46437244,  6.73881409]), np.array([51.46437574,  6.73882946]), np.array([51.46435836,  6.73882319]), np.array([51.46436442,  6.73883811])]
        bearing[3286] = 2.877122618598874
        carpos[3286] = [np.array([51.4643726,  6.73881193]), np.array([51.46437625,  6.73882883]), np.array([51.46435912,  6.73882235]), np.array([51.46436479,  6.73883699])]
        bearing[3287] = 2.95943247322537
        carpos[3287] = [np.array([51.46437705,  6.73881457]), np.array([51.46438101,  6.73883046]), np.array([51.46436374,  6.73882187]), np.array([51.46436778,  6.73883662])]
        bearing[3288] = 2.882140457199899
        carpos[3288] = [np.array([51.46437303,  6.73881073]), np.array([51.46437765,  6.73882886]), np.array([51.46436074,  6.73882053]), np.array([51.46436556,  6.73883686])]
        bearing[3289] = 2.9264352848229445
        carpos[3289] = [np.array([51.46437642,  6.73881273]), np.array([51.46437973,  6.73882902]), np.array([51.46436385,  6.73882139]), np.array([51.46436701,  6.73883557])]

        bearing[4027] = -0.5219724392318666
        carpos[4027] = [np.array([51.46424249,  6.7389868 ]), np.array([51.4642347 ,  6.73897393]), np.array([51.46425172,  6.73897167]), np.array([51.46424504,  6.73895807])]
        bearing[4028] = -0.5540929271133909
        carpos[4028] = [np.array([51.46424116,  6.73898918]), np.array([51.46423405,  6.73897795]), np.array([51.4642514,  6.7389721]), np.array([51.46424305,  6.73896227])]
        bearing[4029] = -0.46673446572009253
        carpos[4029] = [np.array([51.46424119,  6.738988  ]), np.array([51.46423427,  6.73897845]), np.array([51.46425317,  6.73897375]), np.array([51.46424365,  6.7389631 ])]
        bearing[4030] = -0.5110170587960822
        carpos[4030] = [np.array([51.46424182,  6.73899115]), np.array([51.46423382,  6.73897752]), np.array([51.46425259,  6.73897467]), np.array([51.46424362,  6.73896227])]
        bearing[4031] = -0.47730922186927155
        carpos[4031] = [np.array([51.46423708,  6.73899018]), np.array([51.46423026,  6.73897946]), np.array([51.46424827,  6.73897513]), np.array([51.46424064,  6.73896382])]
    else:
        # calculate positions from data
        # fixed points, where coordinates in google maps and pixel frame are known
        def loss_factory(a_pxpos, b_pxpos, c_pxpos, d_pxpos, e_pxpos, a_gmpos=gully_u_gmpos, b_gmpos=gully_o_gmpos, c_gmpos=well_gmpos, d_gmpos=dach_u_gmpos, e_gmpos=dach_o_gmpos):
            def loss(t):
                t_mat = np.array([[t[0], t[1]], [t[2], t[3]]])
                offset = np.array([t[4], t[5]])
                return np.sum((np.matmul(a_pxpos, t_mat)+offset - a_gmpos) ** 2) + \
                       np.sum((np.matmul(b_pxpos, t_mat)+offset - b_gmpos) ** 2) + \
                       np.sum((np.matmul(c_pxpos, t_mat)+offset - c_gmpos) ** 2) + \
                       np.sum((np.matmul(d_pxpos, t_mat)+offset - d_gmpos) ** 2) + \
                       np.sum((np.matmul(e_pxpos, t_mat)+offset - e_gmpos) ** 2)
            return loss
        frnr_losses = []
        with open(point_file) as f:
            lines = f.readlines()
            #line[0] = C:\Users\Idefix\PycharmProjects\tmpProject\cam_footage\DJI_0456_frames\frame_2075.jpg,0.0984375#0.41015625,0.9765625#0.4296875,0.5078125#0.8583984375,0.48125#0.6181640625,0.47291666666666665#0.568359375,0.5151041666666667#0.6025390625,0.5083333333333333#0.556640625,0.5140625#0.5654296875
            for line in lines:
                framenr = int(line.split(",")[0].split("_")[-1].replace(".jpg", ""))
                points_pxpos = [(float(tup.split("#")[0]), float(tup.split("#")[1])) for tup in line.split(",")[1:]]
                gully_u_pxpos, gully_o_pxpos, well_pxpos, dach_u_pxpos, dach_o_pxpos = points_pxpos[:5]
                frnr_losses.append((framenr, loss_factory(a_pxpos=gully_u_pxpos, b_pxpos=gully_o_pxpos, c_pxpos=well_pxpos, d_pxpos=dach_u_pxpos, e_pxpos=dach_o_pxpos), points_pxpos))

        # fit liner transformation t pixelPosition -> googlemapsPosition, so that t(objp)-objg for (objp, objg) in [gully_u, gully_o, well] is minimised
        t_flat = np.array([1, 0, 0, 1, 0, 0])

        gps_pos_all = {}  # framenumber -> [(gps_lat, gps_long)], index look at point_of_interest_numbers.jpg
        for (frnr, loss, pxposs) in frnr_losses:
            print("frnr = ", frnr)
            res = scipy.optimize.minimize(loss, t_flat)
            if res.success:
                #x: array([-7.45058060e-09,  4.99999925e-02, -5.00000075e-02, -7.45058060e-09])
                t = res.x
                print("res = ", t)
                t_mat = np.array([[t[0], t[1]], [t[2], t[3]]])
                offset = np.array([t[4], t[5]])
                gps_pos = [np.matmul(cp, t_mat)+offset for cp in pxposs]
                gps_pos_all[frnr] = gps_pos
                bearing[frnr] = bearing_from_gps_points(0.5*np.array(gps_pos[5])+0.5*np.array(gps_pos[6]), 0.5*np.array(gps_pos[7])+0.5*np.array(gps_pos[8]))
                carpos[frnr] = gps_pos[5:9]
                print(f"bearing[frnr={frnr}] =", bearing_from_gps_points(0.5*np.array(gps_pos[5])+0.5*np.array(gps_pos[6]), 0.5*np.array(gps_pos[7])+0.5*np.array(gps_pos[8])))
                print(f"carpos[frnr={frnr}] =", str(gps_pos[5:9]).replace("array", "np.array"))
            else:
                print("minimize failed, res = ", res)
        gps_pos_avg = [(51.0, 6.0)]*len(gps_pos_all[list(gps_pos_all.keys())[0]])
        assert len(gps_pos_avg) == 88
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
            dists_loss = np.sum(np.array([abs_meter_diff([flat_gps_pos[2*k0], flat_gps_pos[2*k0+1]], [flat_gps_pos[2*k1], flat_gps_pos[2*k1+1]])-dists_m[(k0, k1)] for (k0, k1) in dists_m.keys()])**2)
            mespos_loss = np.sum([abs_meter_diff(gps_pos_avg[i], (flat_gps_pos[2*i], flat_gps_pos[2*i+1]))**2 for i in range(len(gps_pos_avg)) if i not in [5, 6, 7, 8]])
            return dists_loss + mespos_loss

        flat_gpspos = np.array([gps_pos_avg[int(i//2)][int(i%2)] for i in range(2*len(gps_pos_avg))])
        print("constpoint_loss = ", 0)
        print("prae_loss = ", loss(flat_gpspos))

        res = scipy.optimize.minimize(loss, flat_gpspos)

        res_flat_gpspos = res.x
        print("post_loss = ", loss(res_flat_gpspos))
        gps_pos_optimised = [(res_flat_gpspos[2*i], res_flat_gpspos[2*i+1]) for i in range(len(gps_pos_avg))]
        #print("success = ", res.success)
        print("gps_pos_optimised = [", ", ".join([f"({gp[0]},{gp[1]})" for gp in gps_pos_optimised])+"]")
        print("dist(gps_pos, gps_pos_optimised) = ", [abs_meter_diff(gps, opt_gps) for (gps, opt_gps) in zip(gps_pos_avg, gps_pos_optimised)])

    fixpoitns_gpspos = gps_pos_optimised[:5]
    blue_cone_gpspos = gps_pos_optimised[9:47]
    yellow_cone_gpspos = gps_pos_optimised[47:]

    #for (pred_fixpoint, real_fixpoint) in zip(fixpoitns_gpspos, [gully_u_gmpos, gully_o_gmpos, well_gmpos, dach_u_gmpos, dach_o_gmpos]):
    #    print("diff between real and pred fixpoint position: ", abs_meter_diff(pred_fixpoint, real_fixpoint))
    #print("2m = ", abs_meter_diff(gps_pos_optimised[68], gps_pos_optimised[67]), "m")
    #for k in carpos.keys():
    #    print(f"car_gpspos[{k}] = ", (np.average([x for (x, y) in carpos[k]]), np.average([y for (x, y) in carpos[k]])))
    #print("blue_cone_gpspos = ", str(blue_cone_gpspos).replace("array", "").replace("[", "").replace("]", ""))
    #print("yellow_cone_gpspos = ", str(yellow_cone_gpspos).replace("array", "").replace("[", "").replace("]", ""))

    plot_on_googlemap = False
    if plot_on_googlemap:
        # plot gps_pos on google maps
        gmap = CustomGoogleMapPlotter(51.4639933,  6.73884552, 25, map_type='satellite')  # making 25 larger -> smaler map cutout shown by default
        #gmap.draw("map.html")
        gmap.color_scatter([x for (x, y) in gps_pos_optimised[:5]], [y for (x, y) in gps_pos_optimised[:5]], size=0.228, c="black")
        for k in carpos.keys():
            gmap.color_scatter([x for (x, y) in carpos[k]], [y for (x, y) in carpos[k]], size=0.228, c="purple")
        gmap.color_scatter([x for (x, y) in blue_cone_gpspos], [y for (x, y) in blue_cone_gpspos], size=0.228, c="blue")
        gmap.color_scatter([x for (x, y) in yellow_cone_gpspos], [y for (x, y) in yellow_cone_gpspos], size=0.228, c="yellow")
        gmap.draw('map.html')

    return fixpoitns_gpspos, blue_cone_gpspos, yellow_cone_gpspos, carpos, bearing


def custom_pnp():
    #run 3 (14:46)
    # meterpos: [(meter_north, meter_east) from car to cone.]. needs to be rotated into car?/camera frame
    gps_pos = [(51.46411414940673,6.738944977588712), (51.46444738631447,6.738809863549967), (51.464376920188684,6.739083154724557), (51.46446718888097,6.738927768062857), (51.46447029093418,6.738909215184113), (51.46433948081566,6.738858198040306), (51.464340634576544,6.738867556464109), (51.464332301312695,6.738860518358256), (51.46433352256563,6.7388693023431845), (51.46425560352164,6.738978243689988), (51.46427108152239,6.738956349310485), (51.46430420716902,6.738933864796242), (51.46434835826103,6.738916791956062), (51.46439114717442,6.738898908652391), (51.464435010009225,6.738881909437699), (51.46447785743073,6.738871411732899), (51.46450255330567,6.738853593635997), (51.464523350787786,6.738819787999573), (51.464533504173595,6.738776268215176), (51.46453037892502,6.738730218636211), (51.464517197636866,6.7386905132803125), (51.46449659091481,6.738663015861728), (51.46446663771645,6.738648698012101), (51.464438661684675,6.738653402331532), (51.46441130495734,6.738677405727721), (51.46439626718731,6.73871541407409), (51.464389944382496,6.7387554556266025), (51.46438436411264,6.738785735746747), (51.46437249221701,6.738802412049573), (51.46433474605532,6.738820935341453), (51.46429112213464,6.738838149987502), (51.46425084032891,6.738857744036035), (51.46420852749894,6.73887419245192), (51.46416856291396,6.738895720484548), (51.46412900038989,6.738912834604563), (51.46410496482645,6.738923655505124), (51.464079595253736,6.738958957232722), (51.464067119195114,6.738998752044335), (51.464066961660166,6.739049464522678), (51.46407733183846,6.739086333417893), (51.46410152721491,6.739123619222857), (51.46413179871855,6.739137612689789), (51.46415989439572,6.7391333201885475), (51.46419621559581,6.739098942910411), (51.464209958613,6.739064370135348), (51.464218071281536,6.739045220236359), (51.464230863835915,6.739022540553707), (51.464235195017565,6.73894986483897), (51.46425254304922,6.738924631810385), (51.464298461923505,6.738896452978141), (51.4643428654891,6.7388789814292425), (51.46438668534625,6.738860624573461), (51.46442779156127,6.7388451840515105), (51.46447030037779,6.7388281064257844), (51.46448691323747,6.738816478338111), (51.46449890894766,6.738795981253502), (51.46450483971843,6.738769736369862), (51.4645028126984,6.738742394083106), (51.464494201192664,6.738718220083149), (51.464480374977576,6.738700955868589), (51.46446375293953,6.738694369123524), (51.464446342496,6.7386975953166015), (51.46443094576782,6.738710948324009), (51.46442003090998,6.738732241930608), (51.46441452623774,6.738758460709004), (51.4644094765204,6.738799210265234), (51.464398695574275,6.738827028726462), (51.464380796405976,6.738840788713605), (51.4643624111585,6.738848593652515), (51.4643380159601,6.738858309555241), (51.46429508893946,6.738875725158482), (51.46425547308892,6.738894756649162), (51.464215777303274,6.738910776637766), (51.46417554371516,6.7389327114558935), (51.46413319131306,6.738958079522211), (51.46411580753114,6.73896635691615), (51.46410299161643,6.738986446365815), (51.46409511800807,6.7390122939678285), (51.46409565819885,6.739041007544318), (51.46410386007197,6.739066872605536), (51.4641179539995,6.739085351021696), (51.46413588370013,6.739092211889237), (51.464154482508064,6.739087866698141), (51.464172734232235,6.739071377317364), (51.464183024539835,6.739047157912321), (51.46419395425206,6.739020047700429), (51.46420877083953,6.738992929967574), (51.46422267511233,6.738971629815222)]
    carpos = {3279:[np.array([51.46436821,  6.73881693]), np.array([51.46437184,  6.73883334]), np.array([51.46435367,  6.73882403]), np.array([51.46435905,  6.73884112])],
              4029:[np.array([51.46424119,  6.738988  ]), np.array([51.46423427,  6.73897845]), np.array([51.46425317,  6.73897375]), np.array([51.46424365,  6.7389631 ])]}
    #bounding_box_file = f"C:/Users/Idefix/PycharmProjects/datasets/fscoco_sample_translated/labels/frame_{left_cam_frnr}.txt"

    # cone_nr: (0=blue cone, 1=yellow cone), (0=left .. 1=right), (0=top .. 1=bottom), width/img_width, height/img_height
    # cone class labels, position and size read from labels/frame_2632.txt:
    # point_of_interest_index read from comparing drone view, camara view and point_of_interest_numbers.png
    # cones[point_of_interest_id] = (class 0=Blue, 1=Yellow, position_width, position_height, width, height)
    coneboundingboxes_camL3_2032 = {
        28: (0, 0.21848958333333332, 0.6345833333333334, 0.08802083333333334, 0.1925),
        27: (0, 0.325, 0.44083333333333335, 0.046875, 0.125),
        26: (0, 0.14557291666666666, 0.31666666666666665, 0.034895833333333334, 0.08333333333333333),
        25: (0, 0.08463541666666667, 0.24333333333333335, 0.0140625, 0.055),
        24: (0, 0.1390625, 0.21291666666666667, 0.013541666666666667, 0.0475),
        23: (0, 0.259375, 0.1975, 0.0125, 0.04),
        22: (0, 0.38697916666666665, 0.19333333333333333, 0.009375, 0.03833333333333333),
        21: (0, 0.5395833333333333, 0.19791666666666666, 0.009375, 0.0325),
        20: (0, 0.6763020833333333, 0.21875, 0.0109375, 0.030833333333333334),
        19: (0, 0.8239583333333333, 0.24333333333333335, 0.007291666666666667, 0.03333333333333333),
        18: (0, 0.9802083333333333, 0.27375, 0.0125, 0.0325),
        56: (1, 0.91640625, 0.27291666666666664, 0.0171875, 0.03916666666666667),
        57: (1, 0.8098958333333334, 0.2525, 0.010416666666666666, 0.028333333333333332),
        58: (1, 0.7028645833333333, 0.23291666666666666, 0.011979166666666667, 0.0375),
        59: (1, 0.59921875, 0.22125, 0.009895833333333333, 0.0275),
        60: (1, 0.5036458333333333, 0.215, 0.0125, 0.035),
        61: (1, 0.4278645833333333, 0.22333333333333333, 0.0109375, 0.03833333333333333),
        62: (1, 0.38697916666666665, 0.23583333333333334, 0.011458333333333333, 0.028333333333333332),
        63: (1, 0.39921875, 0.25708333333333333, 0.0171875, 0.04583333333333333),
        64: (1, 0.52265625, 0.2941666666666667, 0.018229166666666668, 0.058333333333333334),
        65: (1, 0.8427083333333333, 0.39666666666666667, 0.040625, 0.09333333333333334)
    }

    coneboundingboxes_camL3_2632 = {
        46: (0, 0.478125, 0.424583, 0.051042, 0.120833),
        45: (0, 0.581771, 0.342500, 0.033333, 0.076667),
        44: (0, 0.599219, 0.302500, 0.024479, 0.063333),
        43: (0, 0.605990, 0.252917, 0.019271, 0.044167),
        42: (0, 0.752604, 0.241250, 0.013542, 0.037500),
        41: (0, 0.871094, 0.253333, 0.016146, 0.030000),
        84: (1, 0.947396, 0.351667, 0.025000, 0.070000),
        83: (1, 0.890885, 0.308750, 0.022396, 0.052500),
        82: (1, 0.925521, 0.295417, 0.013542, 0.042500)
    }
    #cone_keypoints[point_of_interest_id] = (top, band_left_top, band_left_bottom, left_bottom, band_right_top, band_right_bottom, right_bottom)
    # keypoints of cone point_of_interest_id in frame 2632 of camL3
    cone_keypoints_camL3frame_2032 = {
        28: ((0.2916666666666667,0.042222222222222223), (0.19722222222222222,0.3433333333333333), (0.18666666666666668,0.6344444444444445), (0.18833333333333332,0.9222222222222223), (0.5272222222222223,0.29333333333333333), (0.67,0.5633333333333334), (0.7866666666666666,0.8222222222222222)),
        27: ((0.575,0.0811111111111111), (0.3861111111111111,0.31444444444444447), (0.2833333333333333,0.5922222222222222), (0.17333333333333334,0.8366666666666667), (0.7172222222222222,0.3188888888888889), (0.7733333333333333,0.6066666666666667), (0.8244444444444444,0.8433333333333334)),
        26: ((0.4722222222222222,0.08222222222222222), (0.3416666666666667,0.31), (0.2683333333333333,0.59), (0.18,0.8388888888888889), (0.6122222222222222,0.29333333333333333), (0.715,0.5944444444444444), (0.7833333333333333,0.8177777777777778)),
        25: ((0.565,0.12777777777777777), (0.34444444444444444,0.3688888888888889), (0.23722222222222222,0.6788888888888889), (0.1827777777777778,0.83), (0.7061111111111111,0.35333333333333333), (0.7705555555555555,0.7222222222222222), (0.81,0.9244444444444444)),
        24: ((0.6572222222222223,0.08888888888888889), (0.48333333333333334,0.2722222222222222), (0.34833333333333333,0.5855555555555556), (0.2594444444444444,0.77), (0.7838888888888889,0.2788888888888889), (0.8383333333333334,0.6366666666666667), (0.8511111111111112,0.7977777777777778)),
        18: ((0.625,0.26222222222222225), (0.4583333333333333,0.43333333333333335), (0.35,0.7211111111111111), (0.2611111111111111,0.9444444444444444), (0.7794444444444445,0.4666666666666667), (0.8155555555555556,0.7577777777777778), (0.8505555555555555,0.9777777777777777)),
        56: ((0.5916666666666667,0.2922222222222222), (0.4122222222222222,0.4677777777777778), (0.3372222222222222,0.6722222222222223), (0.2611111111111111,0.8966666666666666), (0.7316666666666667,0.48777777777777775), (0.7461111111111111,0.7022222222222222), (0.7577777777777778,0.9644444444444444)),
        65: ((0.6672222222222223,0.08666666666666667), (0.4488888888888889,0.3244444444444444), (0.36,0.5655555555555556), (0.25555555555555554,0.8066666666666666), (0.7844444444444445,0.37555555555555553), (0.8105555555555556,0.6088888888888889), (0.8366666666666667,0.8622222222222222))
    }
    cone_keypoints_camL3frame_2632 = {
        #poii: ((width, height), ...)
        46: ((0.46444444444444444, 0.0811111111111111), (0.29833333333333334, 0.3411111111111111), (0.24222222222222223, 0.6244444444444445), (0.195, 0.8788888888888889), (0.6572222222222223, 0.32), (0.7411111111111112, 0.6077777777777778), (0.8088888888888889, 0.8622222222222222)),
        45: ((0.5666666666666667, 0.052222222222222225), (0.3761111111111111, 0.29777777777777775), (0.27555555555555555, 0.6088888888888889), (0.205, 0.8544444444444445), (0.7238888888888889, 0.30333333333333334), (0.7805555555555556, 0.6366666666666667), (0.8116666666666666, 0.8844444444444445)),
        44: ((0.7261111111111112, 0.09333333333333334), (0.505, 0.2911111111111111), (0.2811111111111111, 0.5666666666666667), (0.16444444444444445, 0.8088888888888889), (0.8316666666666667, 0.32), (0.8366666666666667, 0.6633333333333333), (0.8411111111111111, 0.9022222222222223)),
        43: ((0.7338888888888889, 0.06), (0.54, 0.2822222222222222), (0.3761111111111111, 0.6666666666666666), (0.26166666666666666, 0.9155555555555556), (0.8605555555555555, 0.2733333333333333), (0.895, 0.6777777777777778), (0.9016666666666666, 0.95)),
        42: ((0.53, 0.12222222222222222), (0.35888888888888887, 0.37), (0.235, 0.6877777777777778), (0.1438888888888889, 0.8911111111111111), (0.6444444444444445, 0.35888888888888887), (0.6816666666666666, 0.7188888888888889), (0.7094444444444444, 0.9488888888888889)),
        41: ((0.6283333333333333, 0.20777777777777778), (0.4905555555555556, 0.38222222222222224), (0.4061111111111111, 0.6933333333333334), (0.3477777777777778, 0.9511111111111111), (0.7261111111111112, 0.41444444444444445), (0.7533333333333333, 0.7522222222222222), (0.7694444444444445, 0.9611111111111111)),
        84: ((0.7577777777777778, 0.11666666666666667), (0.55, 0.35888888888888887), (0.45666666666666667, 0.5666666666666667), (0.33944444444444444, 0.7888888888888889), (0.9116666666666666, 0.37222222222222223), (0.9383333333333334, 0.6088888888888889), (0.94, 0.8755555555555555)),
        83: ((0.6566666666666666, 0.12666666666666668), (0.43944444444444447, 0.39222222222222225), (0.3377777777777778, 0.6133333333333333), (0.2388888888888889, 0.8866666666666667), (0.735, 0.42777777777777776), (0.7338888888888889, 0.6444444444444445), (0.7327777777777778, 0.89)),
        82: ((0.7966666666666666, 0.09222222222222222), (0.5755555555555556, 0.38), (0.47944444444444445, 0.5755555555555556), (0.35777777777777775, 0.8433333333333334), (0.9372222222222222, 0.3933333333333333), (0.9561111111111111, 0.6277777777777778), (0.9722222222222222, 0.8744444444444445)),
    }
    datas = [(carpos[4029], coneboundingboxes_camL3_2632, cone_keypoints_camL3frame_2632, "d4026camL32632"), (carpos[3279], coneboundingboxes_camL3_2032, cone_keypoints_camL3frame_2032, "d3287camL32032")]
    all_true_dist = []
    all_estfrom_dist = []
    all_true_angle = []
    all_estfrom_angle = []
    for data in [datas[1]]:
        car_gpsposes, cones, cone_keypoints, name = data

        car_gpspos = (np.average([x for (x, y) in car_gpsposes]), np.average([y for (x, y) in car_gpsposes]))
        car_bearing = bearing_from_gps_points(0.5*np.array(car_gpsposes[0])+0.5*np.array(car_gpsposes[1]), 0.5*np.array(car_gpsposes[2])+0.5*np.array(car_gpsposes[3]))
        blue_cone_meterpos = [gps2meter(blue_cone[0], blue_cone[1], car_gpspos[0], car_gpspos[1]) for blue_cone in gps_pos[9:47]]
        yellow_cone_meterpos = [gps2meter(yellow_cone[0], yellow_cone[1], car_gpspos[0], car_gpspos[1]) for yellow_cone in gps_pos[47:]]
        poii_cone_meterpos = {}
        for i, v in enumerate(blue_cone_meterpos):
            poii_cone_meterpos[i+9] = v
        for i, v in enumerate(yellow_cone_meterpos):
            poii_cone_meterpos[i+47] = v
        seen_kones_poii = list(cone_keypoints.keys())  # only estimate dist for cones where keypoints could be seen
        true_dist = np.array([abs_value(poii_cone_meterpos[k]) for k in seen_kones_poii])
        true_angle = np.array([car_bearing-np.arctan2(poii_cone_meterpos[k][1], poii_cone_meterpos[k][0]) for k in seen_kones_poii])  # angle between (0, 0), poii_cone_meterpos[k] and (1, 0)
        all_true_dist += list(true_dist)
        all_true_angle += list(true_angle)

        est_from = np.array([avg_pxprom_from_conekeypoints(keypoints=cone_keypoints[k], bounding_box=cones[k]) for k in seen_kones_poii])
        all_estfrom_dist += list(est_from)
        fun, parameters = fit_linear(in_x=est_from, out_true=true_dist)
        estimated_dist = fun(est_from, parameters)
        print("\nest_from = ", est_from)
        print("true_dist     =", true_dist)
        print("fun(est_from) =", estimated_dist)
        print(f"parameters[{name}] = {parameters}, with in_x = cone_keypoints_dist")
        print("diff_dist = ", np.sqrt(np.sum((estimated_dist-true_dist)**2)))

        est_from = np.array([cones[k][1]-0.5*cones[k][3]+cones[k][3]*np.average(np.array([cone_keypoints[k][i][0] for i in range(7)])) for k in seen_kones_poii])
        all_estfrom_angle += list(est_from)
        fun, parameters = fit_linear(in_x=est_from, out_true=true_dist)
        est_angle = fun(est_from, parameters)
        print("\ncones_bearings  = ", true_angle)
        print("est_cones_angle = ", est_angle)
        print(f"parameters[{name}] = {parameters}, with in_x = width position of first keypoint on cone")
        print("diff_angle = ", np.sqrt(np.sum((est_angle-true_angle)**2)))

    all_true_dist = np.array(all_true_dist)
    all_estfrom_dist = np.array(all_estfrom_dist)
    fun, parameters = fit_linear(in_x=all_estfrom_dist, out_true=all_true_dist)
    estimated_dist = fun(all_estfrom_dist, parameters)
    print("\nall_est_from_dist = ", all_estfrom_dist)
    print("all_true_dist     =", all_true_dist)
    print("fun(all_est_from_dist) =", estimated_dist)
    print(f"parameters = {parameters}, with in_x = cone_keypoints_dist")
    print("total_diff_dist = ", np.sqrt(np.sum((estimated_dist-all_true_dist)**2)))
    # parameters = [ 2.06350268e+03  2.99755291e+01  2.57930663e-03 -2.28852681e+00], with in_x = cone_keypoints_dist
    # total_diff_dist =  3.2965310164315325

    all_true_angle = np.array(all_true_angle)
    all_estfrom_angle = np.array(all_estfrom_angle)
    fun, parameters = fit_linear(in_x=all_estfrom_angle, out_true=all_true_angle)
    estimated_angle = fun(all_estfrom_angle, parameters)
    print("\nall_est_from_angle=", all_estfrom_angle)
    print("all_true_angle      =", all_true_angle)
    print("fun(all_est_from_angle) =", estimated_angle)
    print(f"parameters = {parameters}, with in_x = width position of first keypoint on cone")
    print("total_diff_angle = ", np.sqrt(np.sum((estimated_angle-all_true_angle)**2)))
    #parameters = [-5.14513443  4.46082258 -0.10399315  1.74553475], with in_x = width position of first keypoint on cone
    #total_diff_angle =  0.9507291891244651


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
    my_dicts = [read_csv(csv_files_dir / csv_file) for csv_file in csv_files]
    run = my_dicts[0]
    n = 1000  # number of entries per Seconds
    heading_y = run["GNSS_heading_UsbFlRec"]
    heading_x = run["GNSS_heading_UsbFlRec_x"]
    speed_y = run["GNSS_speed_over_ground_UsbFlRec"]  # value
    speed_x = run["GNSS_speed_over_ground_UsbFlRec_x"]  # timestamps
    timesteps, xp, yp = get_path(run)
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
    dt = int(n)  # take heading and speed over 0.1s intervals to avoid noise
    for i in range(len(xp)-dt):
        d_pos = np.array(gps2meter(xp[i+dt], yp[i+dt], xp[i], yp[i]))  # dpos = vector from position to position in 1 seconds in meter
        d_t = timesteps[i+dt]-timesteps[i]
        length = np.sqrt(sum(d_pos**2))
        if d_t > 0:
            #time
            speed_from_pos_x.append(0.5*timesteps[i+dt]+0.5*timesteps[i])
            #speed
            speed_from_pos_y.append(length/d_t)
        if d_t > 0 and length > 0:
            #heading
            heading_from_pos_x.append(0.5*timesteps[i+dt]+0.5*timesteps[i])
            #x_pos: lattitude position
            #y_pos: longitude position
            heading_from_pos_y.append(bearing_from_gps_points((xp[i], yp[i]), (xp[i+dt], yp[i+dt])))
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
    sx, sheading_y, sheading_from_pos_y = timesinc(heading_x, heading_y, heading_from_pos_x, heading_from_pos_y)
    print("len(sx, sheading_y, sheading_from_posy) = ", len(sx), len(sheading_y), len(sheading_from_pos_y))
    axe.plot(heading_from_pos_x, heading_from_pos_y*180/np.pi, label="sheading_from_pos_y")
    axe.plot(heading_x, heading_y, label="heading_y")
    #axe.plot(sx, sheading_from_pos_y, label="sheading_from_pos_y")
    #axe.plot(heading_x, heading_y, label="heading")
    axe.legend()
    axe.grid()
    plt.show()


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


def get_synced_frame(camL3_frame):
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
    fixpoitns_gpspos, blue_cone_gpspos, yellow_cone_gpspos, carpos, bearing = true_pos_from_droneimg_pxpos()
    sensor_data = read_csv(csv_files_dir / "alldata_2022_12_17-14_43_59_id3.csv")
    td = 27.92  # seconds that sensor_data started recording before drone3
    for k in bearing.keys():
        #"GNSS_heading_UsbFlRec", "GNSS_latitude_UsbFlRec", "GNSS_longitude_UsbFlRec"
        gps_heading = get_at_time(sensor_data["GNSS_heading_UsbFlRec"+x_ending], sensor_data["GNSS_heading_UsbFlRec"], k/25+td)
        print(f"gps_heading.at({k/25+td}) = {gps_heading} = {(bearing[k]+np.pi)*180/np.pi} = bearing[{k}]")
    for k in carpos.keys():
        gps_lat = get_at_time(sensor_data["GNSS_latitude_UsbFlRec"+x_ending], sensor_data["GNSS_latitude_UsbFlRec"], k/25+td)
        gps_long = get_at_time(sensor_data["GNSS_longitude_UsbFlRec"+x_ending], sensor_data["GNSS_longitude_UsbFlRec"], k/25+td)
        carposition = (np.average([lat for (lat, lon) in carpos[k]]), np.average([lon for (lat, lon) in carpos[k]]))
        #print(f"sensor_data.gps[{k/25+td}] = {(gps_lat, gps_long)}, \tcarpos[k] = {carposition}")
        print(f"abs meter diff between sensor_data.gps[{k/25+td}] and carpos[{k}] = ", abs_meter_diff((gps_lat, gps_long), carposition))


def visual_pipeline(frame_path):
    bounding_boxes = [(0, 0.5, 0.5, 0.2, 0.2)]
    bearing = 0
    car_pos = (0, 0)
    detections = []
    for bb in bounding_boxes:
        keypoints = [(0.5, 0), (0.4, 0.3), (0.3, 0.6), (0.2, 0.9), (0.6, 0.3), (0.7, 0.6), (0.8, 0.9)]
        mpropx = avg_pxprom_from_conekeypoints(keypoints, bb)
        dist = 2063.50268e+03*mpropx+29.9755291*mpropx**2+2.57930663e-03/mpropx-2.28852681e+00
        width_pxpos = bb[1]-0.5*bb[3]+bb[3]*np.average(np.array([keypoints[i][0] for i in range(7)]))
        heading = 1*width_pxpos+0-bearing
        # heading = 0 -> cone is in front of car
        # heading = pi/2 -> cone is right of car
        (front, right) = (dist*np.cos(heading), dist*np.sin(heading))  # TODO is not left/right, cause bearing of car already integreated
        detections.append((front, right))
    cone_pos = [(car_pos[0]+x, car_pos[1]+y) for (x, y) in detections]
    #TODO plot cone_pos and match with true_cone_pos


if __name__ == "__main__":
    # (take bearing for left wheels and right wheels, then average) is not equal to (take average of front wheels and back wheels, then bearing)
    #custom_pnp()

    #true_pos_from_droneimg_pxpos("C:/Users/Idefix/PycharmProjects/datasets/keypoints/droneview_annotations.csv")
    show_sensorlogs()
