import os
import pathlib

import matplotlib.collections
import numpy as np
import mat73
import datetime
import matplotlib.pyplot as plt
import scipy.optimize

from util import getType, plot_and_save, smothing, to_range, get_at_time, multi_timesinc, fit_poly_fun_and_print
import gps_util

#<constants>
epoch = datetime.datetime.utcfromtimestamp(0)

time_format_str = "%Y_%m_%d-%H_%M_%S"
x_ending = "_x"
relevant_keys = ["BMS_SOC_UsbFlRec",
                 "Converter_L_N_actual_UsbFlRec", "Converter_R_N_actual_UsbFlRec",
                 "Converter_L_RPM_Actual_Filtered_UsbFlRec", "Converter_R_RPM_Actual_Filtered_UsbFlRec",
                 "Converter_L_Torque_Out_UsbFlRec", "Converter_R_Torque_Out_UsbFlRec",
                 "ECU_ACC_X_UsbFlRec", "ECU_ACC_Y_UsbFlRec", "ECU_ACC_Z_UsbFlRec",
                 "GNSS_heading_UsbFlRec", "GNSS_latitude_UsbFlRec", "GNSS_longitude_UsbFlRec", "GNSS_speed_over_ground_UsbFlRec",
                 "SWS_angle_UsbFlRec"]
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
def poii2cls(poii):
    # returns class of cone marked by poi with id=poii. (0=blue, 1=yellow, 2=car, 3=fixed, -99=outofbounds)
    return 0 if poii_bluecones_range[0] <= poii < poii_bluecones_range[1] else \
        (1 if poii_yellowcones_range[0] <= poii < poii_yellowcones_range[1] else
        (2 if poii_carpos_points_range[0] <= poii < poii_carpos_points_range[1] else
        (3 if poii_const_points_range[0] <= poii < poii_const_points_range[1] else -99)))

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


drone_som_frame = {0:2075, 3:2340}
camL_som_frame = {0:2610, 3:1281}
camR_som_frame = {0:2508, 3:1225}
ssd_som_seconds = {0:54, 3:118}  # lat and long would be syncd between gnss and droneimage if ssd_som_seconds[3] = 120
# drone: drone_frnr
# camL: caml_frnr
# t: time since start of moving (som)
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


def remove_zeros(x, y, zero=0):
    # remove elements from x and y if the element of y is zero
    assert len(x) == len(y)
    tmp = [(a, b) for (a, b) in zip(x, y) if b != zero]
    x = np.array([a for (a, b) in tmp])
    y = np.array([b for (a, b) in tmp])
    return x, y


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
    r = float(np.median([obj_Distm[(i, j)]/abs_value(keypoints[i]-keypoints[j]) for i in range(6) for j in range(i+1, 7)]))
    # distance to object [m] = real object size(m) * focal length (mm) / object height in frame (mm)
    #  with object height in frame (mm) = sensor height (mm) * object height (px) / sensor height (px)
    # distance to object [m] = real object size(m)/ object height (px) * constant
    #  with constant = focal length (mm) * sensor height (px) / sensor height (mm)
    # this returns average of
    #  real object size(m) / (object size (px) / sensor height (px))
    return r
#</utility methods>


def plot_colorgradient_carpath(name: str, lat_pos: [gps_util.lattitude], long_pos: [gps_util.longitude], time: [seconds] = None) -> plot:
    # https://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line-in-matplotlib
    # https://matplotlib.org/2.0.2/examples/axes_grid/demo_parasite_axes2.html
    """
    assumes x_pos, y_pos are positions of car during run and plots it and cones.
    """
    assert len(lat_pos) == len(long_pos)
    fig, ax0 = plt.subplots()
    cmap = plt.get_cmap('jet')

    #plot path
    points = np.array([long_pos, lat_pos]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = matplotlib.collections.LineCollection(segments, array=np.linspace(0, 1, len(lat_pos)), cmap=cmap, norm=plt.Normalize(0.0, 1.0), linewidth=2, alpha=1)
    ax0.add_collection(lc)  # label="gnss positions", but add_collection has no label

    # plot cones
    poi_true_gpspos, carpos = true_pos_from_droneimg_pxpos()
    ycps = poi_true_gpspos[poii_yellowcones_range[0]:poii_yellowcones_range[1]]
    ax0.scatter([ycp[1] for ycp in ycps], [ycp[0] for ycp in ycps], c="yellow", label="yellow cones")
    bcps = poi_true_gpspos[poii_bluecones_range[0]:poii_bluecones_range[1]]
    ax0.scatter([bcp[1] for bcp in bcps], [bcp[0] for bcp in bcps], c="blue", label="blue cones")


    # plot carpos (car positions from drone view)
    carposkeys = list(carpos.keys())
    carposkeys.sort()
    carpos_time = [t2ssdt(drone2t(carposkeys[i]))+2 for i in range(len(carposkeys))]
    carpos_val = [gps_util.carposs_to_gnsspos(carpos[k]) for k in carposkeys]
    ax0.plot([long for (lat, long) in carpos_val], [lat for (lat, long) in carpos_val], color=(0, 0, 0), label="true_position")

    # add time to points on path
    if time is not None:
        assert len(lat_pos) == len(time)
        t = 0
        ax0.text(long_pos[0], lat_pos[0], str(int(time[0])) + "s", c=cmap(0))
        for i in range(len(lat_pos)):
            dt = 10
            if time[i] > t + dt:  # plot only one number per dt seconds.
                ax0.text(long_pos[i], lat_pos[i], str(int(time[i])) + "s", c=cmap(i / len(lat_pos)))
                t += dt
        ax0.text(long_pos[-1], lat_pos[-1], str(int(time[-1])) + "s", c=cmap(1))
    ax0.set_title(f"Path of {name} testrun")
    ax0.scatter(long_pos, lat_pos, s=10, c=cmap(np.linspace(0, 1, len(lat_pos))), alpha=0.5)

    # add labels and meter scaling on both axies  # TODO meter labels on axis may be inaccurate, use gps_utils
    #lat_pos, long_pos = gps2meter(lat_pos, long_pos, lat_pos[0], long_pos[0])
    #tmp = 111320 * np.cos(np.average(lat_pos))
    #long_avg = np.average(long_pos)
    #axright = ax0.secondary_yaxis('right', functions=(lambda y: (y - long_avg) * tmp, lambda y: long_avg + y / tmp))
    #axright.set_xlabel("m North")
    #lat_avg = np.average(lat_pos)
    #axtop = ax0.secondary_xaxis("top", functions=(lambda x: (x - np.average(lat_pos)) * 111320, lambda x: lat_avg + x / 111320))
    #axtop.set_ylabel('m East')

    ax0.set_xlabel("Longitude")
    ax0.set_ylabel("Lattitude")
    ax0.grid()
    ax0.legend()
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
    ssd_name = data[startTimestamp_key].strftime(time_format_str)
    print("\nvisualise data ", ssd_name)
    keys_where_zero_is_nan = ["BMS_SOC_UsbFlRec", "GNSS_latitude_UsbFlRec", "GNSS_longitude_UsbFlRec"]
    show_avgs = {"GNSS_speed_over_ground_UsbFlRec", "Converter_L_N_actual_UsbFlRec", "Converter_R_N_actual_UsbFlRec", "Converter_L_Torque_Out_UsbFlRec", "Converter_R_Torque_Out_UsbFlRec"}
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
            print(f"{k} is in cam_keys={cam_keys}, but should only be in relevant_keys={relevant_keys}")
        elif len(data[k]) == 0:
            print(f"{k} is empty")
        else:
            min_elem = min(data[k])
            max_elem = max(data[k])
            if min_elem == max_elem:
                print(f"{k} is constantly {min_elem} for all {len(data[k])} elements")
            else:
                tmp = [t for t in data[k] if t != 0]
                print(f"k = {k}, type(data[k]) = {getType(data[k])}")
                print(f"{k}: {getType(data[k])}\non average = {np.average(data[k])}, nonzero_average = {np.average(tmp)}, (min, max) = ({min_elem}, {max_elem})")
                # plot
                x, y = data[k + x_ending], data[k]
                if k in keys_where_zero_is_nan:
                    x, y = remove_zeros(x, y)  # might be unnesary cause of outlier filtering in plot_and_save
                plot_name = k_to_name(k)
                plot_and_save(plot_name, x, [y], f"vis_out/{ssd_name}__{plot_name}.png", avgs=k not in show_avgs)

    # show gnss path
    timesteps, (lat_val, long_val) = multi_timesinc([(data["GNSS_latitude_UsbFlRec"+x_ending], data["GNSS_latitude_UsbFlRec"]*np.pi/180), (data["GNSS_longitude_UsbFlRec"+x_ending], data["GNSS_longitude_UsbFlRec"]*np.pi/180)])
    plot_colorgradient_carpath(ssd_name, lat_val, long_val, time=timesteps)
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


def true_pos_from_droneimg_pxpos(point_file=None) -> ([gps_util.gps_pos_radiant], {drone_frnr: [(gps_util.gps_pos_radiant, gps_util.gps_pos_radiant, gps_util.gps_pos_radiant, gps_util.gps_pos_radiant)]}):
    #poi_true_gps_positions_radiants, carposes = true_pos_from_droneimg_pxpos()
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
        gps_pos_optimised = [(0.8982182210076008,0.11761677884573399), (0.8982240207851603,0.11761439661254934), (0.8982228157511569,0.11761916589113572), (0.8982243813174298,0.11761644478756571), (0.8982244243271972,0.11761612654085912), (0.8982211786204956,0.11761627729857761), (0.8982211027052457,0.11761604569032416), (0.8982214064340305,0.11761615453335231), (0.8982213368963968,0.11761592329539663), (0.8982207097104085,0.11761734785107755), (0.898220972923226,0.11761696946853356), (0.8982215344994736,0.11761656131399705), (0.89822230121576,0.117616263725659), (0.8982230448865751,0.11761595884154948), (0.898223809397927,0.11761565807781019), (0.898224556775676,0.11761548206172813), (0.8982249801367144,0.11761516501640397), (0.8982253333728721,0.11761457206936862), (0.8982254997517212,0.1176138119848852), (0.8982254440273627,0.11761300829754578), (0.8982252074775846,0.11761231751350192), (0.898224848418991,0.11761183584807507), (0.8982243243362253,0.11761158645940734), (0.8982238339162804,0.11761167409156476), (0.8982233622988527,0.11761209196087766), (0.8982231180250234,0.11761274985271367), (0.8982230162559344,0.11761344361293623), (0.8982229161889856,0.11761399105148052), (0.8982227135243428,0.11761428194190957), (0.8982220565241292,0.11761460733142542), (0.898221290897569,0.11761491014141702), (0.898220607303135,0.11761525393311756), (0.8982198735896013,0.11761553718065094), (0.8982191822190919,0.11761591631005809), (0.8982184996065987,0.11761622159797593), (0.8982180828136586,0.1176164074035166), (0.8982176456689048,0.11761701634542274), (0.8982174331995965,0.11761771066119721), (0.8982174376359787,0.1176185948769381), (0.8982176243955783,0.11761923906031348), (0.8982180541208966,0.1176198792226819), (0.8982185840518709,0.1176201228045366), (0.8982190704151516,0.11762004367505255), (0.8982196902077038,0.1176194445501204), (0.8982199270837671,0.11761885027138556), (0.898220069507932,0.11761851238197939), (0.898220283525288,0.11761812324925625), (0.8982203479531942,0.11761685645617254), (0.8982206436196373,0.1176164177844938), (0.8982214209270633,0.11761591669755603), (0.8982221947801746,0.11761561177912765), (0.8982229628243179,0.11761529394079227), (0.898223680417543,0.11761501606389126), (0.8982244119942173,0.11761472067097846), (0.8982246981336011,0.11761451278874505), (0.8982249045466844,0.11761414827672716), (0.898225000295421,0.11761369039677744), (0.8982249612521553,0.1176132154612773), (0.8982248069140316,0.11761279968727775), (0.8982245668531369,0.1176124982844021), (0.8982242773784097,0.11761238206427871), (0.8982239721133454,0.11761243887413597), (0.8982237061207208,0.11761267046835096), (0.8982235208196628,0.11761304772533318), (0.8982234279587308,0.1176135044717351), (0.8982233371337076,0.11761423272994267), (0.898223171068248,0.11761470556281013), (0.8982228580203477,0.11761494777700605), (0.8982225388163443,0.11761508769106767), (0.8982221151382654,0.11761525461371326), (0.8982213682804293,0.1176155607962523), (0.8982206919272518,0.11761589759573904), (0.8982200060614122,0.11761617552900053), (0.8982193090154453,0.11761656184402262), (0.8982185735523716,0.11761700456914706), (0.8982182711919975,0.11761714348792981), (0.8982180530442236,0.117617494189846), (0.8982179213263409,0.11761794279791318), (0.8982179373454671,0.11761844035753771), (0.8982180808215943,0.11761888809496952), (0.8982183253382147,0.11761920763018495), (0.8982186420238885,0.11761933185917242), (0.8982189625412564,0.11761925496300554), (0.8982192765327836,0.11761896363862441), (0.898219456953607,0.11761853927278595), (0.898219643065057,0.11761807539578412), (0.8982198950043748,0.11761760494509589), (0.8982201331093526,0.11761722960944655)]
        # gps_pos_optimised[poii] = gps position of that poi, in radiants
        # carpos = {donre3_framenr: [front_left_tire, front_right_tire, rear_left_tire, rear_right_tire]:[gps_util.gps_pos_radiants]
        # maybe have this in a file or something. its 180 lines long.
        carpos = {}
        with open("C:/Users/Idefix/PycharmProjects/tmpProject/vp_labels/droneview/droneview_frnr_carposes.txt") as f:
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
        dists_m_car = {(5, 6): 1.23, (7, 8): 1.23, (5, 7): 1.54, (6, 8): 1.54, (5, 8): np.sqrt(1.23**2+1.54**2), (6, 7): np.sqrt(1.23**2+1.54**2)}
        dists_m_cones = {(11, 12): l, (12, 13): l, (13, 14): l, (30, 29): l, (67, 68): 2, (49, 50): l, (50, 51): l, (49, 51):2*l, (67, 69):l, (69, 70):l,
                   (74, 75): 2, (75, 76): 2, (76, 77): 2, (77, 78): 2, (78, 79): 2, (79, 80): 2, (80, 81): 2, (81, 82): 2, (82, 83): 2, (83, 84): 2,
                   (36, 76): kw, (37, 77): kw, (38, 78): kw, (39, 79): kw, (40, 80): kw, (41, 81): kw, (42, 82): kw, (43, 83): kw, (44, 84): kw,
                   (11, 49): 2.5+cw, (12, 50): 2.5+cw, (13, 51): 2.5+cw, (14, 52): 2.5+cw,
                   (53, 54): 2, (54, 55): 2, (55, 56): 2, (56, 57): 2, (57, 58): 2, (58, 59): 2, (60, 61): 2, (61, 62): 2, (62, 63): 2, (63, 64): 2,
                   (17, 55): kw, (18, 56): kw, (19, 57): kw, (20, 58): kw, (21, 59): kw, (22, 60): kw, (23, 61): kw, (24, 62): kw, (25, 64): kw, (26, 65): kw}
        dists_m = {**dists_m_cones, **dists_m_const, **dists_m_car}  # replace with dist_m = dist_m_cones | dist_m_const when python version >= 3.9
        #for (k0, k1) in dists_m.keys():
        #    print(f"dist_m[{k0}][{k1}] = {dists_m[(k0,k1)]} = {abs_meter_diff(gps_pos[k0], gps_pos[k1])}")

        def loss(flat_gps_pos):
            dists_loss = np.sum(np.array([gps_util.gps_to_dist((flat_gps_pos[2*k0], flat_gps_pos[2*k0+1]), (flat_gps_pos[2*k1], flat_gps_pos[2*k1+1]))-dists_m[(k0, k1)] for (k0, k1) in dists_m.keys()])**2)
            mespos_loss = np.sum(np.array([gps_util.gps_to_dist(gps_pos_avg[i], (flat_gps_pos[2*i], flat_gps_pos[2*i+1])) for i in range(len(gps_pos_avg)) if i not in [5, 6, 7, 8]])**2)
            # add parallel-loss
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


def get_boundingboxes_keypoints_poii(cam: str, framenr: int, filter=False) -> [(cone_bounding_box, cone_keypoitns, poii_id)]:
    bbs = get_boundingboxes(cam, framenr)
    if filter:
        tmp = [(bb, get_cone_keypoints(cam, framenr, i), get_poii(cam, framenr, i)) for (i, bb) in enumerate(bbs)]
        return [(bb, kp, poii) for (bb, kp, poii) in tmp if (kp is not None and poii > 0)]
    else:
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

cone_keypoints_cache = {}
def get_cone_keypoints(cam: str, framenr: int, cone: int) -> [(normalised_px_w, normalised_px_h)]:
    if len(cone_keypoints_cache.keys()) == 0:
        cone_keypoints_file = pathlib.Path("./vp_labels/cone_annotations.csv")
        #filename = f"/cones/{cam}_frame_{framenr}.jpg_cone_{cone}.jpg"
        with open(cone_keypoints_file) as f:
            for line in f.readlines()[1:]:
                ls = line.split(",")
                keypoints = [(float(strtup.split("#")[0]), float(strtup.split("#")[1])) for strtup in line.split(",")[1:]]
                assert len(keypoints) == 7
                names = ls[0].split("/")[-1].replace(".jpg", "").split("_")
                #/cones/camL3_frame_1572.jpg_cone_1.jpg
                #/cones/cone_19028.jpg
                if len(names) == 5:
                    cone_keypoints_cache[(names[0], int(names[2]), int(names[4]))] = keypoints
    if (cam, framenr, cone) in cone_keypoints_cache.keys():
        return cone_keypoints_cache[(cam, framenr, cone)]
    else:
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
    try:
        with open(file) as f:
            lines = f.readlines()
            return int(lines[conenr])
    except:
        #print(f"WARNING: no poii available for cam={cam}, framenr={framenr}, conenr={conenr}")
        return -1


def custom_pnp_find_parameters():
    #run 3 (14:46)
    all_true_dist = []
    all_estfrom_dist = []
    all_true_angle = []
    all_estfrom_angle = []
    bbi_poii_label_dir = "C:/Users/Idefix/PycharmProjects/tmpProject/vp_labels/bbi_poii"
    fully_labeld_frames = [int(str(f).replace(".txt", "").split("_")[-1]) for f in os.listdir(bbi_poii_label_dir)]
    poi_true_gpspos, carpos = true_pos_from_droneimg_pxpos()
    print("image_frnr = ", [(camL3_frnr, get_synced_frame(camL3_frnr)) for camL3_frnr in fully_labeld_frames])
    carpos_time = list(carpos.keys())
    carpos_time.sort()
    carpos_time = np.array(carpos_time)
    carpos_value = np.array([carpos[drone3_frnr] for drone3_frnr in carpos_time])
    for camL3_frnr in fully_labeld_frames:
        drone3_frnr = get_synced_frame(camL3_frnr)
        print("(camL3_frnr, drone3_frnr) =", (camL3_frnr, drone3_frnr))
        # load data from files
        car_gpsposes, _ = get_at_time(carpos_time, carpos_value, drone3_frnr)
        name = f"drone3frame_{drone3_frnr}_camL3_{camL3_frnr}"
        bb_kp_poii = get_boundingboxes_keypoints_poii("camL3", camL3_frnr, filter=True)


        # car_gpspos and _bearing from drone image:
        car_gpspos = gps_util.carposs_to_gnsspos(car_gpsposes)
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
    fun, parameters = fit_poly_fun_and_print(in_x=all_estfrom_dist, out_true=all_true_dist, name="dist_total", exponents=[0, 1])

    fig, (dist_axe, heading_axe) = plt.subplots(1, 2)
    dist_axe.set_title(f"customPnP fit dist and heading.")
    dist_axe.set_xlabel("median(dist physical cone[m]/image[px])")
    dist_axe.set_ylabel("true dist(car, con) [m]")
    dist_axe.scatter(all_estfrom_dist, all_true_dist, s=1)
    all_estfrom_dist.sort()
    dist_axe.plot(all_estfrom_dist, np.array([fun(efd, parameters) for efd in all_estfrom_dist]), label=f"{parameters[1]:.3E}*est_from+{parameters[0]:.3E}", color="red")
    dist_axe.grid()
    dist_axe.legend()
    # parameters[dist_total] = [1.03694031e+00 1.69160357e+03]  # average
    # diff_dist_total = 0.03767625078303172
    # parameters[dist_total] = [2.04826435e-01 1.81102254e+03]  # median
    # diff_dist_total = 0.01701737260345642
    all_true_angle = np.array(all_true_angle)
    all_estfrom_angle = np.array(all_estfrom_angle)
    fun, parameters = fit_poly_fun_and_print(in_x=all_estfrom_angle, out_true=all_true_angle, name="bearing_total", exponents=[0, 1])
    heading_axe.set_title(f"({len(all_estfrom_angle)} values from {len(fully_labeld_frames)} frames)")
    heading_axe.set_xlabel("average(width_pos image[px])")
    heading_axe.set_ylabel("true heading from car to cone [rad]")
    heading_axe.scatter(all_estfrom_angle, all_true_angle, s=1)
    heading_axe.plot(all_estfrom_angle, np.array([fun(efa, parameters) for efa in all_estfrom_angle]), label=f"{parameters[1]:.3E}*est_from+{parameters[0]:.3E}", color="red")
    heading_axe.grid()
    heading_axe.legend()
    fig.savefig(vis_out_path/"pnp_dist_heading_fit")
    fig.show()
    # parameters[bearing_total] = [-1.00448915  1.12712196]  # average
    # diff_bearing_total = 0.000993987008819239
    # parameters[bearing_total] = [-1.00109719  1.12032384]  # median
    # diff_bearing_total = 0.0009948634495913714


def custom_pnp_validate():
    bbi_poii_label_dir = "C:/Users/Idefix/PycharmProjects/tmpProject/vp_labels/bbi_poii"
    fully_labeld_frames = [int(str(f).replace(".txt", "").split("_")[-1]) for f in os.listdir(bbi_poii_label_dir)]
    poi_true_gpspos, dronefrmrn_carposes = true_pos_from_droneimg_pxpos()

    carpos_time = list(dronefrmrn_carposes.keys())
    carpos_time.sort()
    carpos_time = np.array(carpos_time)
    carpos_value = np.array([dronefrmrn_carposes[drone3_frnr] for drone3_frnr in carpos_time])
    gps_base = (0.8982182491908725, 0.11761678938800951)
    poi_true_mpos = [gps_util.gps_to_meter(tp, gps_base) for tp in poi_true_gpspos]
    camL3frnr_petcp = {}  # camL3_frnr -> poii_est_true_car_pos of that frame
    for camL3_framenr in fully_labeld_frames:
        poii_est_true_pos = []
        drone3_framenr = get_synced_frame(camL3_framenr)
        car_poses, _ = get_at_time(carpos_time, carpos_value, drone3_framenr)

        car_posm = gps_util.gps_to_meter(gps_util.carposs_to_gnsspos(car_poses), gps_base)
        car_bearing = gps_util.carposs_to_heading(car_poses)
    # < visual pipeline modified
        bounding_boxes = get_boundingboxes("camL3", camL3_framenr)
        if len(bounding_boxes) == 0:
            print(f"visual_pipeline: no cone detected in camL3_frame_{camL3_framenr}")
            continue
        #print(f"\nvisual_pipeline: framenumber = {framenr}, car_bearing={car_bearing}, car_pos = {car_pos}")
        for i, bb in enumerate(bounding_boxes):
            keypoints = get_cone_keypoints("camL3", camL3_framenr, i)
            poii = get_poii("camL3", camL3_framenr, i)
            if keypoints is not None and poii != -1:
                dist, heading = custom_PnP(keypoints, bb)
                #heading = bearing_from_gps_points(cone_gpspos, car_gpspos)-car_bearing
                #print(f"{i}: cone {bb[0]} detected at (dist={dist}, heading={heading}")
                est_pos = car_posm+gps_util.distazimuth_to_meter((dist, to_range(car_bearing+heading)))
                cls = poii2cls(poii)
                if cls != bb[0]:
                    print(f"error: (camL3_framenr={camL3_framenr}, bbi={i} poii {poii} should be cls {cls}, but its bb is cls {bb[0]}")
                    assert cls == bb[0]
                poii_est_true_pos += [(poii, est_pos, poi_true_mpos[poii])]  # (cls, est_pos, true_pos)
        # end visual pipeline modified>
        #avg_error = gps_util.average([est_cone_pos-true_cone_pos for (cls, est_cone_pos, true_cone_pos) in poii_est_true_pos[new_cetp_startindex:]])
        #print(f"avg_error on frame {camL3_framenr} = {avg_error}")

        #if camL3_framenr in [1282, 1283]:  # number_of_visible_cones=len(poii_est_true_pos)-new_cetp_startindex > 15
        camL3frnr_petcp[camL3_framenr] = (poii_est_true_pos, car_poses)

    for camL3_framenr in []:#[1282, 1283]:
        # plot estimated cone positions and true cone positions in same image
        poii_est_true_pos, car_poses = camL3frnr_petcp[camL3_framenr]
        car_posm = car_posm = gps_util.gps_to_meter(gps_util.average(car_poses), gps_base)
        car_bearing = gps_util.carposs_to_heading(car_poses)
        print("plot true_conepos-est_conepos of camL3frame", camL3_framenr)
        fig, axe = plt.subplots()
        axe.set_title(f"true_conepos-est_conepos of camL3 frame {camL3_framenr}")
        axe.set_xlabel("meter east")
        axe.set_ylabel("meter north")
        axe.grid()
        # plot all true cone positions
        # true positions x, estimated position o, unseen cones black, seen cones blue/yellow
        tmp = [gps_util.gps_to_meter(yc, gps_base=gps_base) for yc in poi_true_gpspos[poii_yellowcones_range[0]:poii_yellowcones_range[1]] if gps_util.gps_to_dist(yc, gps_base) < 30]
        axe.scatter(x=[t[1] for t in tmp], y=[t[0] for t in tmp], c="black", marker='x', label="true_pos_yellow_unseen")
        tmp = [gps_util.gps_to_meter(bc, gps_base=gps_base) for bc in poi_true_gpspos[poii_bluecones_range[0]:poii_bluecones_range[1]] if gps_util.gps_to_dist(bc, gps_base) < 30]
        axe.scatter(x=[t[1] for t in tmp], y=[t[0] for t in tmp], c="black", marker='x', label="true_pos_blue_unseen")
        # plot vision cone
        vision_rays = [car_posm+20*np.array([np.cos(car_bearing-1), np.sin(car_bearing-1)]), car_posm, car_posm+20*np.array([np.cos(car_bearing+0.15), np.sin(car_bearing+0.15)])]
        axe.plot([t[1] for t in vision_rays], [t[0] for t in vision_rays], color="red")
        vision_rays = [car_posm+20*np.array([np.cos(car_bearing-0.15), np.sin(car_bearing-0.15)]), car_posm, car_posm+20*np.array([np.cos(car_bearing+1), np.sin(car_bearing+1)])]
        axe.plot([t[1] for t in vision_rays], [t[0] for t in vision_rays], color="green")
        car_poses_m = [gps_util.gps_to_meter(cp, gps_base) for cp in [car_poses[0], car_poses[1], car_poses[3], car_poses[2], car_poses[0], 0.5*np.array(car_poses[0])+0.5*np.array(car_poses[1]), 0.5*np.array(car_poses[2])+0.5*np.array(car_poses[3])]]
        axe.plot([t[1] for t in car_poses_m], [t[0] for t in car_poses_m], color="black")
        # plot detected cone positions, connected to their true positions
        for (poii, est_cone_pos, true_cone_pos) in poii_est_true_pos:
            color = "blue" if poii2cls(poii)==0 else "yellow"
            axe.scatter(x=[est_cone_pos[1]], y=[est_cone_pos[0]], c=color, marker='o', label="true_pos_seen")
            axe.scatter(x=[true_cone_pos[1]], y=[true_cone_pos[0]], c=color, marker='x', label="est_pos")
            axe.plot([est_cone_pos[1], true_cone_pos[1]], [est_cone_pos[0], true_cone_pos[0]], color=color)
            axe.text(x=0.5*est_cone_pos[1]+0.5*true_cone_pos[1], y=0.5*est_cone_pos[0]+0.5*true_cone_pos[0], s=f"{gps_util.meter_meter_to_dist(true_cone_pos, est_cone_pos)}"[:4])
        #axe.legend()
        fig.savefig(vis_out_path / f"true_est_conepos_camL3_frame_{camL3_framenr}.png")
        fig.show()

    # plot (distance between detected cone and true cone position) against true distance
    fig, axe = plt.subplots()
    axe.set_title(f"accuracy of visual pipeline over distance ({sum([len(camL3frnr_petcp[k][0]) for k in camL3frnr_petcp.keys()])} datapoints from {len(fully_labeld_frames)} frames)")
    data = []
    for camL3_frnr in camL3frnr_petcp.keys():
        poii_est_true_pos, car_poses = camL3frnr_petcp[camL3_frnr]
        true_car_mpos = gps_util.gps_to_meter(gps_util.carposs_to_gnsspos(car_poses), gps_base)
        data += [(gps_util.meter_meter_to_dist(true_cone_pos, true_car_mpos), gps_util.meter_meter_to_dist(true_cone_pos, est_cone_pos), "blue" if poii2cls(poii)==0 else "yellow") for (poii, est_cone_pos, true_cone_pos) in poii_est_true_pos]
    axe.scatter(x=[ttd for (ttd, ted, color) in data], y=[ted for (ttd, ted, color) in data], color=[color for (ttd, ted, color) in data], s=1)
    axe.set_xlabel("dist(car, true_cone_pos)")
    axe.set_ylabel("dist(estimated_cone_pos, true_cone_pos)")
    axe.grid()
    fig.savefig(vis_out_path / "visual_pipeline_accuracy.png")
    fig.show()

    # plot error in distance and heading
    data = []
    for camL3_frnr in camL3frnr_petcp.keys():
        poii_est_true_pos, car_poses = camL3frnr_petcp[camL3_frnr]
        true_car_mpos = gps_util.gps_to_meter(gps_util.carposs_to_gnsspos(car_poses), gps_base)
        for (poii, est_pos, true_pos) in poii_est_true_pos:
            (est_dist, est_heading) = gps_util.meter_to_distazimuth(est_pos-true_car_mpos)
            (true_dist, true_heading) = gps_util.meter_to_distazimuth(true_pos-true_car_mpos)
            data.append(((true_dist, gps_util.meter_meter_to_dist(true_pos, est_pos), abs(est_dist-true_dist), abs(np.sin(est_heading-true_heading)*true_dist))))
    # print averages
    avg_eukdist = np.average([euk_dist/true_dist for (true_dist, euk_dist, distx, disty) in data if true_dist < 10])
    avg_distx = np.average([distx/true_dist for (true_dist, euk_dist, distx, disty) in data if true_dist < 10])
    avg_disty = np.average([disty/true_dist for (true_dist, euk_dist, distx, disty) in data if true_dist < 10])
    print(f"avg euklidian dist/td = {avg_eukdist}\navg dist x/td = {avg_distx}\n avg dist y/td = {avg_disty}")
    fig, axe = plt.subplots()
    axe.set_title(f"error of visual pipeline ({sum([len(camL3frnr_petcp[k][0]) for k in camL3frnr_petcp.keys()])} datapoints from {len(fully_labeld_frames)} frames)")
    axe.scatter(x=[true_dist for (true_dist, euk_dist, distx, disty) in data], y=[euk_dist for (true_dist, euk_dist, distx, disty) in data], color="blue", s=1, label="Euclidean_dist", alpha=0.5)
    axe.scatter(x=[true_dist for (true_dist, euk_dist, distx, disty) in data], y=[distx for (true_dist, euk_dist, distx, disty) in data], color="red", s=1, label="parallel_dist", alpha=0.5)
    axe.scatter(x=[true_dist for (true_dist, euk_dist, distx, disty) in data], y=[disty for (true_dist, euk_dist, distx, disty) in data], color="green", s=1, label="perpendicular_dist", alpha=0.5)
    axe.plot([0, 25], [0, avg_eukdist*25], color="blue")
    axe.plot([0, 25], [0, avg_distx*25], color="red")
    axe.plot([0, 25], [0, avg_disty*25], color="green")
    axe.set_xlabel("dist(car, true_cone_pos)")
    axe.set_ylabel("dist(estimated_cone_pos, true_cone_pos)")
    axe.grid()
    axe.legend()
    fig.savefig(vis_out_path / "visual_pipeline_errors.png")
    fig.show()





def custom_PnP(keypoints: [(normalised_px_w, normalised_px_h)], bounding_box) -> (gps_util.meter, gps_util.heading_radiants):
    cls, posw, posh, sizew, sizeh = bounding_box

    mpropx = avg_pxprom_from_conekeypoints(keypoints=keypoints, bounding_box=bounding_box)  # avg of (meter dist on object / pixel dist in image)
    # parameters[dist_total] = [2.04826435e-01 1.81102254e+03]
    # parameters[dist_total] = [-1.97349351e-01  1.82927777e+03]
    # diff_dist_total = 0.018019702882342514
    dist = -1.97349351e-01 + 1.82927777e+03*mpropx

    w = posw-0.5*sizew+sizew*np.average(np.array([keypoints[i][0] for i in range(7)]))  # avg width position of pixels
    # parameters[bearing_total] = [-0.90914045  1.07626391]
    # diff_bearing_total = 0.0009499480879826105
    angle = -0.90914045 + 1.07626391*w
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
    #timesteps, (xp, yp) = multi_timesinc([(xx, xd), (yx, yd)])  # TODO maybe instead of rest of function
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


def visual_pipeline(cam="camL3", framenr=2632) -> [(int, gps_util.meter, gps_util.heading_radiants)]:
    #detections:[(cone_bounding_box, cone_keypoitns, poii_id)] = get_boundingboxes_keypoints_poii(cam, framenr)
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
            #(north_m, east_m) = gps_util.distazimuth_to_meter((dist, to_range(car_bearing+heading)))
            detections.append((bb[0], dist, heading))
    return detections


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


def get_true_carstate():
    visout_praefix = "tcs_"
    #relevant_keys = ["BMS_SOC_UsbFlRec", "Converter_L_N_actual_UsbFlRec", "Converter_R_N_actual_UsbFlRec",
    #"Converter_L_RPM_Actual_Filtered_UsbFlRec", "Converter_R_RPM_Actual_Filtered_UsbFlRec",
    #"Converter_L_Torque_Out_UsbFlRec", "Converter_R_Torque_Out_UsbFlRec",
    # "ECU_ACC_X_UsbFlRec", "ECU_ACC_Y_UsbFlRec", "ECU_ACC_Z_UsbFlRec",
    # "GNSS_heading_UsbFlRec", "GNSS_latitude_UsbFlRec", "GNSS_longitude_UsbFlRec", "GNSS_speed_over_ground_UsbFlRec",
    # "SWS_angle_UsbFlRec"]

    # [|syncd_][vabs|aabs|heading|yawrate]_[gnss|dv|imu]_[time|value]

    # read sensor data
    sdd = read_csv("merged_rundata_csv/alldata_2022_12_17-14_43_59_id3.csv")
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
    vabs_dv_time = [0.0 for _ in range(len(carposkeys))]
    vabs_dv_value = [0.0 for _ in range(len(carposkeys))]
    vabs_dv_time[0] = t2ssdt(drone2t(carposkeys[0]))-0.001
    for i in range(len(carposkeys)-1):
        t0 = t2ssdt(drone2t(carposkeys[i]))
        t1 = t2ssdt(drone2t(carposkeys[i+1]))
        vabs_dv_time[i] = 0.5*t0+0.5*t1
        vabs_dv_value[i] = gps_util.gps_to_dist(gps_util.average(carpos[carposkeys[i-1]]), gps_util.average(carpos[carposkeys[i]]))/(t1-t0)
    vabs_dv_value[0], vabs_dv_value[-1] = (0, 0)
    vabs_time, (sincd_vabs_dv_value, sincd_vabs_gps_value) = multi_timesinc([(vabs_dv_time, vabs_dv_value), (vabs_gnss_time, vabs_gnss_value)])
    plot_and_save(visout_praefix+"vabs from gps and drone", x_in=vabs_time, ys=[smothing(vabs_time, sincd_vabs_dv_value, 0.5), sincd_vabs_gps_value], names=["droneview", "gps"])

    # derivitate of zip(gnss_vabs_time, gnss_vabs_value)
    smothed_gpsv = smothing(time=vabs_gnss_time, values=vabs_gnss_value, t=1)
    ax_gnss_time = np.array([(0.5*vabs_gnss_time[i+1]+0.5*vabs_gnss_time[i]) for i in range(len(smothed_gpsv)-1)])
    ax_gnss_value = [(vabs_gnss_value[i+1]-vabs_gnss_value[i])/(vabs_gnss_time[i+1]-vabs_gnss_time[i]) for i in range(len(vabs_gnss_value)-1)]
    ax_gnss_value[0], ax_gnss_value[-1] = (0, 0)
    #vabs_dv_smothed_value = smothing(vabs_dv_time, smothing(vabs_dv_time, vabs_dv_value, 5), 5)
    #aabs_dv_time = np.array([0.5*vabs_dv_time[i]+0.5*vabs_dv_time[i+1] for i in range(len(vabs_dv_time)-1)])
    #aabs_dv_value = np.array([(vabs_dv_smothed_value[i+1]-vabs_dv_smothed_value[i])/(vabs_dv_time[i+1]-vabs_dv_time[i]) for i in range(len(vabs_dv_value)-1)])
    #aabs_dv_value[0], aabs_dv_value[-1] = (0, 0)
    imu_ax_value_meadian = np.median(ax_imu_value[:int(len(ax_imu_value//2))])
    imu_ay_value_meadian = np.median(ay_imu_value[:int(len(ay_imu_value//2))])
    imu_az_value_meadian = np.median(az_imu_value[:int(len(az_imu_value//2))])
    print("imu_aabs_value[0] = ", abs_value([imu_ax_value_meadian, imu_ay_value_meadian, imu_az_value_meadian]))  # 0.9769764747559495, but in plot same order of magnitude as gnss_aabs_value (m/s)
    imu_aabs_value = [np.sqrt((ax_imu_value[i]-imu_ax_value_meadian)**2+(ay_imu_value[i]-imu_ay_value_meadian)**2+(az_imu_value[i]-imu_az_value_meadian)**2) for i in range(len(ax_imu_time))]
    imu_aabs_value = smothing(ax_imu_time, imu_aabs_value, 3)
    syncd_ax_time, (synced_imu_aabs_vale, synced_aabs_gnss_value) = multi_timesinc([(ax_imu_time, imu_aabs_value), (ax_gnss_time, ax_gnss_value)])  # (aabs_dv_time, aabs_dv_value)
    plot_and_save(visout_praefix+"aabs from gnss and imu", x_in=syncd_ax_time, ys=[synced_imu_aabs_vale, smothing(syncd_ax_time, synced_aabs_gnss_value, 3)], names=["aabs_imu", "aabs_gnss"], avgs=False)

    # get heading from droneview carpositions
    heading_droneview_time = [0.0 for _ in range(len(carposkeys))]
    heading_droneview_value = [0.0 for _ in range(len(carposkeys))]
    for i in range(len(carposkeys)):
        heading_droneview_time[i] = t2ssdt(drone2t(carposkeys[i]))
        heading_droneview_value[i] = gps_util.carposs_to_heading(carpos[carposkeys[i]])
    heading_time, (sincd_heading_droneview_value, sincd_heading_gps_value) = multi_timesinc([(heading_droneview_time, heading_droneview_value), (heading_gnss_time, heading_gnss_value)])
    plot_and_save(visout_praefix+"heading from gps and drone", x_in=heading_time, ys=[sincd_heading_droneview_value, np.array([to_range(x) for x in sincd_heading_gps_value])], names=["droneview", "gps"], avgs=False)

    # derivitate of zip(gnss_heading_time, gnss_heading_value)
    smothed_gps_heading = smothing(time=heading_gnss_time, values=heading_gnss_value, t=1)
    yawrate_gnss_time = [(0.5*heading_gnss_time[i+1]+0.5*heading_gnss_time[i]) for i in range(len(smothed_gps_heading)-1)]
    yawrate_gnss_value = [(smothed_gps_heading[i+1]-smothed_gps_heading[i])/(heading_gnss_time[i+1]-heading_gnss_time[i]) for i in range(len(heading_gnss_time)-1)]
    smothed_dv_heading = smothing(heading_droneview_time, heading_droneview_value, t=1)
    yawrate_dv_time = [(0.5*heading_droneview_time[i+1]+0.5*heading_droneview_time[i]) for i in range(len(heading_droneview_time)-1)]
    yawrate_dv_value = [(smothed_dv_heading[i+1]-smothed_dv_heading[i])/(heading_droneview_time[i+1]-heading_droneview_time[i]) for i in range(len(heading_droneview_time)-1)]
    syncd_yawrate_time, (syncd_yawrate_gnss_value, syncd_yawrate_dv_value) = multi_timesinc([(yawrate_gnss_time, yawrate_gnss_value), (yawrate_dv_time, yawrate_dv_value)])
    plot_and_save(visout_praefix+"yawrate", x_in=syncd_yawrate_time, ys=[syncd_yawrate_gnss_value, syncd_yawrate_dv_value], names=["yawrate_value", "yawrate_dv_value"], avgs=False)

    # get error of odometry:
    diffs = [("vabs", (vabs_dv_time, vabs_dv_value), (vabs_gnss_time, vabs_gnss_value)), ("yawrate", (yawrate_dv_time, yawrate_dv_value), (yawrate_gnss_time, yawrate_gnss_value))]  # ((a_time, a_value), (b_time, b_value)),
    som = 118
    eom = som+68
    for (name, (a_time, a_val), (b_time, b_val)) in diffs:
        som_ai = 0
        eom_ai = len(a_time)
        som_bi = 0
        eom_bi = len(b_time)
        for i, t in enumerate(a_time):
            if t > som:
                som_ai = i
                break
        for i, t in enumerate(a_time):
            if t > eom:
                eom_ai = i
                break
        for i, t in enumerate(b_time):
            if t > som:
                som_bi = i
                break
        for i, t in enumerate(b_time):
            if t > eom:
                eom_bi = i
                break
        a_time = a_time[som_ai:eom_ai]
        a_val = a_val[som_ai:eom_ai]
        b_time = b_time[som_bi:eom_bi]
        b_val = b_val[som_bi:eom_bi]
        syncd_name_time, (syncd_a_val, syncd_b_val) = multi_timesinc([(a_time, a_val), (b_time, b_val)])
        print(f"diff of {name}: syncd_a,b_val type = {getType(syncd_a_val)}, {getType(syncd_b_val)}")
        name_diff = np.array([abs(g-d) for (g, d) in zip(syncd_a_val, syncd_b_val)])
        print(f"avg, median({name}_diff) = {np.average(name_diff)}, {np.median(name_diff)}")
        plot_and_save(f"{visout_praefix}diff_{name}_dv_gnss", x_in=syncd_name_time, ys=[name_diff])

    relevant_keys = ["BMS_SOC_UsbFlRec",
                     "Converter_L_N_actual_UsbFlRec", "Converter_R_N_actual_UsbFlRec",
                     "Converter_L_RPM_Actual_Filtered_UsbFlRec", "Converter_R_RPM_Actual_Filtered_UsbFlRec",
                     "Converter_L_Torque_Out_UsbFlRec", "Converter_R_Torque_Out_UsbFlRec",
                     "ECU_ACC_X_UsbFlRec", "ECU_ACC_Y_UsbFlRec", "ECU_ACC_Z_UsbFlRec",
                     "GNSS_heading_UsbFlRec", "GNSS_latitude_UsbFlRec", "GNSS_longitude_UsbFlRec", "GNSS_speed_over_ground_UsbFlRec",
                     "SWS_angle_UsbFlRec"]

    # no of the combinations look like good fits.
    test_correlation = [((k_to_name("ECU_ACC_Y_UsbFlRec"), sdd["ECU_ACC_Y_UsbFlRec"+x_ending], smothing(sdd["ECU_ACC_Y_UsbFlRec"+x_ending], sdd["ECU_ACC_Y_UsbFlRec"], 5)), ("yawrate_gnss", yawrate_gnss_time, smothing(yawrate_gnss_time, yawrate_gnss_value, 1)))]  # "ECU_ACC_X_UsbFlRec", "Converter_L_N_actual_UsbFlRec", "Converter_L_Torque_Out_UsbFlRec"
    normalise = True
    # [("vx", vabs_gnss_time, smothing(vabs_gnss_time, vabs_gnss_value, 0.5)), ("der(vabs_gnss)", ax_gnss_time, smothing(ax_gnss_time, ax_gnss_value, 3))]:  # ("aabs_imu", ax_imu_time, imu_aabs_value)
    for (name_x, time_x, value_x), (name_y, time_y, value_y) in test_correlation:
        print(f"fit linear function from {name_x} to {name_y}")

        if normalise:
            # normalise values to range [0, 1]
            value_x = np.array(value_x)
            value_y = np.array(value_y)
            minx, maxx = min(value_x), max(value_x)
            miny, maxy = min(value_y), max(value_y)
            value_x = (value_x-minx)/(maxx-minx)
            value_y = (value_y-miny)/(maxy-miny)

        # fit function from normalised value_x -> normalised(value_y)
        time, (syncd_x_value, synced_y_value) = multi_timesinc([(time_x, value_x), (time_y, value_y)])
        fun, parameters = fit_poly_fun_and_print(syncd_x_value, synced_y_value, f"{name_x} to {name_y}", exponents=[0, 1])
        print(f"fit_poly.parameters on {name_x}=", parameters)

        # transform from normalised values back to original
        if normalise:
            syncd_x_value = syncd_x_value*(maxx-minx)+minx
            synced_y_value = synced_y_value*(maxy-miny)+miny
            tmp = (maxy-miny)/(maxx-minx)
            offset, fac = parameters
            parameters = np.array([offset*(maxy-miny)+miny-fac*tmp*minx, fac*tmp])
        plot_and_save(visout_praefix+f"fit_lin_{name_x}_to_{name_y}", x_in=time, ys=[synced_y_value, fun(syncd_x_value, parameters)], names=[name_y, f"{parameters[1]:.2E}*{name_x}+{parameters[0]:.2E}"], avgs=False)
        #fig, axe = plt.subplots()
        #axe.set_title(f"{visout_praefix}fit_lin_{k_name}_to_{name}")
        #axe.plot(time, syncd_data_value, label=name)
        #axe.plot(sdd[k+x_ending], smothing(sdd[k+x_ending], est_data_value, 1), label=f"{parameters[1]:.2E}*{k_name}+{parameters[0]:.2E}")
        ##axe.plot(time, fac*synced_k_value+offset, label=f"{fac:.2E}*{k_name}+{offset:.2E}")
        #axe.legend()
        #axe.grid()
        #fig.show()


def plot_gnss_vs_droneview():
    sdd = read_csv("merged_rundata_csv/alldata_2022_12_17-14_43_59_id3.csv")
    lat_gnns_time = sdd["GNSS_latitude_UsbFlRec"+x_ending]
    lat_gnss_value = sdd["GNSS_latitude_UsbFlRec"]*np.pi/180
    long_gnss_time = sdd["GNSS_longitude_UsbFlRec"+x_ending]
    long_gnss_val = sdd["GNSS_longitude_UsbFlRec"]*np.pi/180
    gnss_heading_time = sdd["GNSS_heading_UsbFlRec"+x_ending]
    gnss_heading_val = sdd["GNSS_heading_UsbFlRec"]*np.pi/180
    time_som_secodns = 100
    for i in range(len(lat_gnns_time)):
        if lat_gnns_time[i] > time_som_secodns:
            lat_gnns_time = lat_gnns_time[i:]
            lat_gnss_value = lat_gnss_value[i:]
            break
    for i in range(len(long_gnss_time)):
        if long_gnss_time[i] > time_som_secodns:
            long_gnss_time = long_gnss_time[i:]
            long_gnss_val = long_gnss_val[i:]
            break
    for i in range(len(gnss_heading_time)):
        if gnss_heading_time[i] > time_som_secodns:
            gnss_heading_time = gnss_heading_time[i:]
            gnss_heading_val = gnss_heading_val[i:]
            break
    poi_true_gps_positions_radiants, carpos = true_pos_from_droneimg_pxpos()
    carposkeys = list(carpos.keys())
    carposkeys.sort()
    carpos_dv_time = [t2ssdt(drone2t(carposkeys[i]))+2 for i in range(len(carposkeys))]  # carpos_dv_time[i] = carposkeys[i]/25+26.5
    #carpos_dv_val = [gps_util.average(carpos[k]) for k in carposkeys]
    carpos_dv_val = [gps_util.carposs_to_gnsspos(carpos[k]) for k in carposkeys]
    carpos_heading_dv_val = [gps_util.carposs_to_heading(carpos[k]) for k in carposkeys]

    for i in range(len(carpos_dv_time)):
        if carpos_dv_time[i] > time_som_secodns:
            carpos_dv_time = carpos_dv_time[i:]
            carpos_dv_val = carpos_dv_val[i:]
            break
    carposlat_val = [lat for (lat, long) in carpos_dv_val]
    carposlong_val = [long for (lat, long) in carpos_dv_val]

    pos_time, (gnss_lat, gnss_long, tp_lat, tp_long) = multi_timesinc([(lat_gnns_time, lat_gnss_value), (long_gnss_time, long_gnss_val), (carpos_dv_time, carposlat_val), (carpos_dv_time, carposlong_val)])
    plot_and_save("lats", x_in=pos_time, ys=[gnss_lat, tp_lat], names=["gnss_lat", "tp_lat"], save_dir="vis_out/lattitudes.png")
    plot_and_save("lats", x_in=pos_time, ys=[gnss_long, tp_long], names=["gnss_long", "tp_long"], save_dir="vis_out/longitudes.png")
    dists_position = [gps_util.gps_to_dist((gnss_lat[i], gnss_long[i]), (tp_lat[i], tp_long[i])) for i in range(len(pos_time))]
    print("avg_dists_position[0.5:] =", np.average(dists_position[int(0.5*len(dists_position)):]))
    plot_and_save("dist(true_gnss_pos, gnss_pos)", x_in=pos_time, ys=[dists_position], save_dir="vis_out/dist_position_gnss_dv.png")
    heading_time, (syncd_heading_dv, syncd_heading_gnss) = multi_timesinc([(carpos_dv_time, carpos_heading_dv_val), (gnss_heading_time, gnss_heading_val)])
    dists_heading = [to_range(syncd_heading_dv[i]-syncd_heading_gnss[i]) for i in range(len(heading_time))]
    print("avg_dists_heading[0.5:] =", np.average(dists_heading[int(0.5*len(dists_heading)):]))
    plot_and_save("dist(heading_dv, heading_gnss)", x_in=heading_time, ys=[dists_heading], save_dir="vis_out/dist_heading_gnss_dv.png")
    plot_colorgradient_carpath("gnss_position", gnss_lat, gnss_long, pos_time)

    # plot heading from gps and droneview against each other
    fig, axe = plt.subplots()
    axe.set_title("heading")
    axe.plot(carpos_dv_time, carpos_heading_dv_val, label="heading from drone")
    axe.plot(gnss_heading_time, gnss_heading_val, label="heading from gps")
    axe.legend()
    axe.grid()
    fig.show()
    fig.savefig("vis_out/headings.png")


def print_hz(name: str, time: [seconds], value: [float]) -> None:
    # print the average distance (in seconds) between two consecetive measurements.
    print(f"{name}: length (seconds)/number of measurements {(max(time)-min(time))/len(time)}")
    seconds_til_new_mes = []
    oldv = value[0]
    oldt = time[0]
    seconds_til_new_time = []
    ot = time[0]
    for (t, v) in zip(time, value):
        if t != ot:
            seconds_til_new_time.append(t-ot)
            ot = t
        if v != oldv:
            seconds_til_new_mes.append(t-oldt)
            oldv = v
            oldt = t
    print(f"{name}: time [s] between different value values average = {np.average(seconds_til_new_mes)}, median = {np.median(seconds_til_new_mes)}, number_of_different_values = {len(seconds_til_new_mes)}")
    #print(f"{name}: time [s] between different time vlaues average = {np.average(seconds_til_new_time)}, median = {np.median(seconds_til_new_time)}")

    #fig, axe = plt.subplots()
    #axe.plot(np.linspace(0, 1, len(seconds_til_new_mes)), seconds_til_new_mes, label="new mes")
    ##axe.plot(np.linspace(0, 1, len(seconds_til_new_time)), seconds_til_new_time, label="new time")
    #axe.grid()
    #axe.legend()
    #axe.set_title(f"{name}: avg time between different mes/time values")
    #axe.set_ylabel("timediff in seconds")
    #axe.set_xlabel(f"np.linspace(0, 1, {len(seconds_til_new_mes)}|{len(seconds_til_new_time)})")
    #fig.show()


def main():
    #sdd = read_csv("merged_rundata_csv/alldata_2022_12_17-14_43_59_id3.csv")
    #visualise_data(sdd)
    #plot_gnss_vs_droneview()
    #test_slamfrontend()
    #custom_pnp_find_parameters()
    #custom_pnp_validate()
    #test_Slam()
    #get_true_carstate()
    custom_pnp_validate()


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
    multi_timesinc()
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
    plot_colorgradient_carpath()
    plot_and_save()
    visualise_data()
    show_sensorlogs()
    plot_cones()
    plot_on_googlemaps()
    plot_gnss_vs_droneview()
    # manuel
    get_laptoptimes_for_camFrames()
    get_car_moves_starttime_from_sensors()
    averegae_diff()
    true_pos_from_droneimg_pxpos()
    custom_pnp_find_parameters()
    custom_pnp_validate()
    print_synced_frame()
    get_synced_frame()
    # testing for cpp implementation
    custom_PnP()
    visual_pipeline()
    main()

# 16bit_int.MAX_VALUE = 32767
# U_I_converter_max = 42.46
# motorkonstante = 0.83
# Trq = Converter_L_Torque_Out_UsbFlRec*U_I_converter_max/(16bit_int.MAX_VALUE*np.sqrt(2))*motorkonstante
# RPM = Converter_L_RPM_Actual_Filtered_UsbFlRec*6000/32767