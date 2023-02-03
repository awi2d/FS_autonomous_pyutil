import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from os import listdir
from os.path import isfile, join
from util import getType, plot_and_save

data = [{}]  # data[k] = {"time": value, "true_X": value, ...}
time = []


def plot_timepoints(name: str, labels):
    global data, time
    #points = [[d[name] for name in labels] for d in data]
    #plot_custompoints(name, points, labels)
    plot_and_save(name, time, [[d[n] for d in data] for n in labels], save_dir=f"kf_logs_visout/{name}.png", names=labels)


def plot_custompoints(name: str, points, labels):
    tmp = [min(t) for t in points]
    tmpm = [max(t) for t in points]
    tmp.sort()
    tmpm.sort()
    y_bounds = [tmp[2]-0.01, tmpm[-3]+0.01]
    print("y_bounds = ", y_bounds)
    ax = plt.gca()
    ax.set_ylim(y_bounds)
    for i in range(len(points[0])):
        plt.plot(time, [p[i] for p in points], label=labels[i], linewidth=0.5)
    plt.title(name)
    plt.legend(loc="upper left")
    plt.show()


def string2float(s: str):
    s = s.replace(" ", "")
    if s == "False":
        return 0
    elif s == "True":
        return 1
    else:
        try:
            return float(s)
        except ValueError:
            #print("string ", s, "could not be convertet to float.")
            return None

def readfile(filename: str, output=None):
    data = []
    with open(filename) as file:
        filecontent = file.readlines()
        if output is None:
            print(filecontent[0].replace("\n", ""))
        else:
            output["header"] = filecontent[0].replace("\n", "")
        coloumNames = filecontent[1].replace("\t", ", ").replace(" ", "").replace("\n", "").split(",")
        #print("coloumNames = \n", coloumNames, "\n")
        for line in filecontent[2:]:
            data_entry = {}
            ls = line.split(",")
            ls = [string2float(val) for val in ls]
            ls = [ls_entry for ls_entry in ls if ls_entry is not None]
            for (name, value) in zip(coloumNames, ls):
                data_entry[name] = value
            data.append(data_entry)
    return data


def print_totalerrors():
    mypath = "C:/Users/Idefix/CLionProjects/velocity_estimator/logs/"
    filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print("all filnames = ", filenames)
    errors = {}
    filedesc = {}
    for filename in filenames:
        tmp = {}
        lok_data = readfile(mypath+filename, tmp)
        filedesc[filename] = tmp["header"]
        if lok_data[-1]["time"] >= 45.34:
            error = [(abs(d["true_X"]-d["state_X"]), abs(d["true_Y"]-d["state_Y"]), abs(d["true_Yaw"]-d["state_Yaw"]), abs(d["true_vx"]-d["state_vx"]), abs(d["true_vy"]-d["state_vy"]), abs(d["true_yawrate"]-d["state_yawrate"]), abs(d["true_ax"]-d["state_ax"]), abs(d["true_ay"]-d["state_ay"]), abs(d["true_srl"]-d["state_srl"]), abs(d["true_srr"]-d["state_srr"])) for d in lok_data]
            error = [sum([e[i] for e in error])/lok_data[-1]["time"] for i in range(len(error[0]))]
        else:
            error = [np.inf for _ in range(10)]
        errors[filename] = error
    min = [filenames[0] for _ in range(len(errors[filenames[0]]))]
    for filename in errors.keys():
        print(f"Errors  of file {filename}= {errors[filename]}\n")
        for i in range(len(errors[filename])):
            if errors[filename][i] < errors[min[i]][i]:
                min[i] = filename
    if len(set(min)) == 1:
        print(f"{min[0]} is best in every way")
    else:
        print("min errors: ", min)
        coloumnames = ["X", "Y", "Yaw", "vx", "vy", "yawrate", "ax", "ay", "srl", "srr"]
        print("min errors: ", [coloumnames[i]+": "+filedesc[min[i]][len("logs from velocity_estimator.kf run at"):] for i in range(len(coloumnames))])
        count = [0]*len(set(min))
        f2i = list(set(min))
        for fname in min:
            count[f2i.index(fname)] += 1
        tmp = []
        for i in range(len(count)):
            tmp.append((filedesc[f2i[i]], count[i]))
        tmp.sort(key=lambda x: -x[1])
        print(tmp)






def linear_with_offset(x, a, b):
    return a*x+b
def fit_linear(x_name, y_name):
    """
    :return:
    prints the parameters popt[0], popt[1], so that popt[0]*x+popt[1] = y holds as good as possible
    """
    global data
    x_data = np.array([d[x_name] for d in data])
    y_data = np.array([d[y_name] for d in data])
    popt, pcov = optimize.curve_fit(linear_with_offset, x_data, y_data)
    #print(f"best fit is {x_name}*{popt[0]}+{popt[1]} = {y_name}")
    #print("popt = ", popt)
    #print("pconv = ", pcov)
    #print(f"integral( abs({popt[0]}*{x_name}+{popt[1]} - {y_name})) = {sum([abs(popt[0]*x+popt[1]-y) for (x, y) in zip(x_data, y_data)])}\n")

    return popt
    #true_ax = u_prdax*0.07906001-4.9776714
    #true_ay = u_prdax*0.01151266-1.44016067
def linear2_with_offset(x, a, b, c, d):
    return a*x[0]*x[1]+b*x[0]+c*x[1]+d
def fit_2linear(x_names, y_name):
    global data
    x_data = np.array([(d[x_names[0]], d[x_names[1]]) for d in data])
    y_data = np.array([d[y_name] for d in data])
    popt, pcov = optimize.curve_fit(linear2_with_offset, x_data, y_data)
    return popt
def quadn_with_offset(n):
    r_str = f"def r_{n}(x"
    rr_str = "\n\treturn "
    out_str = "\"{output} = "
    tups = [(x, y) for x in range(n) for y in range(n) if x >= y]
    name_number = 0
    for i in range(len(tups)):
        r_str += ", a_"+str(name_number)
        rr_str += f"a_{name_number}*x[{tups[i][0]}]*x[{tups[i][1]}] + "
        out_str += "{popt["+str(name_number)+"]}*{input["+str(tups[i][0])+"]}*{input["+str(tups[i][1])+"]} + "
        name_number += 1
    for i in range(n):
        r_str += ", a_"+str(name_number)
        rr_str += f"a_{name_number}*x[{i}] + "
        out_str += "{popt["+str(name_number)+"]}*{input["+str(i)+"]} + "
        name_number += 1
    r_str += ", a_"+str(name_number)
    rr_str += f"a_{name_number}"
    out_str += "{popt["+str(name_number)+"]}"
    name_number += 1
    r_str = r_str+"):"+rr_str
    print(r_str)
    print(out_str+"\"")

def r_0(x, a_0):
    return a_0
def r_1(x, a_0, a_1, a_2):
    return a_0*x[0]*x[0] + a_1*x[0] + a_2
def r_2(x, a_0, a_1, a_2, a_3, a_4, a_5):
    return a_0*x[0]*x[0] + a_1*x[1]*x[0] + a_2*x[1]*x[1] + a_3*x[0] + a_4*x[1] + a_5
def r_3(x, a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9):
    return a_0*x[0]*x[0] + a_1*x[1]*x[0] + a_2*x[1]*x[1] + a_3*x[2]*x[0] + a_4*x[2]*x[1] + a_5*x[2]*x[2] + a_6*x[0] + a_7*x[1] + a_8*x[2] + a_9
def r_4(x, a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10, a_11, a_12, a_13, a_14):
    return a_0*x[0]*x[0] + a_1*x[1]*x[0] + a_2*x[1]*x[1] + a_3*x[2]*x[0] + a_4*x[2]*x[1] + a_5*x[2]*x[2] + a_6*x[3]*x[0] + a_7*x[3]*x[1] + a_8*x[3]*x[2] + a_9*x[3]*x[3] + a_10*x[0] + a_11*x[1] + a_12*x[2] + a_13*x[3] + a_14
def r_5(x, a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10, a_11, a_12, a_13, a_14, a_15, a_16, a_17, a_18, a_19, a_20):
    return a_0*x[0]*x[0] + a_1*x[1]*x[0] + a_2*x[1]*x[1] + a_3*x[2]*x[0] + a_4*x[2]*x[1] + a_5*x[2]*x[2] + a_6*x[3]*x[0] + a_7*x[3]*x[1] + a_8*x[3]*x[2] + a_9*x[3]*x[3] + a_10*x[4]*x[0] + a_11*x[4]*x[1] + a_12*x[4]*x[2] + a_13*x[4]*x[3] + a_14*x[4]*x[4] + a_15*x[0] + a_16*x[1] + a_17*x[2] + a_18*x[3] + a_19*x[4] + a_20
quad_functions = [r_0, r_1, r_2, r_3, r_4, r_5]
def r_str(output, popt, input):
    l = len(input)
    if l == 0:
        return f"{output} = {popt[0]}"
    elif l == 1:
        return f"{output} = {popt[0]}*{input[0]}*{input[0]} + {popt[1]}*{input[0]} + {popt[2]}"
    elif l == 2:
        return f"{output} = {popt[0]}*{input[0]}*{input[0]} + {popt[1]}*{input[1]}*{input[0]} + {popt[2]}*{input[1]}*{input[1]} + {popt[3]}*{input[0]} + {popt[4]}*{input[1]} + {popt[5]}"
    elif l == 3:
        return f"{output} = {popt[0]}*{input[0]}*{input[0]} + {popt[1]}*{input[1]}*{input[0]} + {popt[2]}*{input[1]}*{input[1]} + {popt[3]}*{input[2]}*{input[0]} + {popt[4]}*{input[2]}*{input[1]} + {popt[5]}*{input[2]}*{input[2]} + {popt[6]}*{input[0]} + {popt[7]}*{input[1]} + {popt[8]}*{input[2]} + {popt[9]}"
    elif l == 4:
        return f"{output} = {popt[0]}*{input[0]}*{input[0]} + {popt[1]}*{input[1]}*{input[0]} + {popt[2]}*{input[1]}*{input[1]} + {popt[3]}*{input[2]}*{input[0]} + {popt[4]}*{input[2]}*{input[1]} + {popt[5]}*{input[2]}*{input[2]} + {popt[6]}*{input[3]}*{input[0]} + {popt[7]}*{input[3]}*{input[1]} + {popt[8]}*{input[3]}*{input[2]} + {popt[9]}*{input[3]}*{input[3]} + {popt[10]}*{input[0]} + {popt[11]}*{input[1]} + {popt[12]}*{input[2]} + {popt[13]}*{input[3]} + {popt[14]}"
    elif l == 5:
        f"{output} = {popt[0]}*{input[0]}*{input[0]} + {popt[1]}*{input[1]}*{input[0]} + {popt[2]}*{input[1]}*{input[1]} + {popt[3]}*{input[2]}*{input[0]} + {popt[4]}*{input[2]}*{input[1]} + {popt[5]}*{input[2]}*{input[2]} + {popt[6]}*{input[3]}*{input[0]} + {popt[7]}*{input[3]}*{input[1]} + {popt[8]}*{input[3]}*{input[2]} + {popt[9]}*{input[3]}*{input[3]} + {popt[10]}*{input[4]}*{input[0]} + {popt[11]}*{input[4]}*{input[1]} + {popt[12]}*{input[4]}*{input[2]} + {popt[13]}*{input[4]}*{input[3]} + {popt[14]}*{input[4]}*{input[4]} + {popt[15]}*{input[0]} + {popt[16]}*{input[1]} + {popt[17]}*{input[2]} + {popt[18]}*{input[3]} + {popt[19]}*{input[4]} + {popt[20]}"

def fit_n(x_names, y_name):
    global data
    if len(x_names) >= len(quad_functions):
        quadn_with_offset(len(x_names))
        exit(0)
    x_data = [np.array([d[xn] for d in data]) for xn in x_names]
    y_data = np.array([d[y_name] for d in data])
    fun = quad_functions[len(x_names)]
    popt, pcov = optimize.curve_fit(fun, x_data, y_data)
    return popt, fun, x_data

def find():
    states = ["state_X", "state_Y", "state_Yaw", "state_vx", "state_vy", "state_yawrate", "state_ax", "state_ay", "state_srl", "state_srr"]
    inputs_gps = ["gps_pred_X", "gps_pred_Y", "gps_pred_Yaw", "gps_pred_vx", "gps_pred_yawrate"]
    inputs_imu = ["imu_pred_ax", "imu_pred_ay", "imu_pred_yawrate"]
    inputs_rh = ['rh_pred_vx', 'rh_pred_srl', 'rh_pred_srr', 'rh_pred_yawrate']
    inputs_u = ['u_pred_yawrate', 'u_pred_ax']
    inputs = states+inputs_gps+inputs_imu+inputs_rh+inputs_u
    print_later = set()
    for out in states:
        for input in ((inputs[x], inputs[y]) for x in range(len(inputs)) for y in range(len(inputs)) if x > y and inputs[y] != out and inputs[x] != out):
            popt, fun, x_data = fit_n(input, out)
            pred_error = sum(abs(fun(x_data, *popt)-np.array([d["true"+out[5:]] for d in data])))
            state_error = sum([abs(d[out]-d["true"+out[5:]]) for d in data])
            if pred_error < state_error:
                assert len(input) == 2 and "rewrite check that fun(x_data, *popt) is dependent on all elements of x_data <=> for every element of x_data at least on coefficent in popt is > 0 to work with arbitrary sizes of len(input)"
                zero = 0.1
                if abs(popt[0]) < zero and abs(popt[1]) < zero and abs(popt[3]) < zero:
                    print_later.add((out, input[1]))
                if abs(popt[1]) < zero and abs(popt[2]) < zero and abs(popt[4]) < zero:
                    print_later.add((out, input[0]))
                if abs(popt[1]) > zero or ((abs(popt[0]) > zero or abs(popt[3]) > zero) and (abs(popt[2]) > zero or abs(popt[4]) > zero)):
                    print(f"using {r_str(out, popt, input)} is better with {(state_error-pred_error)/state_error}")
            #else:
            #    print(f" - using {popt[0]}*{input}+{popt[1]} is worse then state{out[5:]}  with {pred_error} > {state_error}")
    for (out, inp) in print_later:
        print(f" - {out} = f({inp}) is better.")

def test_dX():
    # dX is somewhat following true_vx, but currently state_vx is way closer to true_vx than dX
    dX = [0]*len(time)
    oXi = 0
    v = 0
    for i in range(len(time)):
        if data[i]["gps_c"] == 1 and abs(time[i]-time[oXi]) > 0.5:
            v = np.sqrt((data[i]["gps_pred_X"]-data[oXi]["gps_pred_X"])**2+(data[i]["gps_pred_Y"]-data[oXi]["gps_pred_Y"])**2)/(time[i]-time[oXi])
            oXi = i
        dX[i] = v

    plot_custompoints("dX", [[dX[i], data[i]["true_vx"], data[i]["state_vx"]] for i in range(len(time))], ["dX", "true_vx", "state_vx"])


def run():
    #print_totalerrors()
    #exit()
    #min errors:  [
    # 'X:  2022-10-27 18:02:07, use_pred=True, use_gps=True, use_imu=True, use_rh=True, use_u=True, use_pseudo=True.',
    # 'Y:  2022-10-27 18:06:37, use_pred=False, use_gps=True, use_imu=False, use_rh=False, use_u=False, use_pseudo=False.',
    # 'Yaw:  2022-10-27 18:04:00, use_pred=True, use_gps=True, use_imu=True, use_rh=False, use_u=False, use_pseudo=True.',
    # 'vx:  2022-10-27 18:03:38, use_pred=False, use_gps=False, use_imu=True, use_rh=True, use_u=False, use_pseudo=True.',
    # 'vy:  2022-10-27 18:04:23, use_pred=False, use_gps=True, use_imu=False, use_rh=False, use_u=False, use_pseudo=True.',
    # 'yawrate:  2022-10-27 18:06:37, use_pred=False, use_gps=True, use_imu=False, use_rh=False, use_u=False, use_pseudo=False.',
    # 'ax:  2022-10-27 18:06:30, use_pred=False, use_gps=False, use_imu=True, use_rh=False, use_u=False, use_pseudo=False.',
    # 'ay:  2022-10-27 18:04:16, use_pred=False, use_gps=False, use_imu=True, use_rh=False, use_u=False, use_pseudo=True.',
    # 'srl:  2022-10-27 18:04:16, use_pred=False, use_gps=False, use_imu=True, use_rh=False, use_u=False, use_pseudo=True.',
    # 'srr:  2022-10-27 18:04:30, use_pred=True, use_gps=False, use_imu=False, use_rh=False, use_u=False, use_pseudo=True.']

    #min errors:  [
    # 'X:  2022-10-27 23:40:41, use_pred=True, use_gps=True, use_imu=True, use_rh=True, use_u=True, use_pseudo=True.',
    # 'Y:  2022-10-27 23:42:44, use_pred=False, use_gps=True, use_imu=True, use_rh=False, use_u=False, use_pseudo=True.',
    # 'Yaw:  2022-10-27 18:04:00, use_pred=True, use_gps=True, use_imu=True, use_rh=False, use_u=False, use_pseudo=True.',
    # 'vx:  2022-10-27 18:03:38, use_pred=False, use_gps=False, use_imu=True, use_rh=True, use_u=False, use_pseudo=True.',
    # 'vy:  2022-10-27 23:43:00, use_pred=False, use_gps=True, use_imu=False, use_rh=False, use_u=False, use_pseudo=True.',
    # 'yawrate:  2022-10-27 18:06:37, use_pred=False, use_gps=True, use_imu=False, use_rh=False, use_u=False, use_pseudo=False.',
    # 'ax:  2022-10-27 18:06:30, use_pred=False, use_gps=False, use_imu=True, use_rh=False, use_u=False, use_pseudo=False.',
    # 'ay:  2022-10-27 23:43:31, use_pred=False, use_gps=False, use_imu=True, use_rh=True, use_u=True, use_pseudo=False.',
    # 'srl:  2022-10-27 18:04:16, use_pred=False, use_gps=False, use_imu=True, use_rh=False, use_u=False, use_pseudo=True.',
    # 'srr:  2022-10-27 23:45:03, use_pred=False, use_gps=False, use_imu=True, use_rh=False, use_u=False, use_pseudo=False.']

    global data, time
    filename = "kf_log2023-02-01_15-21-21.csv"
    filename = filename.replace(".csv", "")
    data = readfile("C:/Users/Idefix/CLionProjects/state_estimator/logs/"+filename+".csv")
    print("getType(data) =", getType(data))
    print("coloumnames = ", data[0].keys())
    #['time',
    # 'gps_c', 'imu_c', 'rh_ch', 'u_chd',
    # 'state_X', 'state_Y', 'state_Yaw', 'state_vx', 'state_vy', 'state_yawrate', 'state_ax', 'state_ay', 'state_srl', 'state_srr',
    # 'true_X', 'true_Y', 'true_Yaw', 'true_vx', 'true_vy', 'true_yawrate', 'true_ax', 'true_ay', 'true_srl', 'true_srr',
    # 'gps_pred_X', 'gps_pred_Y', 'gps_pred_Yaw', 'gps_pred_vx', 'gps_pred_yawrate',
    # 'imu_pred_ax', 'imu_pred_ay', 'imu_pred_yawrate',
    # 'rh_pred_vx', 'rh_pred_srl', 'rh_pred_srr', 'rh_pred_yawrate',
    # 'u_pred_yawrate', 'u_pred_ax']
    #data = readfile("C:/Users/Idefix/CLionProjects/velocity_estimator/sensordata/sim_data_slalom.txt")
    print("data.size = ", len(data), "x", len(data[0]))
    time = [d["time"] for d in data]

    #plot state-true-pred graphs for all variables
    plot_timepoints("xs_"+filename, ["true_X", "state_X", "gps_pred_X"])
    plot_timepoints("ys_"+filename, ["true_Y", "state_Y", "gps_pred_Y"])
    plot_timepoints("yaws_"+filename, ["true_Yaw", "state_Yaw", "gps_pred_Yaw"])
    plot_timepoints("vxs_"+filename, ["true_vx", "state_vx", "gps_pred_vx", "rh_pred_vx"])
    plot_timepoints("vys_"+filename, ["true_vy", "state_vy"])
    plot_timepoints("yawrates_"+filename, ["true_yawrate", "state_yawrate", "gps_pred_yawrate", "imu_pred_yawrate", "rh_pred_yawrate", "u_pred_yawrate"])
    plot_timepoints("axs_"+filename, ["true_ax", "state_ax", "imu_pred_ax", "u_pred_ax"])
    plot_timepoints("ays_"+filename, ["true_ay", "state_ay", "imu_pred_ay"])
    plot_timepoints("srls_"+filename, ["true_srl", "state_srl", "rh_pred_srl"])
    plot_timepoints("srrs_"+filename, ["true_srr", "state_srr", "rh_pred_srr"])


if __name__ == "__main__":
    run()
