import pathlib

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

#project_root_path = pathlib.Path("C:/Users/johan/PycharmProjects/FS_autonomous_pyutil/")
project_root_path = pathlib.Path("C:/Users/Idefix/PycharmProjects/eTeam_pyutil/")
def getType(x):
    """
    :param x:
    :return:
    a string containing information about the type of x
    """
    deliminating_chars = {"list": ('[', ';', ']'), "tupel": ('<', ';', '>'), "dict": ('{', ';', '}')}
    name = type(x).__name__
    if name == 'list':
        if len(x) == 0:
            return "[0:Nix]"
        return '['+str(len(x))+":"+getType(x[0])+']'  # assumes all element of the list have the same type
    if name == "str":
        return "str_"+str(len(x))
    if name == 'tuple':
        r = '<'+str(len(x))+":"
        for i in x:
            r += getType(i)+'; '
        return r[:-2] + '>'
    if name == 'dict':
        r = "{"+str(len(x.keys()))+":"
        if len(x.keys()) < 10:
            for key in x.keys():
                r += str(key)+": "+getType(x[key])+"; "  # contains more information
        else:
            # assume all keys and values have same type
            key = list(x.keys())[0]
            r += getType(key)+": "+getType(x[key])
        r = r[:-1]+" }"
        return r
    if name == 'dict_keys':
        return name+"("+str(set(getType(k) for k in x))+")"
    if name == 'ndarray':
        return 'ndarray('+str(x.shape)+': '+(str(x.dtype) if x.size > 0 else "Nix")+')'
    if name == 'BatchDataset':
        return str(name)+" : "+str(len(x))
    if name == "Dataset":  # Dataloader.Dataset
        return "Dataset(name=\""+str(x.name)+"\", "+str(x.imgsize)+"->"+str(x.glsize)+" )"
    if name == "TensorSliceDataset":  # tf.data.Dataset
        for elem in x:
            return "TensorSliceDataset "+str(len(x))+": "+getType(elem)
    if name in ["KerasTensor", "Tensor", "EagerTensor"]:
        return str(name)+"("+str(x.shape)+":"+str(x.dtype)+")"
    return name

second = float
def smothing(time: [second], values: [float], t: second) -> [float]:
    assert len(time) == len(values)
    #smothing(time=sdd["GNSS_speed_over_ground_UsbFlRec"+x_ending], values=sdd["GNSS_speed_over_ground_UsbFlRec"], t=3)
    n = int(t * len(time) / max(time))  # number of timestamps/length in seconds = number of timestamps per seconds, assuming timestamps are evenly spaced.
    # doing average over {s} seconds checking for unevenly spaced timestamps was to time-consuimg to run on this pc.
    padded_y = np.array([values[0]] * (n // 2) + list(values) + [values[-1]] * (n - n // 2 - 1))
    y_avg = np.convolve(padded_y, np.ones(n) / n, mode='valid')
    return y_avg


seconds = float
def plot_and_save(name: str, x_in: [seconds], ys: [[float]], save_dir=None, names=None, avgs=True):
    """
    plots y (dependent variable) against x. with 5-second average of y, mean an total average of y and saves the plot to save_dir.
    removes outliers from y and adds name as name of plot and label for y data and label for y-axis
    """
    for y in ys:
        assert len(x_in) == len(y)
    #print(f"plot_and_save(name = {name}, save_dir={save_dir}): ")
    # if any value in y is more than ?double? the distance to mean than the max of the lower ?90?%, remove them
    mean = np.median(ys)
    fig, (axe) = plt.subplots(1)
    axe.set_title(name)
    if names is None:
        names = ["" for _ in range(len(ys))]
    else:
        assert len(ys) == len(names)
    for (name_suffix, y) in zip(names, ys):
        x = list(x_in)
        dist = [abs(ty - mean) for ty in y]
        dist.sort()
        dist = 2 * dist[int(0.9 * len(dist))]  # keep all
        #print("mean = ", mean, ", dist = ", dist)
        if dist != 0:
            old_length = len(x)
            tmp = [(tx, ty) for (tx, ty) in zip(x, y) if abs(ty - mean) < dist]
            print(f"removing outliers changed number of data points in {name}.{name_suffix} from {old_length} to {len(tmp)} ({100 * len(tmp) / old_length}% of original)")
            x = [tx for (tx, ty) in tmp]
            y = [ty for (tx, ty) in tmp]
            del tmp
        # plot unfiltered data
        axe.plot(x, y, label=name_suffix)
        if avgs:
            s=5
            axe.plot(x, smothing(x, y, s), "--", color="green", linewidth=0.5, label="average " + str(s) + "s "+name_suffix)
            # plot average
            axe.plot(x, [np.average(y)] * len(y), "-.", color="black", linewidth=0.5, label="total average "+name_suffix)
            axe.plot(x, [mean] * len(y), "--", color="black", linewidth=0.5, label="mean "+name_suffix)

    axe.set_xlabel("time in seconds")
    axe.set_ylabel(name)
    axe.legend()
    axe.grid()
    if save_dir is None:
        save_dir = "vis_out/"+name
    fig.savefig(save_dir)

def to_range(x):
    while x > np.pi:
        x -= 2*np.pi
    while x < -np.pi:
        x += 2*np.pi
    return x


def mes_to_time_range(mes_time, mes_value, start=None, stop=None):
    if start is not None:
        for i in range(len(mes_time)):
            if mes_time[i] > start:
                mes_time = mes_time[i:]
                mes_value = mes_value[i:]
                break
    if stop is not None:
        for i in range(len(mes_time)-1, 0, -1):
            if mes_time[i] < stop:
                mes_time = mes_time[:i]
                mes_value = mes_value[:i]
                break
    return mes_time, mes_value


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
    i_start = max(0, [tmp_index for tmp_index, tmp_time in enumerate(x) if tmp_time > t][0]-1)
    for i in range(i_start, len(x)):
        if t == x[i]:
            return y[i], i
        if t < x[i]:
            w0 = abs(t-x[i-1])
            w1 = abs(x[i]-t)
            return (w1*y[i-1]+w0*y[i])/(w0+w1), i-1 if w0 < w1 else i


def multi_timesinc(data: [([seconds], [float])]) -> ([seconds], [[float]]):
    # syncd_time, (syncd_a_value, syncd_b_value, syncd_c_value, ...) = multi_timesinc([(a_time, a_value), (b_time, b_value), (c_time, c_value), ...])
    for (data_time, data_value) in data:
        assert len(data_time) == len(data_value)
    indexe = [0 for _ in range(len(data))]
    r = [[] for _ in range(len(data))]
    timesteps = []
    for (data_time, data_value) in data:
        timesteps += list(data_time)
    timesteps = np.array(sorted(timesteps))
    for t in timesteps:
        for indexei in range(len(indexe)):
            in_data_time = data[indexei][0]
            in_data_value = data[indexei][1]
            while indexe[indexei]+1 < len(in_data_time) and in_data_time[indexe[indexei]+1] <= t:
                indexe[indexei] += 1
            ii = min(indexe[indexei], len(in_data_time)-2)
            # timesteps[ii] <= t < timesteps[ii+1]
            if in_data_time[ii] == t:
                r[indexei].append(float(in_data_value[ii]))
            elif ii+1 >= len(in_data_time) or in_data_time[ii] >= timesteps[-1]:
                indexe[indexei] = len(in_data_time)-2
                r[indexei].append(in_data_value[-1])
            elif in_data_time[ii] > t or in_data_time[ii] <= timesteps[0]:  # (in_data_time[ii] > t) iff (ii=0 and min(in_data_time) > t
                r[indexei].append(in_data_value[0])
            else:
                w0 = abs(in_data_time[ii] - t)
                w1 = abs(in_data_time[ii + 1] - t)
                r[indexei].append((w1*in_data_value[ii] + w0*in_data_value[ii+1]) / (w0 + w1))
    assert len(r) == len(data)
    for r_value in r:
        assert len(timesteps) == len(r_value)
    return np.array(timesteps), [np.array(data_value) for data_value in r]


def fit_poly_fun_and_print(in_x, out_true, name, exponents=[-1, 0, 1, 2]):
    # returns fun, parameters, so that fun(in_x, parameters) = out_true
    assert len(in_x) == len(out_true)
    in_x = np.array(in_x)
    out_true = np.array(out_true)
    #fo = np.array([1, 0, 0, 0])
    fo = np.array([(1 if i == 1 else 0) for i in exponents])

    def fun(x, t):
        return sum([t[i]*x**v for (i, v) in enumerate(exponents)])  # np.sum returns deep sum, that means the result is float, even if x is np.array
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
    est_value = fun(in_x, parameters)
    print(f"{name}.est_from ", in_x)
    print(f"{name}.fun(est_from) = ", est_value)
    print(f"{name}.out_true  = ", out_true)
    print(f"parameters[{name}] = {parameters}")
    print(f"diff_{name} = {np.sqrt(np.sum((est_value-out_true)**2))/len(in_x)}\n")
    return fun, parameters

