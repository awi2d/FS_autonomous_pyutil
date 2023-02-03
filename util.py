import matplotlib.pyplot as plt
import numpy as np

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


seconds = float
def plot_and_save(name: str, x_in: [seconds], ys: [[float]], save_dir=None, names=None):
    """
    plots y (dependent variable) against x. with 5-second average of y, mean an total average of y and saves the plot to save_dir.
    removes outliers from y and adds name as name of plot and label for y data and label for y-axis
    """
    for y in ys:
        assert len(x_in) == len(y)
    print(f"plot_and_save(name = {name}, save_dir={save_dir}): ")
    # if any value in y is more than ?double? the distance to mean than the max of the lower ?90?%, remove them
    mean = np.median(ys)
    fig, (axe) = plt.subplots(1)
    axe.set_title(name)
    if names is None:
        names = ["" for _ in range(len(ys))]
    for (name_suffix, y) in zip(names, ys):
        x = list(x_in)
        dist = [abs(ty - mean) for ty in y]
        dist.sort()
        dist = 2 * dist[int(0.9 * len(dist))]  # keep all
        print("mean = ", mean, ", dist = ", dist)
        if dist != 0:
            old_length = len(x)
            tmp = [(tx, ty) for (tx, ty) in zip(x, y) if abs(ty - mean) < dist]
            print("removing outliers changed number of data points from ", old_length, " to ", len(tmp), "(",
                  100 * len(tmp) / old_length, "% of original)")
            x = [tx for (tx, ty) in tmp]
            y = [ty for (tx, ty) in tmp]
            del tmp
        # plot unfiltered data
        axe.plot(x, y, label=name_suffix)
        s = 5
        n = int(s * len(x) / max(x))  # number of timestamps/length in seconds = number of timestamps per seconds, assuming timestamps are evenly spaced.
        # doing average over {s} seconds checking for unevenly spaced timestamps was to time-consuimg to run on this pc.
        padded_y = np.array([y[0]] * (n // 2) + list(y) + [y[-1]] * (n - n // 2 - 1))
        y_avg = np.convolve(padded_y, np.ones(n) / n, mode='valid')
        axe.plot(x, y_avg, "--", color="green", linewidth=0.5, label="average " + str(s) + "s "+name_suffix)
        # plot average
        axe.plot(x, [np.average(y)] * len(y), "-.", color="black", linewidth=0.5, label="total average "+name_suffix)
        axe.plot(x, [mean] * len(y), "--", color="black", linewidth=0.5, label="mean "+name_suffix)

    axe.set_xlabel("time in seconds")
    axe.set_ylabel(name)
    axe.legend()
    axe.grid()
    if save_dir is None:
        save_dir = name
    fig.savefig(save_dir)
