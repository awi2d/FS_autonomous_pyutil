
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