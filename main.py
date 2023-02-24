import os
import pathlib

import numpy as np
from scipy import optimize
import cv2
from pypylon import pylon
import datetime

import matplotlib.pyplot as plt
from util import getType

Basler_cameraMatrix = [[1.55902258e+03, 0, 1.03564443e+03], [0, 1.49628271e+03, 6.89322561e+02], [0, 0, 1]]  # no change by changing object points to reflect size of chessboard
Basler_dist = [[-2.42797289e-01, 9.77514487e-02, -8.00761502e-05, 5.61321688e-03, 1.08419697e-02]] # opencv: [k1, k2, p1, p2, k3]



def tireModel(x, D_R, C_R, B_R, D_F, C_F, B_F):
    (alpha_F, alpha_R, u) = x
    return D_R * np.sin(C_R * np.arctan(B_R * alpha_R)) + D_F * np.sin(C_F * np.arctan(B_F * alpha_F)) * np.cos(u)


def poly1(x, a):
    return x * a


def test_fitfunction_function(x_data, a, b, c):
    print("main.t: x = ", x_data)
    # return a*x*x+b*x+c
    x1, x2 = x_data
    x1 = np.array(x1)
    x2 = np.array(x2)
    return a * x1 + b * x2 + c


def working_example_of_fit_with_multiple_parameters():
    x_data = np.array([(i // 10, i % 10) for i in range(0, 100)])
    x_data = ([x[0] for x in x_data], [x[1] for x in x_data])
    # x_data = np.array(range(0, 100))
    y_data = np.array(test_fitfunction_function(x_data, 1, 2, 3))
    popt, pcov = optimize.curve_fit(test_fitfunction_function, x_data, y_data)
    print("popt = ", popt)
    print("pconv = ", pcov)


def calcTireModelConstants():
    data = []
    with open("C:\\Users\\Idefix\\Documents\\MATLAB_geschwindigkeitsschätzer\\dataToFitTireModelConstTo.txt") as file:
        for line in file.readlines():
            if len(line) > 0 and not line.startswith("#"):
                ls = line.split(" ")
                data.append((
                            float(ls[2].replace(",", "")), float(ls[5].replace(",", "")), float(ls[8].replace(")", "")),
                            float(ls[10].replace("\n", ""))))
            else:
                print("cant parse: " + line)
    alpha_F = np.array([x[0] for x in data])
    alpha_R = np.array([x[1] for x in data])
    u = np.array([x[2] for x in data])
    ay = np.array([x[3] for x in data])

    popt, pcov = optimize.curve_fit(tireModel, (alpha_F, alpha_R, u), ay, maxfev=10000)
    print("giving front and rear wheels seperate parameters:")
    print("popt = ", popt)
    print("pconv = ", pcov)
    # result when using ?: [-5200 -17 -215 -1110 14 -7.6] - diff=22.8, debug=252
    # result when using derivitative(vy) and true_ay*obj.masse+obj.x(4)*obj.x(6): [-15.6823615 -5.15563664 -0.751433806 132.315398 7.95216007  0.0201895185] - diff=79, debug=601
    # result when using derivitative(vy) and (true_ay+obj.x(4)*obj.x(6))*obj.masse: [-2573.52984103 -41.52497208 -124.98369276 2222.71584426 49.99265293 82.22496396] - diff=34, debug=404
    # result when using derivitative(vy) and (true_ay+obj.x(4)*obj.x(6))*obj.masse: [-1759.68449293 -98.26077213 -111.84684164 1369.25828994 102.35937689 131.04420662] - diff=56, debug=491
    # result when using ay and (true_ay+obj.x(4)*obj.x(6))*obj.masse: [-2835.4924782 -41.823415 -88.21428311 1888.60929823 45.9786971 88.70289315] - diff=39, debug=420
    # result when using ay and (true_ay+obj.x(4)*obj.x(6))*obj.masse: [-1530.67564531 -98.272704 -122.31176583 1495.44091065 98.36905497 125.13513188] - diff=50, debug=456
    # result when using ay and true_ay*obj.masse+obj.x(4)*obj.x(6): [107.21117871 13.22164521 158.81296809 -54.46844277 -22.15421784 -23.25711991] - diff=44.5, debug=506
    # result when using ay and true_ay*obj.masse+obj.x(4)*obj.x(6): [43.09586528 34.25102249 44.2889515 -80.20507453 -49.93118206 -54.86137651] - diff=90, debug=586
    # result when using ay and true_ay*obj.masse+obj.x(4)*obj.x(6) and trueX in linearesEinspurModell: [-698.536452 -37.2843675 -2.05775330 -9936.73219 -13.2051835 -0.259868486]  diff=42.5, debug=1013
    # result when using ay and true_ay*obj.masse+obj.x(4)*obj.x(6): [191.71885527 41.25519466 407.42965347 34.41841602 -27.49305764 -40.82026971] - diff=65.8, debug=496


def plot(name: str, points: [(float, float)]):
    # [(x-horizontal position, y-höhe)]
    fig, ax = plt.subplots()
    ax.scatter([x for (x, y) in points], [y for (x, y) in points])
    fig.suptitle(name)
    plt.show()


def whatever():
    data = []
    with open("C:\\Users\\Idefix\\Documents\\MATLAB_geschwindigkeitsschätzer\\KF_zurich_out.txt") as file:
        for line in file.readlines():
            ls = line.split(" , ")
            data.append((float(ls[0]), float(ls[1]), float(ls[2]), float(ls[3]), float(ls[4])))

    data = [(yawrate, ay, vy, whlspd_diff_F, whlspd_diff_R) for (yawrate, ay, vy, whlspd_diff_F, whlspd_diff_R) in data
            if abs(yawrate) > 0.01]
    plot("ay", [(ay, yawrate) for (yawrate, ay, vy, whlspd_diff_F, whlspd_diff_R) in data])
    plot("vy", [(vy, yawrate) for (yawrate, ay, vy, whlspd_diff_F, whlspd_diff_R) in data])
    plot("vdiff_F", [(whlspd_diff_F, yawrate) for (yawrate, ay, vy, whlspd_diff_F, whlspd_diff_R) in data])
    plot("vdiff_R", [(whlspd_diff_R, yawrate) for (yawrate, ay, vy, whlspd_diff_F, whlspd_diff_R) in data])
    exit(0)
    # print([(yawrate, ay, vy, whlspd_diff) for (yawrate, ay, vy, whlspd_diff) in data if abs(yawrate) > 0.01])
    print("yawrate/ay = ", np.mean([yawrate / ay for (yawrate, ay, vy, whlspd_diff_F, whlspd_diff_R) in data]))
    print("yawrate/vy = ", np.mean([yawrate / vy for (yawrate, ay, vy, whlspd_diff_F, whlspd_diff_R) in data]))
    print("yawrate/whlspd_diff_F = ",
          np.mean([yawrate / whlspd_diff_F for (yawrate, ay, vy, whlspd_diff_F, whlspd_diff_R) in data]))
    print("yawrate/whlspd_diff_R = ",
          np.mean([yawrate / whlspd_diff_R for (yawrate, ay, vy, whlspd_diff_F, whlspd_diff_R) in data]))
    exit(0)
    tmp = [yawrate for (yawrate, ay, vy, whlspd_diff_F, whlspd_diff_R) in data]
    popt, pcov = optimize.curve_fit(poly1, [ay for (yawrate, ay, vy, whlspd_diff_F, whlspd_diff_R) in data], tmp,
                                    maxfev=10000)
    print("ay:", popt, "\n", pcov)

    popt, pcov = optimize.curve_fit(poly1, [vy for (yawrate, ay, vy, whlspd_diff_F, whlspd_diff_R) in data], tmp,
                                    maxfev=10000)
    print("vy:", popt, "\n", pcov)

    popt, pcov = optimize.curve_fit(poly1, [whlspd_diff_F for (yawrate, ay, vy, whlspd_diff_F, whlspd_diff_R) in data],
                                    tmp, maxfev=10000)
    print("whlspd_diff_F:", popt, "\n", pcov)

    popt, pcov = optimize.curve_fit(poly1, [whlspd_diff_R for (yawrate, ay, vy, whlspd_diff_F, whlspd_diff_R) in data],
                                    tmp, maxfev=10000)
    print("whlspd_diff_R:", popt, "\n", pcov)


def calc_expetedvalue_variance():
    with open("C:/Users/Idefix/Documents/MATLAB_geschwindigkeitsschätzer/real_data_stationary_inside.txt") as file:
        data = []
        names = ["time"] + [name[11:] for name in file.readline().split("\t")]
        print("names = ", names)
        for line in file.readlines()[1:]:
            data.append([float(dp) for dp in line.split("\t")])

        for namei in range(len(names)):
            print("\n", names[namei])
            dataslice = [t[namei] for t in data]
            erwartungswert = sum(dataslice) / len(dataslice)
            print("E = ", erwartungswert)
            print("V = ", sum([(dp - erwartungswert) ** 2 for dp in dataslice]) / (len(dataslice) - 1))


radius = [40, 40, 40]  # gemessener wert
T = [40, 30, 20]  # gemessener wert
lenkwinkel = [0.05, 0.075, 0.1]  # gemessener wert
radstand = 2  # gemessener wert
masse = 375

def sk_Loss(x, debugout=False):
    # x = [radstand_p, l_v, l_h, sls, masse_p, EG] + k*[radius_p, T_p, lenkwinkel_p, vx, vy, ax, ay, yawrate, schwimmwinkel, slw_v, slw_h]
    #über mehrere Fahrten konstant: Radstand, l_v, l_h, sls, masse, EG
    (radstand_p, l_v, l_h, sls, masse_p, EG) = x[:6]
    residials = np.zeros(31*len(radius))
    for i in range(len(radius)):
        w = {"radius":(1/radius[i]), "radstand":(1/radstand), "T":(1/T[i]), "masse":(1/masse),
                   "lenkwinkel":(1/lenkwinkel[i]), "ay":100, "vy":100, "yawrate":T[i]/(1*np.pi), "slw_v":2/np.pi, "slw_h":2/np.pi, "0":100}
        (radius_p, T_p, lenkwinkel_p, vx, vy, ax, ay, yawrate, schwimmwinkel, slw_v, slw_h) = x[6+i*11:17+i*11]
        residials[i*31+0] = w["radius"]*(radius_p - radius[i])  # Radius = gemessener Radius des COG
        residials[i*31+1] = w["radstand"]*(radstand_p - radstand)  # radstand = gemessener Radstand
        residials[i*31+2] = w["T"]*(T_p - T[i])  # T = gemessene Zeit pro Kreisfahrt
        residials[i*31+3] = w["masse"]*(masse_p-masse)  # masse = gemessener wert
        residials[i*31+4] = w["lenkwinkel"]*(lenkwinkel_p - lenkwinkel[i])  # lenkwinkel = gemessener Lenkwinkel in radiants
        residials[i*31+5] = w["radstand"]*(radstand_p - l_v - l_h)  # radstand = l_v+l_h
        residials[i*31+6] = w["ay"]*(ay - 0)  # ay = 0
        residials[i*31+7] = w["radius"]*w["yawrate"]*(vx - radius_p * yawrate)  # vx = Radius*yawrate
        residials[i*31+8] = w["yawrate"]*(yawrate - 2 * np.pi / T_p)  # yawrate = 2pi/T
        residials[i*31+9] = w["ay"]*(ay - vx * yawrate * np.cos(schwimmwinkel))  # ay = v*yawrate*cos(schwimmwinkel)
        residials[i*31+10] = w["ay"]*(ay - vx ** 2 / radius_p)  # ay = v**2/R
        residials[i*31+11] = w["ay"]*(ay - vx * yawrate)  # ay = vx*yawrate
        residials[i*31+12] = w["ay"]*(ax + vy * yawrate)  # ax = -vy*yawrate
        residials[i*31+13] = w["slw_v"]*(slw_v - (lenkwinkel_p - schwimmwinkel - l_v * yawrate / vx))  # slw_v = lenkwinkel - schwimmwinkel - l_v*yawrate/vx
        residials[i*31+14] = w["slw_v"]*(slw_v - (masse_p / sls) * (ay * l_h / radstand_p))  # slw_v = masse/sls*l_h/radstand*ay
        residials[i*31+15] = w["slw_h"]*(slw_h - (l_h * yawrate / vx - schwimmwinkel))  # slw_h = l_h*yawrate/v - schwimmwinkel
        residials[i*31+16] = w["slw_h"]*(slw_h - (masse_p * l_v * ay / (sls * radstand_p)))  # slw_h = masse/sls * l_v/radstand*ay
        residials[i*31+17] = w["masse"]*(masse_p - sls * (slw_v + slw_h)/(vx * yawrate))  # masse*v*yawrate = sls*(slw_v+slw_h)
        residials[i*31+18] = w["masse"]*(masse_p - sls * slw_h*radstand_p/(ay * l_h))  # masse*(l_h/Radstand)*ay = sls*slw_h
        residials[i*31+19] = w["masse"]*(masse_p - sls * slw_v * radstand_p(ay * l_v))  # masse*(l_v/Radstand)*ay = sls*slw_v
        residials[i*31+20] = w["yawrate"]*(yawrate-lenkwinkel_p*vx/(radstand_p+EG*vx**2))  # yawrate = lenkwinkel* v/(Radstand+EG*v**2)
        residials[i*31+21] = w["0"]*(sls*(slw_v*l_v-slw_h*l_h))  # 0 = sls*(slw_v*l_v - slw_h*l_h)
        residials[i*31+22] = w["lenkwinkel"]*(lenkwinkel_p-radstand_p/radius_p+EG*ay)  # lenkwinkel = radstand/Radius + EG*ay
        residials[i*31+23] = w["lenkwinkel"]*(masse_p*vx*yawrate/sls+radstand_p*yawrate/vx + 2*schwimmwinkel - lenkwinkel_p)  # m*v*yawrate+(sls*l_v+sls*l_h)*yawrate/v+(sls+sls)*schwimmwinkel = sls*lenkwinkel
        residials[i*31+24] = w["lenkwinkel"]*((l_v+l_h**2/l_v)*yawrate/vx + radstand_p*schwimmwinkel/l_v - lenkwinkel_p)  # (sls*l_v**2+sls*l_h**2)*yawrate/v + (sls*l_v+sls*l_h)*schwimmwinkel = sls*l_v*lenkwinkel
        residials[i*31+25] = w["ay"]*((slw_v-slw_h)/EG - ay)  #slw_v-slw_h = EG * ay
        residials[i*31+26] = w["masse"]*(masse_p - EG*(radstand_p*sls)/(l_h-l_v))  # masse*(l_h-l_v)/(Radstand*srs) = EG
        residials[i*31+27] = w["lenkwinkel"]*(slw_v-slw_h - (lenkwinkel_p+(l_h-l_v)*yawrate/vx))  # slw_v-slw_h = lenkwinkel + (l_h - l_v)*yawrate/v
        residials[i*31+28] = w["lenkwinkel"]*(l_v*yawrate/vx+slw_v+schwimmwinkel-lenkwinkel_p)  # lenkwinkel = l_v*yawrate/v + slw_v+schwimmwinkel
        residials[i*31+29] = w["lenkwinkel"]*(radstand_p/radius_p+slw_v-slw_h-lenkwinkel_p)  # lenkwinkel = Radstand/Radius + slw_v-slw_h
        residials[i*31+30] = w["vy"]*(np.tan(schwimmwinkel)*vx-vy)  # tan(schwimmwinkel) = vy/vx
    #TODO scale all residials to be about the same size
    residials = [abs(r) for r in residials]
    if debugout:
        print('\n'.join([str(i)+", "+str(residials[i]) for i in range(len(residials))]))
    return residials


def fit_stationaereKreisfahrt_daten():

    sls = 100  # TODO dont know what typical values look like
    EG = 0
    # x = [radstand_p, l_v, l_h, sls, masse_p, EG] + k*[radius_p, T_p, lenkwinkel_p, vx, vy, ax, ay, yawrate, schwimmwinkel, slw_v, slw_h]
    xnames = ["radstand", "l_v", "l_h", "sls", "masse", "EG"]
    init = [radstand, 0.5*radstand, 0.5*radstand, sls, masse, EG]
    lowb = [radstand-1, 0, 0, -np.inf, 0, -np.inf]
    higb = [radstand+1, radstand, radstand, +np.inf, 1000, +np.inf]
    for i in range(len(radius)):
        yawrate = 2*np.pi/T[i]
        vx = radius[i]*yawrate
        slw_v = lenkwinkel[i] - 0.5*radstand*yawrate/vx
        slw_h = 0.5*radstand*yawrate/vx
        xnames += [name+"_"+str(i) for name in ["radius", "T", "lenkwinkel", "vx", "vy", "ax", "ay", "yawrate", "schwimmwinkel", "slw_v", "slw_h"]]
        init += [radius[i], T[i], lenkwinkel[i], vx, 0, 0, vx*yawrate, yawrate, 0, slw_v, slw_h]
        lowb += [radius[i]-1, 0, -np.pi, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.pi, -np.pi]
        higb += [radius[i]+1, np.inf, +np.pi, 100, +np.inf, +np.inf, +np.inf, +np.inf, np.inf, +np.pi, +np.pi]
    print("lest_squares: ")
    res = optimize.least_squares(fun=sk_Loss, x0=np.array(init), bounds=(np.array(lowb), np.array(higb)), max_nfev=1000*len(init), xtol=1e-20, ftol=1e-20)
    tot_los = sk_Loss(res["x"], debugout=True)
    print("total final loss of least squares= ", sum(tot_los))
    print("\n".join([xnames[i]+": "+str(res["x"][i]) for i in range(len(xnames))]))
    print("\n\n res = ")
    print(res)


def eukl_dist(p0, p1):
    return np.sqrt(abs(p0[0]-p1[0])**2+abs(p0[1]-p1[1])**2)


def calibrate_camera(checkerboard_images: [os.path], manual_focalwidth=True):
    num_corners = 7
    # https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,6,0)
    objp = np.zeros((num_corners*num_corners, 3), np.float32)
    objp[:,:2] = np.mgrid[0:num_corners,0:num_corners].T.reshape(-1,2)
    objp *= 0.05  # one chessboard grid is 5cm wide/height  # doesnt change anything
    print("object points = ", objp)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    #images = glob.glob('*.jpg')
    images = checkerboard_images
    hand_annotatet_corners = {
        "Image__2022-12-05__13-41-43.bmp": [(0.5380859375, 0.05859375), (0.638671875, 0.0537109375), (0.732421875, 0.05078125), (0.8203125, 0.0556640625), (0.8984375, 0.060546875), (0.5439453125, 0.224609375), (0.646484375, 0.21484375), (0.7421875, 0.2109375), (0.826171875, 0.205078125), (0.9052734375, 0.2041015625), (0.5517578125, 0.3984375), (0.6533203125, 0.384765625), (0.75, 0.3701171875), (0.833984375, 0.361328125), (0.9091796875, 0.3505859375), (0.5595703125, 0.564453125), (0.6591796875, 0.544921875), (0.7490234375, 0.5283203125), (0.833984375, 0.51171875), (0.908203125, 0.49609375), (0.568359375, 0.7333984375), (0.6640625, 0.7080078125), (0.7548828125, 0.6826171875), (0.837890625, 0.6591796875), (0.91015625, 0.6396484375)],
        "Image__2022-12-05__13-41-49.bmp": [(0.54296875, 0.1083984375), (0.6376953125, 0.1123046875), (0.7294921875, 0.1181640625), (0.810546875, 0.126953125), (0.8857421875, 0.13671875), (0.54296875, 0.2744140625), (0.640625, 0.2734375), (0.73046875, 0.2724609375), (0.810546875, 0.2744140625), (0.884765625, 0.275390625), (0.5458984375, 0.4404296875), (0.6435546875, 0.4306640625), (0.732421875, 0.42578125), (0.8115234375, 0.4189453125), (0.8828125, 0.4150390625), (0.546875, 0.6005859375), (0.6416015625, 0.58984375), (0.7294921875, 0.576171875), (0.8056640625, 0.5634765625), (0.876953125, 0.5537109375), (0.548828125, 0.763671875), (0.642578125, 0.7392578125), (0.7265625, 0.724609375), (0.806640625, 0.7041015625), (0.8740234375, 0.685546875)],
        "Image__2022-12-05__13-41-57.bmp": [(0.423828125, 0.15234375), (0.509765625, 0.1171875), (0.5908203125, 0.083984375), (0.666015625, 0.0595703125), (0.7265625, 0.0390625), (0.4462890625, 0.296875), (0.5341796875, 0.259765625), (0.611328125, 0.2197265625), (0.6845703125, 0.1953125), (0.75, 0.1630859375), (0.474609375, 0.4443359375), (0.556640625, 0.400390625), (0.634765625, 0.3583984375), (0.705078125, 0.322265625), (0.767578125, 0.2919921875), (0.4990234375, 0.5888671875), (0.578125, 0.541015625), (0.6533203125, 0.4931640625), (0.7216796875, 0.447265625), (0.783203125, 0.4111328125), (0.52734375, 0.7255859375), (0.6025390625, 0.671875), (0.673828125, 0.619140625), (0.7392578125, 0.5712890625), (0.7958984375, 0.53125)],
        "Image__2022-12-05__13-42-11.bmp": [(0.5087890625, 0.046875), (0.6220703125, 0.05078125), (0.7236328125, 0.052734375), (0.8125, 0.0634765625), (0.892578125, 0.0693359375), (0.513671875, 0.240234375), (0.623046875, 0.2314453125), (0.7236328125, 0.2265625), (0.8125, 0.2265625), (0.888671875, 0.2265625), (0.513671875, 0.4296875), (0.6201171875, 0.412109375), (0.7177734375, 0.3994140625), (0.8056640625, 0.388671875), (0.880859375, 0.375), (0.51953125, 0.609375), (0.6220703125, 0.587890625), (0.7158203125, 0.5654296875), (0.802734375, 0.544921875), (0.8759765625, 0.5322265625), (0.521484375, 0.78125), (0.62109375, 0.75), (0.71484375, 0.7236328125), (0.7978515625, 0.6982421875), (0.873046875, 0.6708984375)],
        "Image__2022-12-05__15-42-53.bmp": [(0.265625, 0.109375), (0.3681640625, 0.09765625), (0.4677734375, 0.0966796875), (0.560546875, 0.0947265625), (0.6494140625, 0.09375), (0.2763671875, 0.2587890625), (0.369140625, 0.2490234375), (0.46484375, 0.240234375), (0.5537109375, 0.240234375), (0.6435546875, 0.2373046875), (0.2783203125, 0.4013671875), (0.37109375, 0.3935546875), (0.46484375, 0.384765625), (0.5546875, 0.376953125), (0.63671875, 0.3701171875), (0.2890625, 0.537109375), (0.373046875, 0.525390625), (0.46484375, 0.5166015625), (0.5498046875, 0.5048828125), (0.6328125, 0.49609375), (0.2919921875, 0.662109375), (0.3798828125, 0.6552734375), (0.4638671875, 0.638671875), (0.548828125, 0.630859375), (0.630859375, 0.615234375)],
        "Image__2022-12-05__15-43-05.bmp": [(0.193359375, 0.1142578125), (0.3251953125, 0.1162109375), (0.4541015625, 0.1201171875), (0.5859375, 0.1337890625), (0.7138671875, 0.14453125), (0.2041015625, 0.3056640625), (0.32421875, 0.306640625), (0.4501953125, 0.318359375), (0.5751953125, 0.32421875), (0.6953125, 0.3349609375), (0.2099609375, 0.48046875), (0.326171875, 0.484375), (0.447265625, 0.48828125), (0.5615234375, 0.49609375), (0.677734375, 0.5009765625), (0.220703125, 0.642578125), (0.33203125, 0.6484375), (0.4443359375, 0.6513671875), (0.5556640625, 0.65234375), (0.6650390625, 0.6552734375), (0.228515625, 0.78515625), (0.333984375, 0.7841796875), (0.44140625, 0.7880859375), (0.544921875, 0.7900390625), (0.646484375, 0.796875)],
        "Image__2022-12-05__15-43-08.bmp": [(0.1240234375, 0.064453125), (0.263671875, 0.0615234375), (0.41015625, 0.060546875), (0.5556640625, 0.0703125), (0.697265625, 0.0830078125), (0.1337890625, 0.283203125), (0.2685546875, 0.2783203125), (0.40625, 0.2802734375), (0.548828125, 0.2861328125), (0.68359375, 0.2900390625), (0.1494140625, 0.4716796875), (0.27734375, 0.478515625), (0.4052734375, 0.4755859375), (0.541015625, 0.4775390625), (0.6640625, 0.4814453125), (0.16796875, 0.6455078125), (0.28125, 0.6513671875), (0.408203125, 0.65234375), (0.53125, 0.650390625), (0.65234375, 0.6494140625), (0.1787109375, 0.7978515625), (0.2939453125, 0.802734375), (0.40625, 0.8056640625), (0.5244140625, 0.802734375), (0.6357421875, 0.802734375)],
        "Image__2022-12-05__15-43-19.bmp": [(0.193359375, 0.138671875), (0.2939453125, 0.138671875), (0.3935546875, 0.142578125), (0.484375, 0.1494140625), (0.5869140625, 0.1494140625), (0.189453125, 0.2919921875), (0.2978515625, 0.287109375), (0.3857421875, 0.2890625), (0.4833984375, 0.2900390625), (0.5771484375, 0.296875), (0.2021484375, 0.427734375), (0.2939453125, 0.427734375), (0.3857421875, 0.4326171875), (0.4794921875, 0.4296875), (0.5703125, 0.43359375), (0.205078125, 0.556640625), (0.298828125, 0.556640625), (0.3818359375, 0.556640625), (0.4765625, 0.5576171875), (0.5576171875, 0.564453125), (0.2109375, 0.6845703125), (0.296875, 0.6845703125), (0.3798828125, 0.6826171875), (0.4658203125, 0.685546875), (0.552734375, 0.6826171875)],
        "Image__2022-12-05__15-43-21.bmp": [(0.15234375, 0.08203125), (0.25, 0.080078125), (0.3466796875, 0.0830078125), (0.4423828125, 0.0869140625), (0.541015625, 0.0947265625), (0.15625, 0.2265625), (0.251953125, 0.22265625), (0.3388671875, 0.2236328125), (0.4345703125, 0.232421875), (0.521484375, 0.240234375), (0.1572265625, 0.357421875), (0.2451171875, 0.3603515625), (0.34375, 0.369140625), (0.4267578125, 0.3671875), (0.5185546875, 0.3740234375), (0.15625, 0.4990234375), (0.2470703125, 0.486328125), (0.3447265625, 0.486328125), (0.4248046875, 0.490234375), (0.513671875, 0.4931640625), (0.1689453125, 0.6064453125), (0.2490234375, 0.6103515625), (0.33203125, 0.607421875), (0.416015625, 0.611328125), (0.501953125, 0.607421875)],
        "Image__2022-12-05__15-43-26.bmp": [(0.26953125, 0.09375), (0.37109375, 0.0986328125), (0.474609375, 0.107421875), (0.5751953125, 0.1162109375), (0.671875, 0.1240234375), (0.26953125, 0.2451171875), (0.37109375, 0.25390625), (0.4697265625, 0.25390625), (0.5693359375, 0.263671875), (0.6650390625, 0.271484375), (0.2724609375, 0.3955078125), (0.3671875, 0.4013671875), (0.4658203125, 0.4052734375), (0.560546875, 0.4091796875), (0.6484375, 0.412109375), (0.275390625, 0.5322265625), (0.37109375, 0.5341796875), (0.4599609375, 0.53515625), (0.552734375, 0.5380859375), (0.638671875, 0.5419921875), (0.2822265625, 0.66015625), (0.37109375, 0.6650390625), (0.4580078125, 0.6640625), (0.5458984375, 0.6640625), (0.6298828125, 0.6640625)],
        "Image__2022-12-05__15-43-29.bmp": [(0.2490234375, 0.029296875), (0.357421875, 0.0302734375), (0.470703125, 0.0390625), (0.5791015625, 0.0458984375), (0.685546875, 0.0556640625), (0.248046875, 0.1943359375), (0.3564453125, 0.197265625), (0.4619140625, 0.2001953125), (0.5693359375, 0.2080078125), (0.673828125, 0.2177734375), (0.2509765625, 0.345703125), (0.3515625, 0.353515625), (0.458984375, 0.359375), (0.55859375, 0.359375), (0.658203125, 0.373046875), (0.2578125, 0.4892578125), (0.35546875, 0.4990234375), (0.455078125, 0.498046875), (0.5498046875, 0.4990234375), (0.642578125, 0.50390625), (0.259765625, 0.626953125), (0.353515625, 0.62890625), (0.44921875, 0.6328125), (0.541015625, 0.638671875), (0.6337890625, 0.6357421875)],
        "Image__2022-12-05__15-43-39.bmp": [(0.2314453125, 0.2021484375), (0.341796875, 0.2099609375), (0.45703125, 0.21484375), (0.5615234375, 0.2255859375), (0.669921875, 0.236328125), (0.236328125, 0.365234375), (0.3427734375, 0.365234375), (0.443359375, 0.376953125), (0.5546875, 0.380859375), (0.640625, 0.3828125), (0.2392578125, 0.5078125), (0.3369140625, 0.5107421875), (0.4423828125, 0.51953125), (0.5390625, 0.5224609375), (0.6376953125, 0.5263671875), (0.2421875, 0.650390625), (0.34375, 0.6494140625), (0.4384765625, 0.6494140625), (0.53515625, 0.6533203125), (0.6240234375, 0.654296875), (0.2529296875, 0.7666015625), (0.3427734375, 0.763671875), (0.43359375, 0.767578125), (0.51953125, 0.7666015625), (0.61328125, 0.7646484375)],
        "Image__2022-12-05__15-43-45.bmp": [(0.2890625, 0.083984375), (0.423828125, 0.091796875), (0.5517578125, 0.095703125), (0.6787109375, 0.107421875), (0.7978515625, 0.1220703125), (0.2919921875, 0.2841796875), (0.4228515625, 0.279296875), (0.5380859375, 0.283203125), (0.669921875, 0.2890625), (0.7822265625, 0.298828125), (0.3017578125, 0.455078125), (0.4169921875, 0.455078125), (0.5361328125, 0.455078125), (0.6494140625, 0.458984375), (0.7646484375, 0.466796875), (0.3046875, 0.619140625), (0.4208984375, 0.615234375), (0.52734375, 0.62109375), (0.6357421875, 0.62109375), (0.7392578125, 0.6171875), (0.3095703125, 0.7578125), (0.4169921875, 0.755859375), (0.521484375, 0.7568359375), (0.6220703125, 0.7529296875), (0.7265625, 0.7568359375)],
        "Image__2022-12-05__15-43-48.bmp": [(0.203125, 0.1064453125), (0.3349609375, 0.109375), (0.46875, 0.11328125), (0.6064453125, 0.1220703125), (0.7353515625, 0.1298828125), (0.2080078125, 0.3076171875), (0.3330078125, 0.310546875), (0.4677734375, 0.3134765625), (0.58984375, 0.3173828125), (0.716796875, 0.32421875), (0.216796875, 0.4853515625), (0.3427734375, 0.4853515625), (0.455078125, 0.4892578125), (0.58203125, 0.486328125), (0.6962890625, 0.4931640625), (0.23046875, 0.6494140625), (0.3447265625, 0.6474609375), (0.4580078125, 0.65234375), (0.5703125, 0.6533203125), (0.681640625, 0.65234375), (0.240234375, 0.791015625), (0.3466796875, 0.7939453125), (0.4541015625, 0.796875), (0.5595703125, 0.7978515625), (0.6669921875, 0.794921875)],
        "Image__2022-12-05__15-43-50.bmp": [(0.376953125, 0.26171875), (0.5107421875, 0.265625), (0.6484375, 0.2783203125), (0.775390625, 0.29296875), (0.892578125, 0.3125), (0.3740234375, 0.4541015625), (0.505859375, 0.46484375), (0.630859375, 0.4697265625), (0.7548828125, 0.4794921875), (0.865234375, 0.48828125), (0.3759765625, 0.6240234375), (0.494140625, 0.6318359375), (0.6171875, 0.6376953125), (0.732421875, 0.6396484375), (0.8427734375, 0.64453125), (0.375, 0.7890625), (0.4912109375, 0.7939453125), (0.6005859375, 0.7998046875), (0.7138671875, 0.7958984375), (0.8193359375, 0.796875), (0.3818359375, 0.921875), (0.4814453125, 0.9267578125), (0.591796875, 0.9326171875), (0.6953125, 0.9306640625), (0.80078125, 0.9326171875)],
        "Image__2022-12-05__15-43-52.bmp": [(0.111328125, 0.091796875), (0.2607421875, 0.0947265625), (0.400390625, 0.0947265625), (0.5439453125, 0.107421875), (0.677734375, 0.119140625), (0.12890625, 0.2939453125), (0.26171875, 0.2900390625), (0.3916015625, 0.2978515625), (0.533203125, 0.306640625), (0.6533203125, 0.3203125), (0.146484375, 0.474609375), (0.26171875, 0.4775390625), (0.3994140625, 0.4794921875), (0.5224609375, 0.4912109375), (0.650390625, 0.4970703125), (0.1669921875, 0.611328125), (0.283203125, 0.625), (0.3955078125, 0.6337890625), (0.509765625, 0.630859375), (0.626953125, 0.6337890625), (0.17578125, 0.7529296875), (0.287109375, 0.7607421875), (0.3994140625, 0.765625), (0.5048828125, 0.765625), (0.609375, 0.7734375)],
        "Image__2022-12-05__15-44-00.bmp": [(0.0712890625, 0.0205078125), (0.2236328125, 0.0224609375), (0.384765625, 0.021484375), (0.53125, 0.03515625), (0.67578125, 0.048828125), (0.0791015625, 0.2646484375), (0.2265625, 0.255859375), (0.375, 0.25390625), (0.5263671875, 0.25390625), (0.65234375, 0.255859375), (0.09375, 0.49609375), (0.23046875, 0.486328125), (0.3798828125, 0.47265625), (0.515625, 0.46875), (0.6455078125, 0.470703125), (0.1083984375, 0.6943359375), (0.2421875, 0.68359375), (0.3779296875, 0.6806640625), (0.5126953125, 0.666015625), (0.6357421875, 0.654296875), (0.1279296875, 0.876953125), (0.2529296875, 0.8701171875), (0.3828125, 0.8583984375), (0.5009765625, 0.8388671875), (0.6171875, 0.8251953125)],
        "Image__2022-12-05__15-44-06.bmp": [(0.146484375, 0.1708984375), (0.26171875, 0.166015625), (0.3779296875, 0.16796875), (0.4755859375, 0.177734375), (0.564453125, 0.185546875), (0.142578125, 0.36328125), (0.26171875, 0.361328125), (0.3740234375, 0.3466796875), (0.46875, 0.3408203125), (0.5517578125, 0.3408203125), (0.1396484375, 0.5537109375), (0.2666015625, 0.53515625), (0.369140625, 0.529296875), (0.4638671875, 0.501953125), (0.5439453125, 0.4814453125), (0.16015625, 0.736328125), (0.2724609375, 0.703125), (0.37109375, 0.685546875), (0.4658203125, 0.66015625), (0.5498046875, 0.634765625), (0.162109375, 0.8974609375), (0.2783203125, 0.8603515625), (0.375, 0.83203125), (0.45703125, 0.796875), (0.54296875, 0.7685546875)],
        "Image__2022-12-05__15-44-10.bmp": [(0.23046875, 0.177734375), (0.3515625, 0.1796875), (0.462890625, 0.1884765625), (0.5634765625, 0.1953125), (0.640625, 0.19921875), (0.2265625, 0.376953125), (0.34765625, 0.3681640625), (0.462890625, 0.3662109375), (0.548828125, 0.365234375), (0.63671875, 0.3662109375), (0.232421875, 0.5693359375), (0.3486328125, 0.552734375), (0.453125, 0.5341796875), (0.54296875, 0.5234375), (0.626953125, 0.51171875), (0.234375, 0.751953125), (0.3427734375, 0.7236328125), (0.443359375, 0.701171875), (0.5322265625, 0.68359375), (0.619140625, 0.65625), (0.2412109375, 0.91015625), (0.34765625, 0.87890625), (0.443359375, 0.8486328125), (0.53125, 0.8193359375), (0.609375, 0.796875)],
        "Image__2022-12-05__15-44-13.bmp": [(0.328125, 0.1708984375), (0.4365234375, 0.1806640625), (0.546875, 0.1787109375), (0.6376953125, 0.1796875), (0.720703125, 0.1826171875), (0.3232421875, 0.34765625), (0.4375, 0.3408203125), (0.529296875, 0.333984375), (0.619140625, 0.328125), (0.705078125, 0.3232421875), (0.328125, 0.5068359375), (0.4306640625, 0.4921875), (0.525390625, 0.486328125), (0.61328125, 0.47265625), (0.6943359375, 0.4638671875), (0.32421875, 0.65234375), (0.4228515625, 0.6396484375), (0.5146484375, 0.62109375), (0.6015625, 0.6015625), (0.6767578125, 0.5908203125), (0.3310546875, 0.77734375), (0.423828125, 0.7646484375), (0.5068359375, 0.744140625), (0.591796875, 0.72265625), (0.6650390625, 0.7099609375)]}
    print("images = ", images)
    for fname in images:
        if manual_focalwidth:
            dist_to_chessboard = float(str(fname).split("\\")[-1].split("_")[1].replace("m", "."))
            print("dist to chessboard = ", dist_to_chessboard)
        img = cv2.imread(str(fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #for i in range(5):
        #    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        #    kernel = np.ones((3+2*i,3+2*i), np.float32)/25
        #img = cv2.filter2D(img, -1, kernel)
        #_, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        #print("? = ", _)
        print("img = ", getType(img))
        #cv2.imshow("blured", cv2.resize(img, (1200, 900)))

        print("fname = ", fname)
        print("size = ", img.shape)
        # Find the chess board corners
        gray = img
        ret, corners = cv2.findChessboardCorners(gray, (num_corners,num_corners), None)
        if not ret and str(fname).split("\\")[-1] in hand_annotatet_corners.keys():
            print("use hand_annotetet corners.")
            h, w = gray.shape
            corners = hand_annotatet_corners[str(fname).split("\\")[-1]]
            corners = np.array([ [[float(c[0]*w), float(c[1]*h)]] for c in corners], dtype="float32")
            ret = True
        #print(f"findChessboardCorners: ret = {ret}, corners = {corners}")
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            #print("corners2 = ", getType(corners2), ":\n", corners2)
            #corners2 = corners
            imgpoints.append(corners2)
            #retval, rvec, tvec, inliers = cv2.solvePnPRansac(np.array(objp), np.array(corners2), np.array([[6.00842969e+03, 0, 2.57574988e+03], [0, 5.16761328e+03, 1.31716630e+03], [0, 0., 1.00000000e+00]]), np.array([[3.01768366e+01, -5.49976081e+03, 2.25441512e-01, -8.76616022e-02, -1.50164371e+01]]))
            #print(f"retval = {retval}, rvec = {rvec}\ntvec = {tvec}\ninliers = {inliers}")
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (num_corners,num_corners), corners2, ret)
            #cv2.imshow('img', cv2.resize(img, (1200, 900)))
            #cv2.waitKey(0)
            if manual_focalwidth:
                # manually calculate focal length:
                #print(f"corners2 = {corners2}")
                average_pixdist_y = sum([sum([eukl_dist(corners2[num_corners*j+i][0], corners2[7*j+i+1][0]) for i in range(num_corners-1)]) for j in range(num_corners-1)])/((num_corners-1)**2)
                average_pixdist_x = sum([sum([eukl_dist(corners2[7*j+i][0], corners2[7*(j+1)+i][0]) for i in range(num_corners-1)]) for j in range(num_corners-1)])/((num_corners-1)**2)
                average_pixdist = 0.5*average_pixdist_y+0.5*average_pixdist_x
                print(f"average_pixel_distance = {average_pixdist} ({average_pixdist_x}, {average_pixdist_y}")
                print(f"focal_length [m] = {average_pixdist*dist_to_chessboard/0.05} ({average_pixdist_x*dist_to_chessboard/0.05}, {average_pixdist_y*dist_to_chessboard/0.05})")
                #1m: 3896.1540983042537 (3807.243392394215, 3985.064804214292)
                #1m: 3942.9970260337222 (4048.472753030411, 3837.5212990370333)
                #0.75m: 3721.392345094099 (3681.038369803903, 3761.746320384294)
                print("estimated dist_to_chessboard [m] = (object_width * focalLength) / pixel_width =", (0.05 * 3919) / average_pixdist)
                #end manually calculate focal length
        else:
            print("no chessboard corners detected in ", fname)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(f"ret = {ret}, mtx = {mtx}, dist [k1, k2, p1, p2, k3], so that real_pixel_position = f(mesured_pixel_position, dist)= {dist}, rvecs = {rvecs}, tvecs = {tvecs}")
    h, w = img.shape
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    print(f"cameraMatrix = {newcameramtx}, roi = {roi}")
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    cv2.imshow("undistored_img", cv2.resize(dst, (1200, 900)))
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def read_basler_camera():
    tlFactory = pylon.TlFactory.GetInstance()
    devices = tlFactory.EnumerateDevices()
    print("devices = ", devices)

    tmp = tlFactory.CreateFirstDevice()
    print("first Device = ", tmp)
    camera = pylon.InstantCamera(tmp)
    camera.Open()

    # demonstrate some feature access
    new_width = camera.Width.GetValue() - camera.Width.GetInc()
    if new_width >= camera.Width.GetMin():
        camera.Width.SetValue(new_width)

    camera.BslBrightness.SetValue(1.0)

    index = 0
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    objp = np.zeros((7*7, 3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)  # 3D points in chessboard-reference frame
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    while index < 14:
        camera.StartGrabbingMax(1)
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        # frame_size = 1920x1200

        if grabResult.GrabSucceeded():
            # Access the image data.
            #print("SizeX: ", grabResult.Width)
            #print("SizeY: ", grabResult.Height)
            img = grabResult.Array
            #print("Gray value of first pixel: ", img[0, 0])

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(img, (7,7), None)
            # If found, add object points, image points (after refining them)
            print("could see corners: ", ret)
            if ret:
                cv2.imwrite(f"frame_{index}.jpg", np.array(img))
                index += 1
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(img, corners, (11,11), (-1,-1), criteria)
                cv2.drawChessboardCorners(img, (7,7), corners2, ret)
                imgpoints.append(corners2)
            cv2.imshow("image", cv2.resize(img, (840, 840)))
            cv2.waitKey(1)
        grabResult.Release()
    camera.Close()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
    print(f"ret = {ret}, mtx = {mtx}, dist [k1, k2, p1, p2, k3], so that real_pixel_position = f(mesured_pixel_position, dist)= {dist}, rvecs = {rvecs}, tvecs = {tvecs}")
    h, w, _ = img.shape
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))


def test_solvePnPRansac():
    #https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#solvepnpransac
    nkeypoitns = 7
    #objectPoints – Array of object points in the object coordinate space, 3xN/Nx3 1-channel or 1xN/Nx1 3-channel, where N is the number of points. vector<Point3f> can be also passed here.
    # constant. object coordinate space (0, 0, 0) lies in center of baseplate of cone.
    cone_height = 0.325  # [m]
    cone_diamter_top = 0.046  # (3/21)*np.sqrt(2*0.228**2)  # [m]
    cone_diameter_bottom = 0.169  # (11/21)*np.sqrt(2*0.228**2)  # [m]
    cone_diamter_dif = cone_diameter_bottom-cone_diamter_top
    objectPoints = np.array([[0, cone_height, 0], [cone_diamter_top+(1/3)*cone_diamter_dif, (2/3)*cone_height, 0], [cone_diamter_top+(2/3)*cone_diamter_dif, (1/3)*cone_height, 0], [cone_diamter_top+(3/3)*cone_diamter_dif, (0/3)*cone_height, 0], [-cone_diamter_top-(1/3)*cone_diamter_dif, (2/3)*cone_height, 0], [-cone_diamter_top-(2/3)*cone_diamter_dif, (1/3)*cone_height, 0], [-cone_diamter_top-(3/3)*cone_diamter_dif, (0/3)*cone_height, 0]])  # [[left, height, deepth]]
    objectPoints = np.array([[c, a, b] for [a, b, c] in objectPoints])  # chessboard-objectpoints: [0.2, 0.2, 0]
    #imagePoints – Array of corresponding image points, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel, where N is the number of points. vector<Point2f> can be also passed here.
    # in this case: keypoints, translated into full image
    imagePoints = np.array([(t[0]*1200, t[0]*1920) for t in [(0.244140625, 0.46484375), (0.228515625, 0.51953125), (0.2255859375, 0.568359375), (0.21875, 0.6142578125), (0.2587890625, 0.5185546875), (0.2646484375, 0.5625), (0.271484375, 0.609375)]])  # detected by keypoint_regression on image patch showing cone.

    cameraMatrix = np.array([[1.55902258e+03, 0, 1.03564443e+03], [0, 1.49628271e+03, 6.89322516e+02], [0, 0, 1]])  # Basler Camera
    #cameraMatrix = [[6.22047119e+03, 0, 1.70052709e+03], [0, 7.10611963e+03, 2.29002018e+03], [0, 0, 1]]  # printed by calibrate_camera
    #distCoeffs – Input vector of distortion coefficients (k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]]) of 4, 5, or 8 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.
    distCoeffs = np.array([[-2.42797289e-01, 9.77514487e-02, -8.00761502e-05, 5.61321688e-03, 1.08419697e-02]])  # Basler Camera
    #distCoeffs = [-2.12039244e+00, 1.02842136e+02, 5.43016543e-02, 2.14161540e-02, -1.26235781e+03]
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs)
    #rvec – Output rotation vector (see Rodrigues() ) that, together with tvec , brings points from the model coordinate system to the camera coordinate system.
    #tvec – Output translation vector. tvec*0.05m = [?, ?, distanz in sichtrichtung] (0.05m = size of calibration chessbord)
    #inliers – Output vector that contains indices of inliers in objectPoints and imagePoints .
    print(f"retval = {retval}\ntvec = {tvec}\ninliers = {inliers}")


def show_synscreen():
    #user32 = ctypes.windll.user32
    #screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    screensize = (1920, 1080)
    print("screensize = ", screensize)
    img0 = np.zeros((screensize[1], screensize[0]))
    img1 = np.ones((screensize[1], screensize[0]))*255

    imgs = [img0, img1]

    while True:
        now = datetime.datetime.now()
        print("now = ", str())
        img = np.copy(imgs[now.second%2])
        color = imgs[(now.second+1)%2][0][0]
        cv2.putText(img, str(now), (10,500), cv2.FONT_HERSHEY_SIMPLEX, 4, color, 2, cv2.LINE_AA)
        print("next ", img[0][0])
        cv2.imshow("clock", img)
        cv2.waitKey(1)


def vidoe2frames(video_path, frames_dir):
    pathlib.Path(frames_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    c = 0
    while ret:
        cv2.imwrite(f"{frames_dir}/frame_{str(c)}.jpg", frame)
        c += 1
        ret, frame = cap.read()


if __name__ == "__main__":
    frames_dir = "C:/Users/Idefix/PycharmProjects/datasets/testrun_2022_12_17/cam_footage/right_cam_14_46_00/"
    output_dir = "C:/Users/Idefix/PycharmProjects/OpenLabeling/main/input/"
    som_frame = 1225
    eom_frame = 2586
    for img_file in os.listdir(frames_dir):
        #img_file = "camR3_frame_{frnr}.jpg"
        frnr = int(img_file.replace(".jpg", "").replace("camR3_frame_", ""))
        if som_frame <= frnr <=eom_frame:
            img = cv2.imread(frames_dir+img_file)
            img = np.rot90(img, 2)
            cv2.imwrite(output_dir+img_file, img)
    exit(0)
    lines = []
    with open("C:/Users/Idefix/PycharmProjects/datasets/keypoints/cone_annotations.csv", 'r') as f:
        lines = f.readlines()
    lines = [line.replace("\\", "/").replace("/droneview/camL3_camL3_", "/cones/camL3_") for line in lines]
    with open("C:/Users/Idefix/PycharmProjects/datasets/keypoints/cone_annotations2.csv", 'w') as f:
        f.writelines(lines)
    #test_myPnP()
    #show_synscreen()
    #test_solvePnPRansac()
    #read_basler_camera()
    #read_camera()
    #keypoint_regression.main()
    exit(0)
    #TODO read
    # https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    # https://github.com/opencv/opencv/tree/4.x/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation
    path = "C:/Users/Idefix/PycharmProjects/datasets/calibration_test/"
    fnames_1m = ["chessboard_1m_5cm_0.jpg", "chessboard_1m_5cm_1.jpg", "chessboard_1m_5cm_2.jpg"]
    #fnames_0m75 = ["chessboard_0m75_5cm_0.jpg"]
    #path = "C:/Users/Idefix/PycharmProjects/datasets/Basler Cam recordings (Accel)/chessboard/"
    #fnames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    calibrate_camera([pathlib.Path(path+file) for file in fnames_1m])
    #show_kflogs.run()

"""
C++ Inference

YOLOv5 OpenCV DNN C++ inference on exported ONNX model examples: - https://github.com/Hexmagic/ONNX-yolov5/blob/master/src/test.cpp - https://github.com/doleron/yolov5-opencv-cpp-python

YOLOv5 OpenVINO C++ inference examples: - https://github.com/dacquaviva/yolov5-openvino-cpp-python - https://github.com/UNeedCryDear/yolov5-seg-opencv-dnn-cpp
"""
