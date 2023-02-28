#https://en.wikipedia.org/wiki/Vincenty%27s_formulae

import numpy as np
import warnings

heading_radiants = float  # https://en.wikipedia.org/wiki/Azimuth
# gps_to_azimuth(a, b) = 0 = 0*pi: a is North of b
# gps_to_azimuth(a, b) = pi/2 = 1.57079: a is East of b
# gps_to_azimuth(a, b) = pi = 3.1415: a is South of b
# gps_to_azimuth(a, b) = -pi/2 = -1.57079: a is  West of b
meter = float
meter_north = float
meter_east = float
meter_pos = (meter_north, meter_east)
degree_north = float  # in radiants
lattitude = degree_north
degree_east = float  # in radiants
longitude = degree_east
gps_pos_radiant = (lattitude, longitude)  # e.g. (0.8982182491908725, 0.11761678938800951)
lattitude_degree = float
longitude_degree = float
gps_pos_degree = (lattitude_degree, longitude_degree)  # e.g. (51.46411476026706, 6.738945631812034)

d2r = np.pi/180
r2d = 180/np.pi
# earth constants
a = 6378137.0  # length of semi-major axis of the ellipsoid (radius at equator);
f = 1/298.257223563  # flattening of the ellipsoid;
b = 6356752.314245  # (1-f)*a length of semi-minor axis of the ellipsoid (radius at the poles);


def degree_to_radiants(gps: gps_pos_degree) -> gps_pos_radiant:
    return d2r*gps[0], d2r*gps[1]


def carposs_to_gnsspos(carposes: [gps_pos_degree]) -> gps_pos_degree:
    return np.array([0.5*0.21/1.54*(carposes[0][0]+carposes[1][0])+0.5*(1-0.21/1.54)*(carposes[2][0]+carposes[3][0]), 0.5*0.23/1.23*(carposes[0][1]+carposes[2][1])+0.5*(1-0.23/1.23)*(carposes[1][1]+carposes[3][1])])
def carposs_to_heading(carposes: [gps_pos_degree]) -> heading_radiants:
    # gps = [Left front wheel gps-position in degree, Right front wheel, Left rear wheel, Right rear wheel]
    # 0: front axis is north of rear axis
    # 0.5*pi: front axis is east of rear axis
    if all([carposes[0][i]+carposes[1][i] == carposes[2][i]+carposes[3][i] for i in [0, 1]]):
        warnings.warn(f"gps_util.carposs_to_heading: avg(carposes[0]={carposes[0]}, carposes[1]={carposes[1]}) and avg(carpoeses[2]={carposes[2]}, carpoese[3]={carposes[3]}) = {0.5*np.array(carposes[0])+0.5*np.array(carposes[1])} are identical, heading not defined")
    return gps_to_azimuth(0.5*np.array(carposes[0])+0.5*np.array(carposes[1]), 0.5*np.array(carposes[2])+0.5*np.array(carposes[3]))
def average(poses: [gps_pos_degree]) -> gps_pos_degree:
    return (np.average([lat for (lat, long) in poses]), np.average([long for (lat, long) in poses]))


def _gps_to_distazimuth(gps: gps_pos_radiant, gps_base: gps_pos_radiant) -> (meter, heading_radiants):
    if any([x < -2*np.pi or x > 2*np.pi for x in gps]):
        warnings.warn(f"gps_util._gps_to_distazimuth: gps {gps} should be in radiants, not degree.")
        gps = degree_to_radiants(gps)
    if any([x < -2*np.pi or x > 2*np.pi for x in gps_base]):
        warnings.warn(f"gps_util._gps_to_distazimuth: gps_base {gps_base} should be in radiants, not degree.")
        gps_base = degree_to_radiants(gps_base)

    u1 = np.arctan((1-f)*np.tan(gps_base[0]))
    u2 = np.arctan((1-f)*np.tan(gps[0]))
    L = gps[1]-gps_base[1]
    sin_u1 = np.sin(u1)
    cos_u1 = np.cos(u1)
    sin_u2 = np.sin(u2)
    cos_u2 = np.cos(u2)
    lamba = L
    i = 0
    while True:
        i += 1
        sin_lambda = np.sin(lamba)
        cos_lambda = np.cos(lamba)
        sin_sigma = np.sqrt((cos_u2*sin_lambda)**2+(cos_u1*sin_u2-sin_u1*cos_u2*cos_lambda)**2)
        cos_sigma = sin_u1*sin_u2+cos_u1*cos_u2*cos_lambda
        sigma = np.arctan2(sin_sigma, cos_sigma)
        sin_alpha = cos_u1*cos_u2*sin_lambda/sin_sigma  # TODO RuntimeWarning: invalid value encountered in double_scalars sin_alpha = cos_u1*cos_u2*sin_lambda/sin_sigma
        cos_alpha_sqrd = 1-sin_alpha**2
        cos_2sm = cos_sigma - 2*sin_u1*sin_u2/cos_alpha_sqrd
        C = f/16*cos_alpha_sqrd*(4+f*(4-3*cos_alpha_sqrd))
        old_lambda = lamba
        lamba = L+(1-C)*f*sin_alpha*(sigma+C*sin_sigma*(cos_2sm+C*cos_sigma*(-1+2*cos_2sm**2)))
        if abs(old_lambda-lamba) < 1e-12:  # approx 0.06mm
            #print(f"gps_util.gps_to_distazimuth: {i} iterations used")  # always 2, for my use cases
            break
        if i > 10:
            warnings.warn(f"gps_util._gps_to_distazimuth: dist and azimuth between gps {gps} and gps_base {gps_base} could not be established after {i} iterations. reaturn best guess")
            break
    u_sqared = cos_alpha_sqrd*(a**2-b**2)/b**2
    A = 1+u_sqared/16384*(4096+u_sqared*(-768+u_sqared*(320-175*u_sqared)))
    B = u_sqared/1024*(256+u_sqared*(-128+u_sqared*(74-47*u_sqared)))
    d_sigma = B*sin_sigma*(cos_2sm+0.25*B*(cos_sigma*(-1+2*cos_2sm**2)-B/6*cos_2sm*(-3+4*sin_sigma**2)*(-3+4*cos_2sm**2)))
    dist = b*A*(sigma-d_sigma)
    azimuth1 = np.arctan2(cos_u2*sin_lambda, cos_u1*sin_u2-sin_u1*cos_u2*cos_lambda)
    #azimuth2 = np.arctan2(cos_u1*sin_lambda, -sin_u1*cos_u2+cos_u1*sin_u2*cos_lambda)  # seems to be almost the same as azimuth1
    #print(f"dist, azimutz from {gps_base} to {gps} = {dist, azimuth1, azimuth2}")
    return dist, azimuth1


def gps_to_distazimuth(gps: gps_pos_radiant, gps_base: gps_pos_radiant) -> (meter, heading_radiants):
    if gps[0] == gps_base[0] and gps[1] == gps_base[1]:
        warnings.warn(f"gps_util.gps_to_distazimuth: gps and gps_base {gps} are identical, heading not defined")
        return 0.0, float("nan")  # maybe better to return 0, float("nan")
    return _gps_to_distazimuth(gps, gps_base)
def gps_to_dist(gps: gps_pos_radiant, gps_base: gps_pos_radiant) -> meter:
    if gps[0] == gps_base[0] and gps[1] == gps_base[1]:
        return 0  # no warning, distance between two identical points is fine
    return _gps_to_distazimuth(gps, gps_base)[0]
def gps_to_azimuth(gps: gps_pos_radiant, gps_base: gps_pos_radiant) -> heading_radiants:
    if gps[0] == gps_base[0] and gps[1] == gps_base[1]:
        warnings.warn(f"gps_util.gps_to_azimuth: gps and gps_base {gps} are identical, heading not defined")
        return 0.0  # maybe better to reurn float("nan")
    return _gps_to_distazimuth(gps, gps_base)[1]


def distazimuth_to_gps(gps_base: gps_pos_radiant, dh_vector: (meter, heading_radiants)) -> gps_pos_radiant:
    if any([x < -2*np.pi or x > 2*np.pi for x in gps_base]):
        warnings.warn(f"gps_util.distazimuth_to_gps: gps_base {gps_base} should be in radiants, not degree.")
        gps_base = degree_to_radiants(gps_base)
    if dh_vector[1] < -2*np.pi or dh_vector[1] > 2*np.pi:
        warnings.warn(f"gps_util.distazimuth_to_gps: dh_vector {dh_vector} [1] should be in radiants, not degree")
        dh_vector = (dh_vector[0], d2r*dh_vector[1])
    (lat_base, long_base) = gps_base
    (dist, azimuth) = dh_vector
    u1 = np.arctan((1-f)*np.tan(lat_base))
    sin_alpha = np.cos(u1)*np.sin(azimuth)
    sigma1 = np.arctan2(np.tan(u1), np.cos(azimuth))
    u_squared = (1-(sin_alpha)**2)*(a**2-b**2)/b**2
    A = 1 + (u_squared/16384)*(4096+u_squared*(-768+u_squared*(320-175*u_squared)))
    B = u_squared/1024*(256+u_squared*(-128+u_squared*(74-47*u_squared)))
    sigma = dist/(b*A)
    while True:
        tmp = np.cos(2*sigma1+sigma)
        dsigma = B*np.sin(sigma)*(tmp+0.25*B*(np.cos(sigma)*(-1+2*tmp**2) - (B/6)*tmp*(-3+4*np.sin(sigma)**2)*(-3+4*np.cos(sigma)**2)))
        sigma = dist/(b*A) + dsigma
        if dsigma < 1e-12:
            break
    lat_res = np.arctan2(np.sin(u1)*np.cos(sigma)+np.cos(u1)*np.sin(sigma)*np.cos(azimuth), (1-f)*np.sqrt(sin_alpha**2+(np.sin(u1)*np.sin(sigma)-np.cos(u1)*np.cos(sigma)*np.cos(azimuth))**2))
    tmp2 = np.cos(np.arcsin(sin_alpha))
    C = f/16*tmp2**2*(4+f*(4-3*tmp2))
    long_res = long_base + np.arctan2(np.sin(sigma)*np.sin(azimuth), np.cos(u1)*np.cos(sigma)-np.sin(u1)*np.sin(sigma)*np.cos(azimuth)) - (1-C)*f*sin_alpha*(sigma+C*np.sin(sigma)*tmp+C*np.cos(sigma)*(-1+2*tmp**2))
    #azimuth_2 = np.arctan2(sin_alpha, -np.sin(u1)*np.sin(sigma)+np.cos(u1)*np.cos(sigma)*np.cos(azimuth))  # what even is azimuth_2 in this context?
    #print(f"distazimuth_to_gps({gps_base}, {dh_vector}) = {(lat_res, long_res)}, {azimuth_2}")
    return lat_res, long_res


def meter_to_dist(meter_vector: (meter_north, meter_east)) -> meter:
    # same as meter_to_distazimuth(meter_vector)[0]
    return np.sqrt(meter_vector[0]**2+meter_vector[1]**2)


def meter_meter_to_dist(meter_pos0: (meter_north, meter_east), meter_pos1: (meter_north, meter_east)) -> meter:
    # same as meter_to_distazimuth(np.array(meter_pos0)-np.array(meter_pos1))[0]
    return np.sqrt((meter_pos0[0]-meter_pos1[0])**2+(meter_pos0[1]-meter_pos1[1])**2)

def meter_to_distazimuth(meter_vecotr: (meter_north, meter_east)) -> (meter, heading_radiants):
    return np.sqrt(meter_vecotr[0]**2+meter_vecotr[1]**2), np.arctan2(meter_vecotr[1], meter_vecotr[0])


def distazimuth_to_meter(dh_vector: (meter, heading_radiants)) -> (meter_north, meter_east):
    (dist, azimuth) = dh_vector
    if azimuth < -2*np.pi or azimuth > 2*np.pi:
        warnings.warn(f"gps_util.distazimuth_to_meter: dh_vector {dh_vector} [1] should be in radiants")
        azimuth = d2r*azimuth
    return np.array([np.cos(azimuth)*dist, np.sin(azimuth)*dist])


def gps_to_meter(gps: (lattitude, longitude), gps_base: (lattitude, longitude)) -> (meter_north, meter_east):
    if gps[0] == gps_base[0] and gps[1] == gps_base[1]:
        return (0, 0)
    return distazimuth_to_meter(gps_to_distazimuth(gps, gps_base))


def meter_to_gps(gps_base: (lattitude, longitude), meter_vector: (meter_north, meter_east)) -> (lattitude, longitude):
    return distazimuth_to_gps(gps_base, meter_to_distazimuth(meter_vector))


if __name__ == "__main__":
    # check that this is working correctly.
    base = (51.46411414940673, 6.738944977588712)  # manhole cover (gully)
    south = (51.46390029594143, 6.738944977588712)  # south is south of base
    west = (51.46411414940673, 6.738611222776312)  # west is west of base
    north = (51.464253785357734, 6.738944977588712)
    east = (51.46411414940673, 6.739156827453659)
    north_east = (51.46422840497507, 6.739062279598764)
    south_east = (51.464004904891205, 6.739078372856272)
    south_west = (51.46400281610031, 6.738769918837057)
    north_west = (51.46421085921322, 6.738750472820845)
    sse = (51.46402913898559, 6.739008606790168)
    see = (51.46407175027804, 6.739119247902627)
    nee = (51.464134413871115, 6.73915143440807)
    nne = (51.46424887914565, 6.739003242372594)
    nnw = (51.46424094178242, 6.7388288987956315)
    nww = (51.464158226014476, 6.738712222713402)
    wws = (51.46408344748917, 6.738700152773861)
    wss = (51.463951018180005, 6.738827557691239)

    all_headingpoints = [("nne", nne), ("ne", north_east), ("nee", nee), ("e", east), ("see", see), ("se", south_east), ("sse", sse), ("s", south), ("wss", wss), ("sw", south_west), ("wws",wws), ("w", west), ("nww", nww), ("nw", north_west), ("nnw", nnw), ("n", north)]
    all_headingpoints = [(name, degree_to_radiants(gps)) for (name, gps) in all_headingpoints]
    base = degree_to_radiants(base)
    mv0 = distazimuth_to_meter(gps_to_distazimuth(base, all_headingpoints[-1][1]))
    mv1 = distazimuth_to_meter(gps_to_distazimuth(all_headingpoints[-1][1], all_headingpoints[3][1]))
    mv2 = distazimuth_to_meter(gps_to_distazimuth(base, all_headingpoints[3][1]))
    print(f"adding two meter vector-sides of a triangle is the same as remaining side:\n{mv0[0]+mv1[0], mv0[1]+mv1[1]} = \n{mv2}")
    for (h, point) in all_headingpoints:
        print(f"\n\n{h}:")
        dh_vectpr = gps_to_distazimuth(point, base)
        print("dh_vector = ", dh_vectpr)
        point_again = distazimuth_to_gps(base, dh_vectpr)
        print(f"point = {point}")
        print(f"point = {point_again}")
        print(f"point = {distazimuth_to_gps(base, meter_to_distazimuth(distazimuth_to_meter(dh_vectpr)))}")
        print(f"mv = {distazimuth_to_meter(dh_vectpr)}")
        #assert point == distazimuth_to_gps(base, dh_vectpr)
        #assert meter_to_distazimuth(distazimuth_to_meter(dh_vectpr)) == dh_vectpr

    # matplotlib: (lat, long) = gps_pos
    # plot(x=long, y=lat) -> north is up, east is right. As is should be
    # axe.arrow(x=base[1], y=base[0], dx=point[1]-base[1], dy=point-base[0], width=np.sqrt(dx**2+dy**2)*0.1)  # arrow from base to point, with arrow head on point.
    import matplotlib.pyplot as plt
    fig, (axe) = plt.subplots(1)
    axe.set_title(f"arrow from base to norh")
    cmap = plt.get_cmap('jet')
    axe.scatter(x=np.array([base[1]]), y=np.array([base[0]]), color="black")
    i = 0
    for (name, point) in all_headingpoints:
        color = cmap(i/len(all_headingpoints))
        axe.scatter(x=np.array([point[1]]), y=np.array([point[0]]), color=color, label=name)
        axe.text(point[1], point[0], name, c="black")
        i += 1
    (dx, dy) = d2r*north[1]-base[1], d2r*north[0]-base[0]
    print("dx, dy =", dx, dy)
    axe.arrow(x=base[1], y=base[0], dx=dx, dy=dy, width=np.sqrt(dx**2+dy**2)*0.1)
    axe.set_xlabel("long")
    axe.set_ylabel("lat")
    axe.grid()
    fig.show()

    fig, (ax1) = plt.subplots(1)
    ax1.set_title("meter")
    ax1.scatter(x=np.array([0]), y=np.array([0]), color="black")
    i = 0
    for (name, point) in all_headingpoints:
        color = cmap(i/len(all_headingpoints))
        mv = gps_to_meter(point, base)
        print("mv =", mv)
        ax1.scatter(x=np.array([mv[1]]), y=np.array([mv[0]]), color=color)
        ax1.text(mv[1], mv[0], name, c="black")
        i += 1
    ax1.set_xlabel("east")
    ax1.set_ylabel("north")
    ax1.grid()
    fig.show()
