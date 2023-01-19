#https://en.wikipedia.org/wiki/Vincenty%27s_formulae
import numpy as np

heading = float  # https://en.wikipedia.org/wiki/Azimuth  # 0 = North, 90° = pi/2 = East
meter = float
meter_north = float
meter_east = float
degree_north = float
lattitude = degree_north
degree_east = float
longitude = degree_east

# earth constants
a = 6378137.0  # length of semi-major axis of the ellipsoid (radius at equator);
f = 1/298.257223563  # flattening of the ellipsoid;
b = 6356752.314245  # (1-f)*a length of semi-minor axis of the ellipsoid (radius at the poles);

def gps_to_distazimuth(gps: (lattitude, longitude), gps_base: (lattitude, longitude)) -> (meter, heading):
    u1 = np.arctan((1-f)*np.tan(gps_base[0]))
    u2 = np.arctan((1-f)*np.tan(gps[0]))
    L = gps[1]-gps_base[1]
    sin_u1 = np.sin(u1)
    cos_u1 = np.cos(u1)
    sin_u2 = np.sin(u2)
    cos_u2 = np.cos(u2)
    lamba = L
    while True:
        sin_lambda = np.sin(lamba)
        cos_lambda = np.cos(lamba)
        sin_sigma = np.sqrt((cos_u2*sin_lambda)**2+(cos_u1*sin_u2-sin_u1*cos_u2*cos_lambda)**2)
        cos_sigma = sin_u1*sin_u2+cos_u1*cos_u2*cos_lambda
        sigma = np.arctan2(sin_sigma, cos_sigma)
        sin_alpha = cos_u1*cos_u2*sin_lambda/sin_sigma
        cos_alpha_sqrd = 1-sin_alpha**2
        cos_2sm = cos_sigma - 2*sin_u1*sin_u2/cos_alpha_sqrd
        C = f/16*cos_alpha_sqrd*(4+f*(4-3*cos_alpha_sqrd))
        lamba = L+(1-C)*f*sin_alpha*(sigma+C*sin_sigma*(cos_2sm+C*cos_sigma*(-1+2*cos_2sm**2)))
        print("lambda = ", lamba)
        if lamba < 1e-12:  # approx 0.06mm
            break
    u_sqared = cos_alpha_sqrd*(a**2-b**2)/b**2
    A = 1+u_sqared/16384*(4096+u_sqared*(-768+u_sqared*(320-175*u_sqared)))
    B = u_sqared/1024*(256+u_sqared*(-128+u_sqared*(74-47*u_sqared)))
    d_sigma = B*sin_sigma*(cos_2sm+0.25*B*(cos_sigma*(-1+2*cos_2sm**2)-B/6*cos_2sm*(-3+4*sin_sigma**2)*(-3+4*cos_2sm**2)))
    dist = b*A*(sigma-d_sigma)
    azimuth1 = np.arctan2(cos_u2*sin_lambda, cos_u1*sin_u2-sin_u1*cos_u2*cos_lambda)
    azimuth2 = np.arctan2(cos_u1*sin_lambda, -sin_u1*cos_u2+cos_u1*sin_u2*cos_lambda)
    print(f"dist, azimutz from {gps_base} to {gps} = {dist, azimuth1, azimuth2}")
    return dist, azimuth1


def distazimuth_to_gps(gps_base: (lattitude, longitude), dh_vector: (meter, heading)) -> (lattitude, longitude):
    (lat_base, long_base) = gps_base
    (lat_base, long_base) = (lat_base*np.pi/180, long_base*np.pi/180)
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
    azimuth_2 = np.arctan2(sin_alpha, -np.sin(u1)*np.sin(sigma)+np.cos(u1)*np.cos(sigma)*np.cos(azimuth))  # what even is azimuth_2 in this context?
    (lat_res, long_res) = (lat_res*180/np.pi, long_res*180/np.pi)
    print(f"distazimuth_to_gps({gps_base}, {dh_vector}) = {(lat_res, long_res)}, {azimuth_2}")
    return lat_res, long_res


def meter_to_distazimuth(meter_vecotr: (meter_north, meter_east)) -> (meter, heading):
    return np.sqrt(meter_vecotr[0]**2+meter_vecotr[1]**2), np.arctan2(meter_vecotr[1], meter_vecotr[0])


def distazimuth_to_meter(dh_vector: (meter, heading)) -> (meter_north, meter_east):
    (dist, azimuth) = dh_vector
    return np.sin(azimuth)*dist, np.cos(azimuth)


if __name__ == "__main__":
    base = (51.46411414940673, 6.738944977588712)  # gully
    south = (51.46390029594143, 6.738944977588712)  # b is south of a
    west = (51.46411414940673, 6.738611222776312)
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
    all_headingpoints = [nne, north_east, nee, east, see, south_east, sse, south, wss, south_west, wws, west, nww, north_west, nnw, north]
    for point in all_headingpoints:
        print("\n\n")
        dh_vectpr = gps_to_distazimuth(point, base)
        print(distazimuth_to_gps(base, dh_vectpr))
        #assert point == distazimuth_to_gps(base, dh_vectpr)
        #assert meter_to_distazimuth(distazimuth_to_meter(dh_vectpr)) == dh_vectpr