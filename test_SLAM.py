import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from util import getType, plot_and_save, smothing, to_range, get_at_time
from show_sensorlogs import read_csv, x_ending, true_pos_from_droneimg_pxpos, visual_pipeline, plot_cones, camL2t, get_synced_frame, poii_bluecones_range, poii_yellowcones_range,get_boundingboxes_keypoints_poii, custom_PnP, t2ssdt, drone2t, ssdt2t, t2drone
import gps_util

vis_out_path = pathlib.Path("vis_out_slam/")
gps_base = (0.8982182491908725, 0.11761678938800951)

def map_dist(detected_map: [gps_util.meter_pos]):
    poi_true_gps_positions_radiants, carposes = true_pos_from_droneimg_pxpos()
    poi_true_gps_positions_radiants = [gps_util.gps_to_meter(gps, gps_base) for gps in poi_true_gps_positions_radiants]
    true_blue_conepos = poi_true_gps_positions_radiants[poii_bluecones_range[0]:poii_bluecones_range[1]]
    true_yellow_conepos = poi_true_gps_positions_radiants[poii_yellowcones_range[0]:poii_yellowcones_range[1]]
    associations = []
    for tpi, tp in enumerate(true_blue_conepos):
        # get nearest detected_map
        mi = 0
        mdist = gps_util.meter_meter_to_dist(tp, detected_map[mi])
        for i, dp in enumerate(detected_map):
            if gps_util.meter_meter_to_dist(tp, dp) < mdist:
                mi = i
                mdist = gps_util.meter_meter_to_dist(tp, dp)
        # detected_map[mi] is nearest detection to tp
        if mdist < 1:
            associations.append((tpi, mi))
    truepositive = len(associations)
    falsenegative = len(true_blue_conepos)-len(associations)
    falsepositive = len(detected_map)-len(associations)
    avg_dist = np.average([gps_util.meter_meter_to_dist(detected_map[di], true_blue_conepos[ti])**2 for (ti, di) in associations])
    return (truepositive, falsenegative, falsepositive, avg_dist)


class SLAM():
    def measurment(self, vp_detections, car_mpos, car_heading):
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
        self.bc = []  # [(accuracy-score, (meter_north, meter_est))], with accuracy-score=1: positionis perfect, accuracy-score=0: contains no informatin
        self.yc = []
        self.cp = ((0, 0), 0)  # ((pos_north_meter, pos_est_meter), heading)
        self.last_vp_detections = []


    def measurment(self, vp_detections: [(int, gps_util.meter, gps_util.heading_radiants)], car_mpos: (gps_util.meter_north, gps_util.meter_east), car_heading: gps_util.heading_radiants):
        #print(f"trackingSLAM.measurment(seeing_cones={self.seeing_cones}, bc_mp={blue_cones_meterpos}, yc_mp={yellow_cones_meterpos}, car_mp={car_mpos})")
        car_mpos = np.array(car_mpos)
        # find all detections that should be visible in both last and current detections:
        current_detections_visible_from_old_carpos = []
        for (cls, dist, heading) in vp_detections:
            old_dist, old_heading = gps_util.meter_to_distazimuth(gps_util.distazimuth_to_meter((dist, to_range(heading+car_heading)))+car_mpos-self.cp[0])
            if old_dist < 10 and -1 < (old_heading-self.cp[1]) < 0.12:  # TODO check if + or - old heading
                current_detections_visible_from_old_carpos.append((cls, dist, heading))
        old_detections_visible_from_new_carpos = []
        for (cls, dist, heading) in self.last_vp_detections:
            old_dist, old_heading = gps_util.meter_to_distazimuth(gps_util.distazimuth_to_meter((dist, to_range(heading+car_heading)))+self.cp[0]-car_mpos)
            if old_dist < 10 and -1 < (old_heading-self.cp[1]) < 0.12:  # TODO check if + or - old heading
                old_detections_visible_from_new_carpos.append((cls, dist, heading))
        # math current_detections_visible_from_old_carpos to old_detections_visible_from_new_carpos and get position and heading difference -> odometry measurements
        # and get data associations
        # add positions
        self.bc += [(1-dist/10, car_mpos+gps_util.distazimuth_to_meter((dist, to_range(heading+car_heading)))) for (cls, dist, heading) in vp_detections if cls == 0]
        self.yc += [(1-dist/10, car_mpos+gps_util.distazimuth_to_meter((dist, to_range(heading+car_heading)))) for (cls, dist, heading) in vp_detections if cls == 0]
        self.cp = (car_mpos, car_heading)
        self.last_vp_detections = vp_detections


    def get_map(self) -> (gps_util.meter_pos, [gps_util.meter_pos], [gps_util.meter_pos]):
        # return (car position, blue_cones, yellow_cones), as list of all cones in meterpos
        return self.cp, k_means_cone_clustering(self.bc), k_means_cone_clustering(self.yc)


def k_means_cone_clustering(cones: [(float, gps_util.meter_pos)], dist=1):
    cones_clusteredpos = []  # cones_clusteredpos[i] = position
    cone_acc_mpos = [(acc, pos) for (acc, pos) in cones if acc > 0]
    cone_acc_mpos.sort(key=lambda x: x[0])
    while len(cone_acc_mpos) > 0:
        has_changed = True
        acc_sum, cluster = 0, []
        bc_current_pos = cone_acc_mpos[0][1]
        while has_changed:
            cluster = [(acc, bc) for (acc, bc) in cone_acc_mpos if gps_util.meter_meter_to_dist(bc, bc_current_pos) <= dist]
            if len(cluster) == 0:
                print(f"ERROR: trackingSLAM.get_map: unreachable code reached.\n cluster={cluster}\nbc_current={bc_current_pos}\ncone_acc_mpos={cone_acc_mpos}")
            acc_sum = sum([acc for (acc, bc) in cluster])
            tmp = np.array([0, 0])+sum([acc/acc_sum*np.array(bc) for (acc, bc) in cluster])  # weighted average of cone positions
            has_changed = (tmp[0] != bc_current_pos[0] or tmp[1] != bc_current_pos[1])
            bc_current_pos = tmp
        if acc_sum > 1:  # somewhere between 0.7 and 17 should be okay, but highly depends on framerate, visual detection range abd formula for accuracy.
            cone_acc_mpos = [(acc, bc) for (acc, bc) in cone_acc_mpos if gps_util.meter_meter_to_dist(bc, bc_current_pos) > dist]
            cones_clusteredpos.append(bc_current_pos)
        else:
            cone_acc_mpos = cone_acc_mpos[1:]  # remove the vertex that started this
    return cones_clusteredpos


def test_Slam():
    sensordatadict = read_csv("merged_rundata_csv/alldata_2022_12_17-14_43_59_id3.csv")
    gps_lat_time = sensordatadict["GNSS_latitude_UsbFlRec"+x_ending]
    gps_lat_val = sensordatadict["GNSS_latitude_UsbFlRec"]*np.pi/180
    gps_long_time = sensordatadict["GNSS_longitude_UsbFlRec"+x_ending]
    gps_long_val = sensordatadict["GNSS_longitude_UsbFlRec"]*np.pi/180
    gps_heading_time = sensordatadict["GNSS_heading_UsbFlRec"+x_ending]
    gps_heading_val = sensordatadict["GNSS_heading_UsbFlRec"]*np.pi/180
    poi_true_gpspos, carpos = true_pos_from_droneimg_pxpos()
    # carpos[drone3_frnr] = [np.array([lat, long]), np.array([lat, long]), np.array([lat, long]), np.array([lat, long])
    dronefrnr_carpos = [(frnr, carpos[frnr]) for frnr in carpos.keys()]
    dronefrnr_carpos.sort(key=lambda x: x[0])
    carpos_dv_time = [time for (time, value) in dronefrnr_carpos]
    carpos_dv_value = [np.array(value) for (time, value) in dronefrnr_carpos]
    base_gpspos = (gps_lat_val[0], gps_long_val[0])

    print("gps_lat_time = ", min(gps_lat_time), ", ", max(gps_lat_time))
    print("gps_long_time = ", min(gps_long_time), ", ", max(gps_long_time))
    print(f"test SLAM with camL3-images from (framenr, time) (1282, {camL2t(1282)}) to (2643, {camL2t(2643)})")
    car_pos = []
    use_gpspos = False
    max_spedmulti = 20
    slam_speedmutli_score = [(0, 0, 0, 0) for _ in range(1, max_spedmulti+1)]
    for speed_multi in range(1, max_spedmulti):
        print(f"speed_multi = {speed_multi}/{max_spedmulti}")
        noslam = trackingSLAM()
        all_blue_cones = []
        all_yellow_cones = []
        all_blue_cones_information = []
        all_yellow_cones_information = []
        for frnr in range(1282, 2643, speed_multi):  # frnr of camL3
            if use_gpspos:
                t = camL2t(frnr)
                t_long, i_long = get_at_time(x=gps_long_time, y=gps_long_val, t=t)
                t_lat = gps_lat_val[i_long]
                car_heading = gps_heading_val[i_long]
                car_gps_pos = (t_lat, t_long)
            else:
                carposes, _ = get_at_time(x=carpos_dv_time, y=carpos_dv_value, t=get_synced_frame(frnr))
                car_gps_pos = gps_util.carposs_to_gnsspos(carposes)
                car_heading = gps_util.carposs_to_heading(carposes)
            car_mpos = gps_util.gps_to_meter(car_gps_pos, base_gpspos)
            vp_detections = visual_pipeline(cam="camL3", framenr=frnr)
            #print(f"camL3_frnr={frnr}, bc_mp={blue_cones_meterpos}, yc_mp={yellow_cones_meterpos}")
            noslam.measurment(vp_detections, car_mpos, car_heading)

            car_pos.append(car_mpos)
            all_blue_cones += [car_mpos+gps_util.distazimuth_to_meter((dist, to_range(heading+car_heading))) for (cls, dist, heading) in vp_detections if cls == 0]
            all_yellow_cones += [car_mpos+gps_util.distazimuth_to_meter((dist, to_range(heading+car_heading))) for (cls, dist, heading) in vp_detections if cls == 1]
            all_blue_cones_information += [(1-0.1*dist, car_mpos+gps_util.distazimuth_to_meter((dist, to_range(heading+car_heading)))) for (cls, dist, heading) in vp_detections if cls == 0]
            all_yellow_cones_information += [(1-0.1*dist, car_mpos+gps_util.distazimuth_to_meter((dist, to_range(heading+car_heading)))) for (cls, dist, heading) in vp_detections if cls == 1]


        #print(f"all_blue_cones{speed_multi} = {len(all_blue_cones)}")
        #plot cones
        true_cone_meterpos = [gps_util.gps_to_meter(conepos, base_gpspos) for conepos in poi_true_gpspos[poii_bluecones_range[0]:]]
        cp, detected_blue_cones, detected_yellow_cones = noslam.get_map()
        clusted_blue_cones = k_means_cone_clustering(all_blue_cones_information)
        clusterd_yellow_cones = k_means_cone_clustering(all_yellow_cones_information)
        #plot_cones(car_pos=car_pos, blue_cones=clusted_blue_cones, yellow_cones=clusterd_yellow_cones, name=f"slam_new_clustered{speed_multi}", true_conepos=true_cone_meterpos, save_dir=f"{vis_out_path}slam_new_clustering_{speed_multi}.png")
        #plot_cones(car_pos=car_pos, blue_cones=detected_blue_cones, yellow_cones=detected_yellow_cones, name=f"slam_clustered{speed_multi}", true_conepos=true_cone_meterpos, save_dir=f"{vis_out_path}slam_clustering_{speed_multi}.png")
        #plot_cones(car_pos=car_pos, blue_cones=all_blue_cones, yellow_cones=all_yellow_cones, name=f"slaM_no{speed_multi}", true_conepos=true_cone_meterpos, save_dir=f"{vis_out_path}slam_no_{speed_multi}.png")

        # get average distance between slam-result cones and true pos cones, and number of non-detected cones, number of color misclassification, number of detected cones that doesnt exist.
        falsenegatives = 0  # number of true cones that were not detected
        falsepositives = 0  # number of detected cones that were not true
        truepositives = 0
        tot_dist = 0
        for (true_pos, detected_pos) in [(poi_true_gpspos[poii_bluecones_range[0]:poii_bluecones_range[1]], detected_blue_cones)]:  #, (poi_true_gpspos[poii_yellowcones_range[0]:poii_yellowcones_range[1]], detected_yellow_cones)]:
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
            #print(f"len(detected_pos) = {len(detected_pos)}, len(true_pos) = {len(true_pos)}")
            #print(f"truepositives = {truepositives}, falsenegatives = {falsenegatives}, falsepositives = {falsepositives}, tot_dist = {tot_dist}")
        #print(f"cluster & {speed_multi} & {tot_dist/truepositives} & {truepositives} & {falsenegatives} & {falsepositives} & ? \\\\")
        slam_speedmutli_score[speed_multi] = (tot_dist/truepositives, truepositives, falsenegatives, falsepositives)
    slam_speedmutli_score = slam_speedmutli_score[1:]
    for speed_multi in range(len(slam_speedmutli_score)):
        (avg_dist, truepositives, falsenegatives, falsepositives) = slam_speedmutli_score[speed_multi]
        print(f"cluster & {speed_multi} & {avg_dist} & {truepositives} & {falsenegatives} & {falsepositives} & ? \\\\")
    return slam_speedmutli_score  # (avg_dist, truepositives, falsenegatives, falsepositives)

def plot_factorgraph(car_poses: [(int, gps_util.meter_north, gps_util.meter_east, gps_util.heading_radiants)], cone_positions: [(int, int, gps_util.meter_north, gps_util.meter_east)], constraints: [(int, int)]):
    # build factor graph from droneview measruements
    poi_true_gps_positions_radiants, carpos = true_pos_from_droneimg_pxpos()
    nid = 0
    car_poses = []  # (nid, pos_n, pos_e, heading

    for k in [2341, 2470, 2600]:
        print(f"k={k}")
        pos_n, pos_e = gps_util.gps_to_meter(gps_util.average(carpos[k]), gps_base)
        car_poses.append((nid, pos_n, pos_e, 0))
        nid += 1
    cone_positions = []  # (nid, cls, pos_n, pos_e)
    for i, v in enumerate(poi_true_gps_positions_radiants):
        pos_n, pos_e = gps_util.gps_to_meter(v, gps_base)
        if i > 8 and -20 < pos_n < 0 and pos_e > 2.5 and i not in [11, 73, 74, 81, 82, 41, 42]:
            cone_positions.append((i, 0 if i < 47 else 1, pos_n, pos_e))
    vis_cones = set([i for (i, cls, pos_n, pos_e) in cone_positions])
    cones_visible_from_0 = [(0, t) for t in [10, 46, 45, 44, 43, 42, 41, 49, 39, 87, 86, 85, 84, 83, 81, 80, 79] if t in vis_cones]
    cones_visible_from_1 = [(1, t) for t in [46, 45, 44, 43, 42, 84, 83, 82, 41] if t in vis_cones]
    constraints = [(0, 1), (1, 2)]+cones_visible_from_0+cones_visible_from_1 # (nid, nid)

    # plot factor graph
    fig, axe = plt.subplots()
    axe.set_title("factor graph")
    axe.set_xlabel("meter_east")
    axe.set_ylabel("meter_north")
    nid_pos = {}
    for (nid, pos_n, pos_e, heading) in car_poses:
        nid_pos[nid] = (pos_n, pos_e)
    for ((nid, cls, pos_n, pos_e)) in cone_positions:
        nid_pos[nid] = (pos_n, pos_e)
    for (i0, i1) in constraints:
        val0, val1 = nid_pos[i0], nid_pos[i1]
        axe.plot([val0[1], val1[1]], [val0[0], val1[0]], color="black")
    axe.scatter([pos_e for (nid, pos_n, pos_e, heading) in car_poses], [pos_n for (nid, pos_n, pos_e, heading) in car_poses], c='green')
    axe.scatter([car_poses[0][2]], [car_poses[0][1]], c='green', label='car positions')
    axe.scatter([pos_e for (nid, cls, pos_n, pos_e) in cone_positions], [pos_n for (nid, cls, pos_n, pos_e) in cone_positions], c=['blue' if cls==0 else 'yellow' for (nid, cls, pos_n, pos_e) in cone_positions])
    for nid in nid_pos.keys():
        axe.text(nid_pos[nid][1], nid_pos[nid][0], str(nid))
    axe.grid()
    axe.legend()
    fig.show()
    fig.savefig(vis_out_path/"factor_graph.png")


def test_slamfrontend():
    plot_vp = False
    bbi_poii_label_dir = "C:/Users/Idefix/PycharmProjects/tmpProject/vp_labels/bbi_poii"
    sensordatadict = read_csv("merged_rundata_csv/alldata_2022_12_17-14_43_59_id3.csv")
    gnss_lat_time = sensordatadict["GNSS_latitude_UsbFlRec"+x_ending]
    gnss_lat_val = smothing(gnss_lat_time, sensordatadict["GNSS_latitude_UsbFlRec"]*np.pi/180, 2)
    gnss_long_time = sensordatadict["GNSS_longitude_UsbFlRec"+x_ending]
    gnss_long_val = smothing(gnss_long_time, sensordatadict["GNSS_longitude_UsbFlRec"]*np.pi/180, 2)
    gnss_heading_time = sensordatadict["GNSS_heading_UsbFlRec"+x_ending]
    gnss_heading_val = sensordatadict["GNSS_heading_UsbFlRec"]*np.pi/180  # TODO smothing breaks when it jumps by 2pi.
    print(f"gnss_heading_val: min, max = {min(gnss_heading_val)}, {max(gnss_heading_val)}")
    poi_true_gpspos, dronefrmrn_carposes = true_pos_from_droneimg_pxpos()
    poi_true_mpos = [(poi, gps_util.gps_to_meter(true_gpspos, gps_base)) for (poi, true_gpspos) in enumerate(poi_true_gpspos)]

    fully_labeld_frames = [int(str(f).replace(".txt", "").split("_")[-1]) for f in os.listdir(bbi_poii_label_dir)]
    fully_labeld_frames.sort()
    camL3frnr_vpdet_carpose = {}  # camL3frnr_vpdet_carpose[camL3_frnr] = (vpdet=[(poii, (dist, bearing))], carpose=((posm_n, posm_e), heading))
    for f in fully_labeld_frames:
        bb_kp_poii = get_boundingboxes_keypoints_poii("camL3", f)
        vp_det = [(poii, custom_PnP(kp, bb)) for (bb, kp, poii) in bb_kp_poii if kp is not None and poii != -1]  # [(poii, (dist, bearing))]
        t = f/20+55.98  # camL3_frnr to ssd-time
        lat, lati = get_at_time(gnss_lat_time, gnss_lat_val, t)
        #long, longi = get_at_time(gnss_long_time, gnss_long_val, t)
        #heading, headingi = get_at_time(gnss_heading_time, gnss_heading_val, t)
        # assert lati == longi == headingi was true
        carpose = (gps_util.gps_to_meter((lat, gnss_long_val[lati]), gps_base), gnss_heading_val[lati] if gnss_heading_val[lati] != 0 else 2.67)  # gnss_heading_val == 0 -> invalid/no measurement (before car started moving) -> use value of droneview
        drone3_frame = 10*int(get_synced_frame(f)/10)
        if drone3_frame == 2340:
            drone3_frame = 2341
        if drone3_frame == 2370:
            drone3_frame = 2371
        if drone3_frame == 2330:
            drone3_frame = 2342
        if drone3_frame in dronefrmrn_carposes.keys():
            true_carposes = dronefrmrn_carposes[drone3_frame]
            true_carpose = (gps_util.gps_to_meter(gps_util.carposs_to_gnsspos(true_carposes), gps_base), gps_util.carposs_to_heading(true_carposes))
            #print(f"gnss_carpose = {carpose}\ntrue_carpose = {true_carpose}")
            #carpose = true_carpose  # use true carpose
        else:
            print(f"drone3_frame {drone3_frame} has not annotated car position")
        #est_pos = car_posm+gps_util.distazimuth_to_meter((dist, to_range(car_bearing+heading)))
        est_pos = [(poii, carpose[0]+gps_util.distazimuth_to_meter((dist, to_range(carpose[1]+bearing)))) for (poii, (dist, bearing)) in vp_det]
        camL3frnr_vpdet_carpose[f] = (vp_det, carpose)
        if f in [1282, 1400, 2032] and plot_vp:
            plot_cones([carpose[0], carpose[0]+gps_util.distazimuth_to_meter((1, carpose[1]))], [pos for (poi, pos) in est_pos], [pos for (poi, pos) in poi_true_mpos[9:]], f"est_pos frame {f} in test_slamfrontend")

    if plot_vp:
        # plot ?
        fig, axe = plt.subplots()
        axe.set_title("all detections")
        cones_estposes = []
        for k in camL3frnr_vpdet_carpose.keys():
            (vp_det, carpose) = camL3frnr_vpdet_carpose[k]
            cones_estposes += [carpose[0]+gps_util.distazimuth_to_meter((dist, to_range(bearing+carpose[1]))) for (poi, (dist, bearing)) in vp_det]
        axe.scatter([e for (n, e) in cones_estposes], [n for (n, e) in cones_estposes], color="blue", label="estpos")
        axe.scatter([e for (poi, (n, e)) in poi_true_mpos[9:]], [n for (poi, (n, e)) in poi_true_mpos[9:]], marker='x', color="black", label="truepos")
        axe.legend()
        fig.show()


    if plot_vp:
        # plot all detections of cone 42.
        tmpk = list(camL3frnr_vpdet_carpose.keys())
        tmpk.sort()
        cone42_poses = []
        carposes = []
        for camL3_frnr in tmpk:
            vp_det, carpose = camL3frnr_vpdet_carpose[camL3_frnr]
            cone42 = [pos for (poi, pos) in vp_det if poi == 42]
            assert len(cone42) == 0 or len(cone42) == 1  # one cane can be detected at max once.
            if len(cone42) == 1:
                dist, heading = cone42[0]
                cone42_poses.append(carpose[0]+gps_util.distazimuth_to_meter((dist, to_range(heading+carpose[1]))))
                carposes.append(carpose[0])
        fig, axe = plt.subplots()
        axe.set_title("cone42")
        axe.scatter([e for (n, e) in cone42_poses], [n for (n, e) in cone42_poses], color="blue", label="positions where cone 42 was detected")
        axe.plot([e for (n, e) in carposes], [n for (n, e) in carposes], color="black", label="car positions")
        axe.scatter([poi_true_mpos[42][1][1]], [poi_true_mpos[42][1][0]], marker='x', color="black")
        axe.legend()
        fig.show()


    if plot_vp:
        # plot all detectionis from frame 1282 to 1291
        camL3_frnr = 1282
        n = 5
        # plot true cone positions, detected cone positions, connections.
        fig, axe = plt.subplots()
        axe.set_title(f"cones detected from {camL3_frnr} to {camL3_frnr+n}")
        # plot true positions of detected cones
        allpoii = set()
        for i in range(n):
            allpoii = allpoii.union(set([poii for (poii, (dist, bearing)) in camL3frnr_vpdet_carpose[camL3_frnr+i][0]]))
        tmp = [(n, e) for (poi, (n, e)) in poi_true_mpos if poi in allpoii]
        axe.scatter([e for (n, e) in tmp], [n for (n, e) in tmp], marker='x', color="black", label="true cone positions")
        colors = ["gray", "brown", "blue", "olive", "green", "purple", "cyan", "pink", "orange", "red"]
        carposees = []
        for i in range(n):
            vpdet, carpose = camL3frnr_vpdet_carpose[camL3_frnr+i]
            tmp = [carpose[0]+gps_util.distazimuth_to_meter((dist, to_range(bearing+carpose[1]))) for (poii, (dist, bearing)) in vpdet]
            carposees.append(carpose[0])
            axe.scatter([e for (n, e) in tmp], [n for (n, e) in tmp], color=colors[i], label=f"vpdet[{camL3_frnr-i}]", s=5)
        axe.plot([e for (n, e) in carposees], [n for (n, e) in carposees], color="black", label="car positions")
        for poii in allpoii:
            truepos = [(n, e) for (poi, (n, e)) in poi_true_mpos if poi == poii][0]
            poses = [truepos]
            for i in range(n):
                vpdet, carpose = camL3frnr_vpdet_carpose[camL3_frnr+i]
                estpos = [carpose[0]+gps_util.distazimuth_to_meter((dist, to_range(bearing+carpose[1]))) for (poi, (dist, bearing)) in vpdet if poi == poii]
                if len(estpos) == 1:
                    poses.append(estpos[0])
            poses.append(truepos)
            axe.plot([p[1] for p in poses], [p[0] for p in poses], color="gray", linewidth=1)
        axe.legend()
        fig.savefig(vis_out_path/f"sfe_cones_detected_in_{camL3_frnr}_to_{n}.png")
        fig.show()

    # test data association algorithms
    max_speedmult = 32
    slam_speedmutli_score = [(0, 0) for _ in range(max_speedmult)]  # (correct data associations/total_correct_associations, wrong data associations/total_associations)
    for speedmult in range(max_speedmult):
        tota_correct = 0
        tota_wrong = 0
        tota_trueno = 0
        tota_falseno = 0
        tota_tot = 0
        for camL3_frnr in camL3frnr_vpdet_carpose.keys():
            vpdet, carpose = camL3frnr_vpdet_carpose[camL3_frnr]
            # get if nearest neighbour is the same as true data association
            if camL3_frnr-speedmult in camL3frnr_vpdet_carpose.keys():
                lf_vpdet, lf_carpose = camL3frnr_vpdet_carpose[camL3_frnr-speedmult]
                lf_allpoii = [poii for (poii, (dist, bearing)) in lf_vpdet]
                false_no = 0
                true_no = 0
                correct = 0
                wrong = 0
                # for each detection in vpdet: get nearest cone in lf_vpdet, if same poii: correct += 1
                delta_carpose = (carpose[0]-lf_carpose[0], to_range(carpose[1]-lf_carpose[1]))
                for (poii, (dist, bearing)) in vpdet:
                    # get detection from last frame that is nearest to (dist, bearing)
                    est_cone_pos = delta_carpose[0]+gps_util.distazimuth_to_meter((dist, to_range(bearing+delta_carpose[1])))

                    min_lfpoii = lf_vpdet[0][0]
                    nnd = gps_util.meter_meter_to_dist(gps_util.distazimuth_to_meter(lf_vpdet[0][1]), est_cone_pos)  # dont need to add lf_carpose to lf_vpdet, cause only relative to current frame relevant
                    for (lf_poii, (lf_dist, lf_bearing)) in lf_vpdet:
                        lf_est_cone_pos = gps_util.distazimuth_to_meter((lf_dist, lf_bearing))
                        if gps_util.meter_meter_to_dist(lf_est_cone_pos, est_cone_pos) < nnd:
                            min_lfpoii = lf_poii
                            nnd = gps_util.meter_meter_to_dist(lf_est_cone_pos, est_cone_pos)
                    # if poii==nni: correct association
                    if nnd > 1:
                        if poii in lf_allpoii:
                            false_no += 1
                        else:
                            true_no += 1
                    elif poii == min_lfpoii:
                        correct += 1
                    else:
                        wrong += 1
                tota_correct += correct
                tota_wrong += wrong
                cones_visible_in_both_frames = len(set([poii for (poii, distbearing) in vpdet]).intersection(set([poii for (poii, distbearing) in lf_vpdet])))
                tota_tot += cones_visible_in_both_frames
                tota_trueno += true_no
                tota_falseno += false_no
                #print(f"association between frame lf={camL3_frnr-1} and cf={camL3_frnr}: cones on lf={len(lf_vpdet)}, cf={len(vpdet)}. cones on both={cones_visible_in_both_frames}. correct={correct}, wrong={wrong}, true_no={true_no}, false_no={false_no}")
            #print(f"{camL3_frnr}on frame: {correct_associations} of {len(vpdet)-no_association} data associations correct (from {len(vpdet)} total cones in frame)")
        #print(f"speedmult={speedmult}: total data associations to true map: correct={tott_correct}, wrong={tott_wrong}, no={tott_no}")
        print(f"speedmult={speedmult}: total data associations between frames {speedmult} removed: correct={tota_correct}, wrong={tota_wrong} of {tota_tot} cones total. true_no={tota_trueno}, false no={tota_falseno}.")
        slam_speedmutli_score[speedmult] = (tota_correct/tota_tot, tota_wrong/(tota_correct+tota_wrong), tota_falseno/tota_tot)  # (correct data associations/total_correct_associations, wrong data associations/total_associations)
        # the simple nearest neighbour approach seem to fail at about speedmult 6. (662 correct, 38 wrong of 949 cones total) speedmult 7: ( correct=535, wrong=62 of 932 cones total)
    # plot score(front-end) over speed_multi
    fig, axe = plt.subplots()
    axe.set_title("frontend data associations")
    axe.set_xlabel("speed of car in 2m/s")
    axe.set_ylabel("0.01*percentage of data associations")
    axe.plot(np.array(range(len(slam_speedmutli_score))), np.array([correct for (correct, falsepositive, falsenegative) in slam_speedmutli_score]), color="green")
    axe.scatter(np.array(range(len(slam_speedmutli_score))), np.array([correct for (correct, falsepositive, falsenegative) in slam_speedmutli_score]), marker='o', color="green", label="gnn detected correct/total correct")
    axe.plot(np.array(range(len(slam_speedmutli_score))), np.array([falsepositive for (correct, falsepositive, falsenegative) in slam_speedmutli_score]), color="green")
    axe.scatter(np.array(range(len(slam_speedmutli_score))), np.array([falsepositive for (correct, falsepositive, falsenegative) in slam_speedmutli_score]), marker='x', color="green", label="gnn falsepositive associations/total detections", )
    axe.plot(np.array(range(len(slam_speedmutli_score))), np.array([falsenegative for (correct, falsepositive, falsenegative) in slam_speedmutli_score]), color="green")
    axe.scatter(np.array(range(len(slam_speedmutli_score))), np.array([falsenegative for (correct, falsepositive, falsenegative) in slam_speedmutli_score]), marker='v', color="green", label="gnn falsenegative associations/total correct", )
    axe.legend()
    axe.grid()
    fig.show()
    fig.savefig(vis_out_path/"frontend_data_association.png")


def read_g2o_graphsave(name: str, time_of_save: str = "after"):
    # return the factor graph stored by g2o.
    conevertex_metainfo = {}
    with open(f"C:/src/g2o_git/bin/{name}conevertex_metainfo.txt") as f:
        for line in f.readlines():
            (coneid, information, color) = line.split(",")
            conevertex_metainfo[int(coneid)] = (float(information), int(color))
            file = pathlib.Path(f"C:/src/g2o_git/bin/{name}_{time_of_save}.g2o")

    vertex_se2 = {}
    vertex_point_xy = {}
    edge_se2 = set()
    edge_se2_pointxy = set()
    with open(file) as f:
        # TUTORIAL_PARAMS_SE2_OFFSET 0 0 0 0
        # TUTORIAL_VERTEX_SE2 id posx posy heading
        # TUTORIAL_VERTEX_POINT_XY id posx posy
        # TUTORIAL_EDGE_SE2 0 1 18.0584 -0.870688 2.67041 2.50361 0 0 2.50361 0 4109.14
        # TUTORIAL_EDGE_SE2_POINT_XY 1 2 0 1.1884 -1.02397 -157.32 0 -157.32
        for line in f.readlines()[1:]:
            s = line.split(" ")
            if s[0] == "TUTORIAL_VERTEX_SE2":
                assert len(s) == 5
                vertex_se2[int(s[1])] = (float(s[2]), float(s[3]), float(s[4]))  # id: posn, pose, heading
            elif s[0] == "TUTORIAL_VERTEX_POINT_XY":
                assert len(s) == 4
                key = int(s[1])
                vertex_point_xy[key] = (float(s[2]), float(s[3]), conevertex_metainfo[key][1], conevertex_metainfo[key][0])  # id: posn, pose, color, (negative)information
            elif s[0] == "TUTORIAL_EDGE_SE2":
                assert len(s) == 12
                tmp = (int(s[1]), int(s[2]), float(s[3]), float(s[4]), float(s[5]), float(s[6]), float(s[7]), float(s[8]), float(s[9]))  # vertex_from, vertex_to, constraint_posx, constraint_posy, constraint_heading, 6 unknown values
                edge_se2.add(tmp)
            elif s[0] == "TUTORIAL_EDGE_SE2_POINT_XY":
                assert len(s) == 9
                tmp = (int(s[1]), int(s[2]), float(s[3]), float(s[4]), float(s[5]), float(s[6]), float(s[7]), float(s[7]))  # vertex_from, vertex_to, constraint_posx, constraint_posy, 4 unknown values
                edge_se2_pointxy.add(tmp)
            else:
                if line not in ["FIX 0\n"]:
                    print(f"ERROR: line = {line} could not be parsed by read_g2o_graphsave({file}")

    # vertex_se2[key] = (pos_north, pos_east, heading)
    # vertex_point_xy[key] = (pos_north, pos_east, color, (negative)information)
    # edge_se2 = (key0, key1, distance, distance_quer, headingdiff, 6 ?information values)
    # edge_se2_pointxy = (key0, key1, distance, distance_quer, 4 ? information valuse)
    return vertex_se2, vertex_point_xy, edge_se2, edge_se2_pointxy

def display_g2o_graph(name: str):
    poi_true_gps_positions_radiants, true_carposes = true_pos_from_droneimg_pxpos()
    poi_true_mpos = [gps_util.gps_to_meter(tgp, (0.8982182491908725, 0.11761678938800951)) for tgp in poi_true_gps_positions_radiants[poii_bluecones_range[0]:]]
    for time_of_save in ["after"]:  # "before", "optimised",
        vertex_se2, vertex_point_xy, edge_se2, edge_se2_pointxy = read_g2o_graphsave(name, time_of_save)

        print(f"\nname time_of_save = {name} {time_of_save}")
        print("vertex_se2 =", len(vertex_se2))
        print("vertex_point_xy =", len(vertex_point_xy))
        print("edge_se2 =", len(edge_se2))
        print("edge_se2_pointxy =", len(edge_se2_pointxy))

        carpose_keys = list(vertex_se2.keys())
        carpose_keys.sort()
        fig, axe = plt.subplots()
        axe.set_title(f"g2o graph {name} at {time_of_save}")
        for vp in edge_se2_pointxy:
            if vp[1] in [1, 2]:
                p0, p1 = vertex_se2[vp[0]], vertex_point_xy[vp[1]]
                axe.plot([p0[1], p1[1]], [p0[0], p1[0]], color="red")
        for odometry in edge_se2:
            if odometry[0] != 0:  # dont map GNSS constraint
                p0, p1 = vertex_se2[odometry[0]], vertex_se2[odometry[1]]
                axe.plot([p0[1], p1[1]], [p0[0], p1[0]], color="green")
        axe.scatter([pose for (posn, pose) in poi_true_mpos], [posn for (posn, pose) in poi_true_mpos], marker='x', color="black")
        axe.scatter([vertex_se2[k][1] for k in carpose_keys], [vertex_se2[k][0] for k in carpose_keys], color="black", label="car poses", s=2)

        cone_abspos = [(vertex_point_xy[k], vertex_point_xy[k][2]) for k in vertex_point_xy.keys()]
        cone_abspos = [(pos, 0) for pos in k_means_cone_clustering([(vertex_point_xy[k][3], np.array([vertex_point_xy[k][0], vertex_point_xy[k][1]])) for k in vertex_point_xy.keys()])]
        print("vertex_point_xy_clusterd =", len(cone_abspos))

        axe.scatter([pose for ((posn, pose), cls) in cone_abspos], [posn for ((posn, pose), cls) in cone_abspos], color=["blue" if cls==0 else "yellow" for ((posn, pose), cls) in cone_abspos], label="conepos")
        #axe.scatter([pose for (posn, pose) in clustered_cone_abspos], [posn for (posn, pose) in clustered_cone_abspos], color="yellow", label="clustered conepos")
        # plot true path
        true_carpose_keys = list(true_carposes.keys())
        true_carpose_keys.sort()
        true_carmpos = [gps_util.gps_to_meter(gps_util.carposs_to_gnsspos(true_carposes[k]), gps_base) for k in true_carpose_keys]
        axe.plot([pose for (posn, pose) in true_carmpos], [posn for (posn, pose) in true_carmpos], color="black", label="true car path")
        axe.grid()
        axe.legend()
        fig.savefig(vis_out_path/f"slam_{name}_{time_of_save}.png")
        fig.show()
        return map_dist([abspos for (abspos, cls) in cone_abspos if cls == 0])


def get_map_dist(slam_name, speedmult):
    name = slam_name+"_"+str(speedmult)
    vertex_se2, vertex_point_xy, edge_se2, edge_se2_pointxy = read_g2o_graphsave(name, "after")
    cone_abspos = [(pos, 0) for pos in k_means_cone_clustering([(vertex_point_xy[k][3], np.array([vertex_point_xy[k][0], vertex_point_xy[k][1]])) for k in vertex_point_xy.keys()])]
    (truepositive, falsenegative, falsepositive, avg_dist) = map_dist([abspos for (abspos, cls) in cone_abspos if cls == 0])
    return (truepositive, falsenegative, falsepositive, get_path_dist(vertex_se2, speedmult))


def writ_vpaux():
    from dataset_transform import bb_overlap
    # auxilary visual pipeline "output" for orange cones.
    # format: camL3_frnr,dist,heading,dist,heading
    out = []
    poi_true_gps_positions_radiants, carposes = true_pos_from_droneimg_pxpos()
    carposes_time = list(carposes.keys())
    carposes_time.sort()
    carposes_val = [(gps_util.carposs_to_gnsspos(carposes[k]), gps_util.carposs_to_heading(carposes[k])) for k in carposes_time]  # ((meter, meter), heading)
    carposes_val = [np.array([a, b, c]) for ((a, b), c) in carposes_val]
    blue_trupos = poi_true_gps_positions_radiants[9]
    yellow_truepos = poi_true_gps_positions_radiants[47]
    for camL3_frnr in range(1280, 2643):
        bbs = [bb for (bb, kp, poii) in get_boundingboxes_keypoints_poii("camL3", camL3_frnr, filter=False) if bb[0] == 2]
        if len(bbs) > 0:
            #tmp = [gps_util.gps_to_distazimuth(poi_true_gpspos[poii], car_gpspos) for (bb, kp, poii) in bb_kp_poii]
            #true_dist = [dist for (dist, azimuth) in tmp]
            #true_angle = [to_range(azimuth-car_bearing) for (dist, azimuth) in tmp]
            true_carpose, _ = get_at_time(carposes_time, carposes_val, get_synced_frame(camL3_frnr))
            distb, headingb = gps_util.gps_to_distazimuth(blue_trupos, (true_carpose[0], true_carpose[1]))
            disty, headingy = gps_util.gps_to_distazimuth(yellow_truepos, (true_carpose[0], true_carpose[1]))
            headingb = to_range(headingb-true_carpose[2])
            headingy = to_range(headingy-true_carpose[2])
            if camL3_frnr < 1290:
                print(f"{camL3_frnr}: carpose = {true_carpose} ({0} meter from last")
            out.append(f"{camL3_frnr},{distb},{headingb},{disty},{headingy}\n")
    with open("C:/Users/Idefix/PycharmProjects/tmpProject/vp_labels/camL3_orangedistheading_l_r.txt", 'w') as f:
        f.writelines(out)


def write_dronefrnr_gnsscarpose():
    filename = "C:/Users/Idefix/PycharmProjects/tmpProject/vp_labels/droneview/droneFrnr_gnssMPose2.txt"
    sdd = read_csv("merged_rundata_csv/alldata_2022_12_17-14_43_59_id3.csv")
    lat_gnns_time = sdd["GNSS_latitude_UsbFlRec"+x_ending]
    lat_gnss_value = sdd["GNSS_latitude_UsbFlRec"]*np.pi/180
    long_gnss_val = sdd["GNSS_longitude_UsbFlRec"]*np.pi/180
    gnss_heading_val = sdd["GNSS_heading_UsbFlRec"]*np.pi/180
    vabs_gnss_time = sdd["GNSS_speed_over_ground_UsbFlRec"+x_ending]
    vabs_gnss_val = sdd["GNSS_speed_over_ground_UsbFlRec"]
    out = []  # (drone_frnr,meter_north,meter_east,vabs,frnr,pose)
    old_heading = gnss_heading_val[0]
    # TODO only write on frames where new GNSS measurement is available
    last_gnss_val = (0, 0)
    for i in range(len(lat_gnss_value)):
        drone_frnr = t2drone(ssdt2t(lat_gnns_time[i]))
        if lat_gnss_value[i] != last_gnss_val[0] or long_gnss_val[i] != last_gnss_val[1]:
            # new gnss measurement -> append to out.
            last_gnss_val = (lat_gnss_value[i], long_gnss_val[i])
            lat, long = gps_util.gps_to_meter((lat_gnss_value[i], long_gnss_val[i]), gps_base)
            heading = gnss_heading_val[i]
            vabs, _ = get_at_time(vabs_gnss_time, vabs_gnss_val, lat_gnns_time[i])
            yawrate = heading-old_heading
            old_heading = heading
            out.append(f"{drone_frnr},{lat},{long},{heading},{vabs},{yawrate}\n")
            #out.append((drone_frnr, lat, long, heading, vabs, yawrate))
    with open(filename, 'w') as f:
        f.writelines(out)


def get_path_dist(vertex_se2, speedmult):
    vertex_se2_keys = list(vertex_se2.keys())
    vertex_se2_keys.sort()  # k=vertex_se2_keys[0] -> camL3_frnr=1280
    vertex_se2_keys_camL3frnr = {}
    for i, k in enumerate(vertex_se2_keys):
        vertex_se2_keys_camL3frnr[k] = 1280+speedmult*i
    poi_true_gps_positions_radiants, carposes = true_pos_from_droneimg_pxpos()
    carposes_keys = list(carposes.keys())
    carposes_keys.sort()
    car_mpos = [gps_util.gps_to_meter(gps_util.carposs_to_gnsspos(carposes[k]), gps_base) for k in carposes_keys]
    # carposes: drone3_frnr -> carposes
    # vertex_se2: k -> est_carpose, with get_synced_frame(vertex_se2_keys_camL3frnr[k]) = drone3_frnr
    pos_diff_dv_g2oest = []
    for k in vertex_se2_keys:
        drone3_frnr = get_synced_frame(vertex_se2_keys_camL3frnr[k])
        true_car_mpos, _ = get_at_time(carposes_keys, car_mpos, drone3_frnr)
        dist = gps_util.meter_meter_to_dist(vertex_se2[k], true_car_mpos)
        pos_diff_dv_g2oest.append(dist)
    #print(f"avg(pos_diff_dv_g2oest) = ", np.average(pos_diff_dv_g2oest))
    #print(f"avg(pos_diff_dv_g2oest[0.5:]) = ", np.average(pos_diff_dv_g2oest[int(0.5*len(pos_diff_dv_g2oest)):]))
    #print("max(pos_diff_dv_g2oest) = ", max(pos_diff_dv_g2oest))
    #plot_and_save("pos_diff_dv_g2oest", x_in=vertex_se2_keys, ys=[pos_diff_dv_g2oest])
    return np.average(pos_diff_dv_g2oest)


def odometry_error():
    poi_true_gps_positions_radiants, carposes = true_pos_from_droneimg_pxpos()
    carposes_keys = list(carposes.keys())
    carposes_keys.sort()
    car_mpos_heading = [(gps_util.gps_to_meter(gps_util.carposs_to_gnsspos(carposes[k]), gps_base), gps_util.carposs_to_heading(carposes[k])) for k in carposes_keys]

    sdd = read_csv("merged_rundata_csv/alldata_2022_12_17-14_43_59_id3.csv")
    gnss_heading_val = sdd["GNSS_heading_UsbFlRec"]*np.pi/180
    gnssheading_time = sdd["GNSS_heading_UsbFlRec"+x_ending]
    vabs_gnss_time = sdd["GNSS_speed_over_ground_UsbFlRec"+x_ending]
    vabs_gnss_val = sdd["GNSS_speed_over_ground_UsbFlRec"]
    old_heading = gnss_heading_val[0]
    old_carpose = car_mpos_heading[0]
    for drone_frnr in range(2341, 4032):
        t = t2ssdt(drone2t(drone_frnr))+2
        heading, _ = get_at_time(gnssheading_time, gnss_heading_val, t)
        vabs, _ = get_at_time(vabs_gnss_time, vabs_gnss_val, t)
        yawrate = heading-old_heading
        old_heading = heading

        carmpos_dv, _ = get_at_time(carposes_keys, car_mpos_heading, drone_frnr)
        vabs_dv = gps_util.meter_meter_to_dist(carmpos_dv[0], old_carpose[0])*25
        yawrate_dv = carmpos_dv[1]-old_carpose[1]
        old_carpose = carmpos_dv


def main():
    write_dronefrnr_gnsscarpose()
    exit(0)
    # plot SLAM truepositive over speedmult
    g2o_slam_speedmulti_scores = {}
    for name in ["full", "noFrontend"]:
        tests = [(name, i) for i in range(1, 20)]  # (name, speedmult
        out = ""
        fullslam_speedmutli_score = []
        for (name, speedmult) in tests:
            (truepositive, falsenegative, falsepositive, path_dist) = get_map_dist(name, speedmult)  # (truepositive, falsenegative, falsepositive, avg_dist)
            out += f"{name} & {speedmult} & {path_dist} & {truepositive} & {falsenegative} & {falsepositive} & ? \\\\\n"
            fullslam_speedmutli_score.append((path_dist, truepositive, falsenegative, falsepositive))
        g2o_slam_speedmulti_scores[name] = fullslam_speedmutli_score
        print(out)

    # plot score(slam) over speed_multi

    fig, (axe_tp, axe_fp, axe_pd) = plt.subplots(3)
    axe_tp.set_title("true positive")
    axe_fp.set_title("false positive")
    axe_pd.set_title("path dist")
    axe_tp.set_xlabel("speed of car in 2m/s")
    axe_fp.set_xlabel("speed of car in 2m/s")
    axe_pd.set_xlabel("speed of car in 2m/s")
    axe_tp.set_ylabel("detections")
    axe_fp.set_ylabel("number of")
    axe_pd.set_ylabel("avg dist [m]")
    #slam_speedmutli_score = test_Slam()
    axe_tp.plot([1, 1+max(len(g2o_slam_speedmulti_scores["full"]), len(g2o_slam_speedmulti_scores["noFrontend"]))], [38, 38], label="true number of blue cones", color="blue")

    #axe.plot(np.array(range(1, 1+len(slam_speedmutli_score))), np.array([truepositives for (avg_dist, truepositives, falsenegatives, falsepositives) in slam_speedmutli_score]), color="green")
    #axe.scatter(np.array(range(1, 1+len(slam_speedmutli_score))), np.array([truepositives for (avg_dist, truepositives, falsenegatives, falsepositives) in slam_speedmutli_score]), marker='o', label="cluster_truepositives", color="green")
    #axe.plot(np.array(range(1, 1+len(slam_speedmutli_score))), np.array([falsepositives for (avg_dist, truepositives, falsenegatives, falsepositives) in slam_speedmutli_score]), color="green")
    #axe.scatter(np.array(range(1, 1+len(slam_speedmutli_score))), np.array([falsepositives for (avg_dist, truepositives, falsenegatives, falsepositives) in slam_speedmutli_score]), marker='x', label="cluster_falsepositives", color="green")

    noFront_speedmutli_score = g2o_slam_speedmulti_scores["noFrontend"]
    axe_tp.plot(np.array(range(1, 1+len(noFront_speedmutli_score))), np.array([truepositives for (path_dist, truepositives, falsenegatives, falsepositives) in noFront_speedmutli_score]), color="green")
    axe_tp.scatter(np.array(range(1, 1+len(noFront_speedmutli_score))), np.array([truepositives for (path_dist, truepositives, falsenegatives, falsepositives) in noFront_speedmutli_score]), marker='o', label="noFrontend_truepositives", color="green")
    axe_fp.plot(np.array(range(1, 1+len(noFront_speedmutli_score))), np.array([falsepositives for (path_dist, truepositives, falsenegatives, falsepositives) in noFront_speedmutli_score]), color="green")
    axe_fp.scatter(np.array(range(1, 1+len(noFront_speedmutli_score))), np.array([falsepositives for (path_dist, truepositives, falsenegatives, falsepositives) in noFront_speedmutli_score]), marker='x', label="noFrontend_falsepositives", color="green")
    axe_pd.plot(np.array(range(1, 1+len(noFront_speedmutli_score))), np.array([path_dist for (path_dist, truepositives, falsenegatives, falsepositives) in noFront_speedmutli_score]), color="green")
    axe_pd.scatter(np.array(range(1, 1+len(noFront_speedmutli_score))), np.array([path_dist for (path_dist, truepositives, falsenegatives, falsepositives) in noFront_speedmutli_score]), marker='^', label="noFrontend_pathdist", color="green")

    fullslam_speedmutli_score = g2o_slam_speedmulti_scores["full"]
    axe_tp.plot(np.array(range(1, 1+len(fullslam_speedmutli_score))), np.array([truepositives for (path_dist, truepositives, falsenegatives, falsepositives) in fullslam_speedmutli_score]), color="red")
    axe_tp.scatter(np.array(range(1, 1+len(fullslam_speedmutli_score))), np.array([truepositives for (path_dist, truepositives, falsenegatives, falsepositives) in fullslam_speedmutli_score]), marker='o', label="full_truepositives", color="red")
    axe_fp.plot(np.array(range(1, 1+len(fullslam_speedmutli_score))), np.array([falsepositives for (path_dist, truepositives, falsenegatives, falsepositives) in fullslam_speedmutli_score]), color="red")
    axe_fp.scatter(np.array(range(1, 1+len(fullslam_speedmutli_score))), np.array([falsepositives for (path_dist, truepositives, falsenegatives, falsepositives) in fullslam_speedmutli_score]), marker='x', label="full_falsepositives", color="red")
    axe_pd.plot(np.array(range(1, 1+len(fullslam_speedmutli_score))), np.array([path_dist for (path_dist, truepositives, falsenegatives, falsepositives) in fullslam_speedmutli_score]), color="red")
    axe_pd.scatter(np.array(range(1, 1+len(fullslam_speedmutli_score))), np.array([path_dist for (path_dist, truepositives, falsenegatives, falsepositives) in fullslam_speedmutli_score]), marker='^', label="full_pathdist", color="red")

    axe_tp.legend()
    axe_fp.legend()
    axe_pd.legend()
    axe_tp.grid()
    axe_fp.grid()
    axe_pd.grid()
    fig.show()
    fig.savefig(vis_out_path/"slam_cluster_acc_over_speed.png")


if __name__ == "__main__":
    # TODO
    #  check odometry accuracy
    #  better metric to evalute the maps
    #  better front-end (m-dist)
    #  speed and yawrate in carpose-vertex, constrains between them that speed and yawrate are constant. Then gnss.speed_over_ground is constraint on one carpose
    #  only use gnss constraint when new measurements are available, instead of every frame.
    main()

"""
negative information:
when applied at every step makes the result worse.

"""