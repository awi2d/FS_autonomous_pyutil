#!usr/bin/env python

# Partikel Filter allg.:
# 	file:///tmp/mozilla_linuxflo0/sensors-21-00438-v2.pdf
# 	https://www.cc.gatech.edu/~dellaert/07F-Robotics/Schedule_files/11-ParticleFilters.ppt.pdf
# 	https://www.ri.cmu.edu/pub_files/2011/9/iros2011.pdf
# 	https://ae640a.github.io/assets/winter17/references/AMRobots5.pdf
#
# Partikelfilter Code:
# 	https://github.com/nwang57/FastSLAM/blob/f26f48be31960cdcecd33057208a77cfedf639a3/particle.py#L168
# #
# maximum likelihood:#
# 	https://arxiv.org/pdf/1303.6170.pdf
# #	http://andrewjkramer.net/intro-to-the-ekf-step-2/
# #	https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=144581
#
# #SLAM:
# #	https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6548759


# source /home/linuxflo/tf/devel/setup.bash

import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time

from numpy import genfromtxt
from sklearn.neighbors import NearestNeighbors

import rospy
from rospy import Timer as _timer
import message_filters
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import PoseStamped

from tf import TransformBroadcaster
import tf

sys.path.insert(1, os.path.join(sys.path[0], "/home/nano/FS_WS_V01/src/utils/ros_m/"))
from ros_m import _publish_point_cloud, _create_point_xyz, _publish_PoseStamped, _create_quaternion



# Estimation parameter of PF
Q = np.diag([0.05]) ** 2  # range error  [distance, bearing]
R = np.diag([0.05, 0.05, np.deg2rad(7)]) ** 2  # input error    [x, y, psi]
Q_SQRT = math.sqrt(Q[0,0])

# Particle filter parameter
num_particles = 20  # Number of Particle
NTh = num_particles / 1.5  # Number of particle for re-sampling

#map_file = "/home/nano/FS_WS_V01/src/path_extraction/src/maps/test_track_map.csv"
map_file = "/home/nano/FS_WS_V01/src/localization/maps/skidpad_school.csv"
X0 = np.array([[0], [0], [0]])

DEBUGGING = False
RATE = 1000 # [Hz]

class Map():
    def __init__(self, file_path):
        self.lm = self.load_csv_data(file_path)     # load landmarks
        self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.lm)

    def load_csv_data(self, path):
        # the csv file is sorted as [x,y]
        my_data = genfromtxt(path, delimiter=',', usecols=(0, 1))
        #with open(path, 'r') as f: 
        #    my_data = np.genfromtxt(f, dtype=None, delimiter=',', usecols=(0, 1))
        my_data[0,0] = 0
        print("Loaded map:", my_data)
        return my_data

class Particles():
    def __init__(self, x0, num_of_particles):
        self.state = np.repeat(x0, num_of_particles, axis = 1) # states [x, y, theta]  # state of particle     
        self.w = np.zeros((1, num_particles)) + 1.0 / num_particles  # weights

class ParticleFilter():

    def motion_model(self, x, u, dt):
        A = np.array([[1.0, 0, 0],
                      [0, 1.0, 0],
                      [0, 0, 1.0]])

        B = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])

        #x = A.dot(x) + B.dot(u)
        dx = transform_delta_x(x, u)
        x = A.dot(x) + dx

        return x


    def gauss_likelihood(self, x, mu, sigma):
        possibility = 1.0 / (math.sqrt(2.0 * math.pi) * sigma) * math.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        return possibility


    def calc_covariance(self, x_est, p_state, p_weight):
        """
        calculate covariance matrix
        """
        cov = np.zeros((3, 3))
        n_particle = p_state.shape[1]
        for i in range(n_particle):
            dx = (p_state[:, i:i + 1] - x_est)[0:3]
            cov += np.matmul(p_weight[0, i] * dx, dx.T)	# previous: cov += p_weight[0, i] * dx @ dx.T
        cov *= 1.0 / (1.0 - np.matmul(p_weight, p_weight.T)) # previuous: cov *= 1.0 / (1.0 - p_weight @ p_weight.T)

        return cov


    def pf_localization(self, p_state, p_weight, z, u, map, dt):
        """
        Localization with Particle filter
        """

        p_state = p_state.astype(float)
        
        for ip in range(num_particles):
            x = np.array([p_state[:, ip]]).T
            w = p_weight[0, ip]

            #  Predict with random input sampling
            ud1 = u[0, 0] + (2*np.random.randn()-1) * R[0, 0] ** 0.5
            ud2 = u[1, 0] + (2*np.random.randn()-1) * R[1, 1] ** 0.5
            ud3 = u[2, 0] + (2*np.random.randn()-1) * R[2, 2] ** 0.5
            ud = np.array([[ud1, ud2, ud3]]).T
            x = self.motion_model(x, ud, dt)

            #calc dist and angle to every landmark
            z_ass = np.zeros_like(map.lm)
            dz_ass = map.lm - x[0:2].T   # [x, y]
            z_ass[:,0] = np.hypot(dz_ass[:,0], dz_ass[:,1])    # distance
            z_ass[:, 1] = np.arctan2(dz_ass[:,1], dz_ass[:,0]) # theta
            
            #  Calc Importance Weight
            R_ = calc_R(x[2, 0])

            t0 = time.time()
            for i in range(len(z)):
                obs_xy = x[0:2, 0] + np.matmul(R_, z[i, 1:])
                print("obs_xy0: ", obs_xy)
                dz, indice = map.nbrs.kneighbors(np.array([obs_xy]))
                 
                w = w * self.gauss_likelihood(dz, 0, Q_SQRT)
                if w == 0:
                    print("!! w == 0 !!")
                    w = 0.0000000001
            print("test0: ", time.time()-t0)

            t0 = time.time()
            print("R: ", R_)
            print("z: ",z)
            print("Matmul:", np.matmul(R_, z[:, 1:].T))
            obs_xy = x[0:2].T + np.matmul(R_, z[:, 1:].T).T
            print("obs_xy1: ", obs_xy)
            dz, indice = map.nbrs.kneighbors(np.array(obs_xy))
            for i in range(len(z)):
                 
                w = w * self.gauss_likelihood(dz[i], 0, Q_SQRT)
                if w == 0:
                    print("!! w == 0 !!")
                    w = 0.0000000001
            print("test1: ", time.time()-t0)


            p_state[:, ip] = x[:, 0]
            p_weight[0, ip] = w
        p_weight = p_weight / p_weight.sum()  # normalize
        p_weight[np.isnan(p_weight)] = 0 

        x_est = p_state.dot(p_weight.T)
        p_est = self.calc_covariance(x_est, p_state, p_weight)

        #NTh = num_particles / 2.0  Number of particle for re-sampling

        N_eff = 1.0 / (p_weight.dot(p_weight.T))[0, 0]  # Effective particle number
        if N_eff < NTh:
            p_state, p_weight = self.re_sampling(p_state, p_weight)

        return x_est, p_est, p_state, p_weight


    def re_sampling(self, p_state, p_weight):
        """
        low variance re-sampling
        """

        w_cum = np.cumsum(p_weight)
        base = np.arange(0.0, 1.0, 1.0 / num_particles)
        re_sample_id = base + np.random.uniform(0, 1.0 / num_particles)
        indexes = []
        ind = 0
        for ip in range(num_particles):
            while re_sample_id[ip] > w_cum[ind]:
                ind += 1
            indexes.append(ind)

        p_state = p_state[:, indexes]
        p_weight = np.zeros((1, num_particles)) + 1.0 / num_particles  # init weight

        return p_state, p_weight

class Chatter:
    def __init__(self):
        # init ROS
        rospy.init_node('monte_carlo_localization', anonymous=False)
        rate = rospy.Rate(RATE)
        self.map_pcl_pub = rospy.Publisher('/FS/map_pcl', PointCloud, queue_size=100)
        self.est_pose_pub = rospy.Publisher('/FS/est_pose', PoseStamped, queue_size=10)   #[topic, msg-type, limit of queued msgs]

        # init Classes
        self.p = Particles(X0, num_particles)
        self.pf = ParticleFilter()
        self.map = Map(map_file)

        # create point cloud of known landmarks in map and publish real map
        self.map_pcl = []
        for i in range(len(self.map.lm)):
            point = _create_point_xyz([self.map.lm[i, 0], self.map.lm[i, 1], 0])
            self.map_pcl.append(point)

        _publish_point_cloud(self.map_pcl, "world", self.map_pcl_pub)


        # create TransformBroadcaster
        self.b = TransformBroadcaster()

        # init starting variables
        self.time0 = time.time()
        self.time_pcl0 = 0
        self.x_est0 = np.zeros((3,1))
        self.msg_odo_last_stamp = 0

        # cache for estimated pose via odometry node
        odo_popse_sub = message_filters.Subscriber('/FS/odo', PoseStamped)
        self.cache_odo_pose = message_filters.Cache(odo_popse_sub, 1000)

        # init ROS subscriber
        rospy.Subscriber('/FS/pcl_bbox2', PointCloud, self.callback, queue_size = 1)

    def callback(self, msg_pcl):
        # 1. get delta time and time of bbox_pcl
        # 2. return if bbox_pcl is already used
        # 3. take nearest (time based to time_pcl1) state of odometry
        # 4. return if delta time between msg_pcl and msg_odo is too big 
        # 5. calculate difference between last odo_msg (x_est0) and current (x_est1)
        # 6. convert current pcl_bbox in [dist, x, y]. (same cooridnate frame)
        # 7. start particle filter localization

        # 1. get delta time and time of bbox_pcl
        time_pcl1 = msg_pcl.header.stamp
        time1 = time.time()
        DT = time1 - self.time0
        

        # 2. return if bbox_pcl is already used
        if time_pcl1 == self.time_pcl0: return

        # 3. take nearest (time based to time_pcl1) state of odometry, return if None or odo_msg already used
        msg_odo = self.cache_odo_pose.getElemBeforeTime(time_pcl1)
        if msg_odo == None: print("msg_odo is None!"); return
        if msg_odo.header.stamp == self.msg_odo_last_stamp: print("odo_msg already used"); return
        self.msg_odo_last_stamp = msg_odo.header.stamp 

        x = msg_odo.pose.position.x
        y = msg_odo.pose.position.y
        quarternion = [msg_odo.pose.orientation.x, msg_odo.pose.orientation.y, msg_odo.pose.orientation.z, msg_odo.pose.orientation.w]
        euler_angles = tf.transformations.euler_from_quaternion(quarternion)
        x_est1 = np.array([[x], [y], [euler_angles[2]]])

        # 4. return if delta time between msg_pcl and msg_odo is too big 
        # dt_msgs = abs(time_pcl1 - msg_odo.header.stamp)
        # if dt_msgs > ?!: print("dt between msg_pcl and msg_odo is to big !!!!! dt=", dt_msgs); return

        # 5. calculate difference between last odo_msg (x_est0) and current (x_est1)
        #dx_est = x_est1 - self.x_est0
        dx_est = calc_delta_x(x_est1, self.x_est0)
        self.x_est0 = x_est1
        u = dx_est

        # 6. convert current pcl_bbox in [dist, x, y]. (same cooridnate frame)
        z = np.zeros((len(msg_pcl.points), 3))
        i=0
        for point in msg_pcl.points:
            zi = np.array([math.hypot(point.x, point.y), point.x, point.y])  # zi = [dist, x, y]
            # filter observation in the origin = [x=0, y=0]
            if zi[0] < 0.02: # [0.02 m]
                continue
            z[i] = zi
            i += 1



        # Filter to far cones
        z = z[z[:,0] <= 2]

        # 7. start particle filter localisation
        x_est, PEst, self.p.state, self.p.w = self.pf.pf_localization(self.p.state, self.p.w, z, u, self.map, DT)

        # ROS particle publishing
        for i in range(num_particles):
            translation = (self.p.state[0, i], self.p.state[1, i], 0)
            rotation = tf.transformations.quaternion_from_euler(0.0, 0.0, self.p.state[2, i])
            self.b.sendTransform(translation, rotation, rospy.Time.now(), str(i), '/world')

        # ROS x_est publishing
        translation = (x_est[0], x_est[1], 0)
        rotation = tf.transformations.quaternion_from_euler(0.0, 0.0, x_est[2])
        self.b.sendTransform(translation, rotation, rospy.Time.now(), 'estimation', '/world')
        _publish_PoseStamped(translation, rotation, "world", self.est_pose_pub)


        # publish real map
        _publish_point_cloud(self.map_pcl, "world", self.map_pcl_pub)


        print("Time passed for cb: ",time.time() - time1)
        print("DT (time betwwen last and current iteration: ", DT)
        self.time0 = time1
        self.time_pcl0 = time_pcl1




def calc_delta_x(x_now, x_last):
    dx = x_now - x_last
    R = calc_R_inv(x_last[2])
    dx[0:2] = np.matmul(R, dx[0:2])
    return dx

def transform_delta_x(x_particle, dx):
    R = calc_R(x_particle[2])
    dx[0:2] = np.matmul(R, dx[0:2])
    return dx

def calc_R(phi):
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    return np.array([[cos_phi, -sin_phi], [sin_phi, cos_phi]])

def calc_R_inv(phi):
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    return np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])



def main():
    print(__file__ + " start!!")

    node = Chatter()
    print("Node initialized")
    rospy.spin()


if __name__ == '__main__':
    main()
