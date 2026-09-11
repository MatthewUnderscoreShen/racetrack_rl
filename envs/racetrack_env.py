import gymnasium as gym
import numpy as np
import yaml

from tracks.bezier_curve import BezierCurve

class RacetrackEnv(gym.Env):


    def __init__(self, track_name):
        super().__init__()

        # load track data from file
        # a track is defined as a set of ordered vectors that make the centerline and a track
        # radius that gives it width. out of bounds is calculated via distance to centerline
        # possible to establish a variable radius, maybe later
        # yaml stores heading in deg, will convert to rad in bezier_curve
        with open("tracks/track_list.yaml") as fpath:
            data = yaml.safe_load(fpath)
        my_track = data.get(track_name)             # this might cause type issues. maybe cast?
        self.waypoints = my_track.get("waypoints")  # all track waypoints
        for point in self.waypoints:                # deg 2 rad all points first
            point[2] = np.deg2rad(point[2])
        self.r_track = my_track.get("radius")       # track radius

        # initalize arcs/curves into a list
        self.n_pts = len(self.waypoints)            # number of total waypoints
        self.arcs = []
        for point, i in zip(self.waypoints, range(len(self.waypoints))):
            self.arcs.append(BezierCurve(point, self.waypoints[(i+1)%self.n_pts])) # wrap last point to 1st point

        # constants for physical constraints loosely based on a honda accord
        self.ts = 0.1                           # time step (s)
        self.max_spd = 120 * (1609.34/3600)     # 120mph -> 53.6m/s
        self.max_a_fwd = (60/8.5)*(1609.34/3600) # max forward throttle 0->60mph in 8.5s -> 7.059mph/s -> 3.156m/s^2
        self.max_turn = np.pi/3                 # furthest wheel can turn both ways (rad)
        self.max_dturn = 2*np.pi                # max heading derivative (fastest spin) (rad/s)
        self.max_a_fric = 9.8                   # mu_f*g = max accel from friction (m/s^2) (tire limit)
        self.wb = 2.829                         # wheelbase (m) (bike model)

        # observation space [x, y, heading, v_x, v_y, steering_angle, distance_from_centerline]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        # action space [steer, throttle]
        self.action_space = gym.spaces.Box( low=np.array([-self.max_dturn, -self.max_a_fwd]),
                                            high=np.array([self.max_dturn, self.max_a_fwd]), 
                                            shape=(2,), dtype=np.float32)

        # not constants, just initializing in case of fuckery
        self.cur_waypt = 0  # arc index
        self.steps = 0      # total step count
        self.cur_arc = self.arcs[self.cur_waypt]
        # the current arc is the one that the car will measure distance to
        # whether or not the car is within the distance indicated by the radius determines if the 
        # car is within track boundaries


    def _getObs(self):
        # returns self.state. seems kinda redundant but whatever
        # can change if the obs space structure ever changes
        return self.state
    

    # given the obs and action at the start of the timestep, compute the new position
    # use self.state as obs. Do not set self.state in this function
    # obs: [x, y, heading(th), v_long, v_lat, steering_angle(phi), distance_from_centerline]
    # action: [steering, throttle]
    def update_pos(self, action):
        steering, throttle = action[0], action[1]

        # steering angle (relative to vehicle heading)
        phi = self.state[5] + steering*self.ts                # phi+ = phi + d_phi*dt
        phi = np.sign(phi)*min(np.abs(phi), self.max_turn)    # apply max constraint

        # basis change to front wheel
        A_f = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])     # rotation matrix forward
        v_long_f, v_lat_f = np.matmul(A_f, np.array([self.state[3], self.state[4]]))  # change of basis

        # longitudinal and lateral acceleration ***relative to front wheel
        a = np.array([throttle, v_long_f**2*np.sin(phi)/self.wb]) # a_lat = v_long^2/r
        if a[0] > 0:                                           # acceleration limits
            limit = (a[0]/self.max_a_fwd)**2 + (a[1]/self.max_a_fric)**2  # forward accel limit
        else:
            limit = (a[0]/self.max_a_fric)**2 + (a[1]/self.max_a_fric)**2  # backwards accel limit
        if limit > 1.0:                                         # constrain to limit
            a *= 1 / np.sqrt(limit)

        # new v_***_f based on constrained acceleration
        v_long_f_np1 = v_long_f + a[0]*self.ts      # forward euler
        v_lat_f_np1 = v_lat_f + a[1]*self.ts

        # v back to bike frame. invert A_f for rotation backwards
        v_long, v_lat = np.matmul(A_f.T, np.array([v_long_f_np1, v_lat_f_np1]))           # change basis back

        # use v_long_f_np1 to calculate angular velocity
        # omega = v / r = v / (wheelbase / sin(phi)) = v * sin(phi) / wb
        omega = v_long_f_np1 * np.sin(phi) / self.wb
        th = self.state[2] + omega * self.ts

        # basis change velocity relative to world coordinates
        A_w = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        v_x, v_y= np.matmul(A_w, np.array([v_long, v_lat]))

        # forward euler x and y
        x, y = self.state[0] + v_x*self.ts, self.state[1] + v_y*self.ts

        # call d2arc for nearest distance to current arc
        dist2c = self.cur_arc.d2arc((x,y))

        return (x, y, th, v_long, v_lat, phi, dist2c)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # initialize variables
        self.steps = 0 # reset total step count
        self.cur_waypt = 0 # index of the waypoint corresponding to the start of the current arc
        # car at start (initial position)
        # initial position is the first waypoint
        self.state = np.array(np.concatenate((self.waypoints[self.cur_waypt], [0, 0])), dtype=np.float32)
        self.prev_pos = self.state

        # the current arc is the one that the car measures distance to centerline from
        # the modulo thing is to wrap the last point back to the first point
        # since the current waypoint was just initialized to 0, this technically could be hard coded,
        # but just in case there's only 1 point (circular track), it's implemented w/ variables
        # that said the 2nd order bezier curve calculation probably can't handle the 1 waypoint case anyways
        self.cur_arc = self.arcs[self.cur_waypt]

        # for checkpoints
        self.cur_check = self.cur_arc.check_end_distance(self.state[:2]) & self.cur_arc.check_end_line(self.state[:2])
        self.last_check = self.cur_check

        # finish check
        self.is_race_complete = False
        self.is_out = False

        obs = self._getObs()
        info = {}
        return obs, info


    def step(self, action):
        # update state w/ action, compute reward
        # lets just say tentatively t_s = 0.01s
        # [ x += speed*cos(heading)*ts ]
        # state: [x, y, heading, speed, dist_2_curve]
        # action: [steering, throttle]

        # calculate first, then assign to self.state
        # calculate new heading (#2)
        th = self.state[2] + action[0]*self.ts
        th = np.sign(th)*min(np.abs(th), self.max_turn)

        # calculate acceleration w/ friction constraints
        # a = [heading accel, lateral accel = v^2/r]
        a = np.array([action[1], self.state[3]**2*np.sin(self.state[2])/self.wb])
        if a[0] > 0:    # acceleration vector
            limit = (a[0]/self.max_a_fwd)**2 + (a[1]/self.max_a_fric)**2  # forward accel limit
        else:
            limit = (a[0]/self.max_a_fric)**2 + (a[1]/self.max_a_fric)**2 # backward accel limit
        if limit > 1.0: # normalize to limit if above limit
            scale = 1 / np.sqrt(limit)
            a *= scale
        
        # the acceleration changes the velocity, so calculate velocity using the constrained acceleration
        v_long = self.state[3] + a[0]*self.t_s
        v_lat = 

        dth = self.state[2] + action[0]*self.ts
        dv = self.state[3] + action[1]*self.ts    # technically dv should be d_spd but whatever

        self.state = np.array([
            self.state[0] + self.state[3]*np.cos(self.state[2])*self.ts,
            self.state[1] + self.state[3]*np.sin(self.state[2])*self.ts,
            self.state[2] + action[0]*self.ts,
            self.state[3] + action[1]*self.ts,
            self.d2arc()
        ])
        # apply constraints, accel contraints built into action space
        self.state[2] = np.sign(self.state[2])*min(np.abs(self.state[2]), self.max_turn)
        self.state[3] = np.sign(self.state[3])*min(np.abs(self.state[3]), self.max_spd)
        obs = self._getObs()

        # check to see if car is onto next arc
        # this check = check distance and check pass
        self.cur_check = self.cur_arc.check_end_distance(self.state[:2]) & self.cur_arc.check_end_line(self.state[:2])
        # if previous check false and this check true, 
        if (not self.last_check) & (self.cur_check):
            # increment to next arc
            self.cur_waypt += 1
            # if that was the last arc, race is complete
            if self.cur_waypt >= self.n_pts:
                self.is_race_complete = True
            else: # otherwise go to the next arc
                self.cur_arc = self.arcs[self.cur_waypt]
        # update last check
        self.last_check = self.cur_check

        # reward: negative constant at every step, incentivise finishing faster
        reward = -self.ts

        # check for termination conditions
        # termination can occur either from out of bounds or finishing the race
        if self.is_out:
            reward = -1000
            terminated = True
        if self.is_race_complete:
            terminated = True

        # truncation should only happen if there are too many steps
        if self.steps > 1000000: # arbitrary number
            reward = -1000
            truncated = True

        # throw stuff into info
        info = {}

        #terminated, truncated, info = 
        return obs, reward, terminated, truncated, info
    

    def render(self):
        # plot that shit
        pass