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
        self.r_track = my_track.get("radius")       # track radius

        # initalize arcs/curves into a list
        self.n_pts = len(self.waypoints) # number of total waypoints
        self.arcs = []
        for point, i in zip(self.waypoints, range(len(self.waypoints))):
            self.arcs.append(BezierCurve(point, self.waypoints[(i+1)%self.n_pts])) # wrap last point to 1st point


        # observation space [x, y, heading, speed, distance_from_centerline]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        # action space [steer, throttle]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # constants
        self.ts = 0.1       # time step (s)
        self.max_spd = 10    # use ur eyes (m/s)
        self.max_accel = 2  # max throttle, in essence (m/s^2)
        self.max_turn = np.pi/3   # both ways (rad)
        self.max_dturn = np.pi  # max heading derivative (rad/s)

        # not constants, just initializing in case of fuckery
        self.cur_waypt = 0  # arc index
        self.steps = 0      # total step count
        self.cur_arc = self.arcs[self.cur_waypt]
        # the current arc is the one that the car will measure distance to
        # whether or not the car is within the distance indicated by the radius determines if the 
        # car is within track boundaries


    def _getObs(self):
        # returns self.pos. seems kinda redundant but whatever
        # can change if the obs space structure ever changes
        return self.pos
    

    def d2arc(self):
        # returns a distance form self.obs to the nearest point on self.cur_arc
        # function for distance squared is D(t) = (B(t) - P)^2 where B is the
        # bezier curve and P is the position of the car. The derivative is cubic
        # and thats already solved easy
        # B(t) = At^2 + Bt + C, D(t) = ( At^2 + Bt + C - P )^2
        A, B, C = self.cur_arc.get_bezier_coeff()
        dCP = C - self.pos[:2]

        dD_coeff = [
            4*np.dot(A,A),
            6*np.dot(A,B),
            2*(2*np.dot(A,dCP) + np.dot(B,B)),
            2*np.dot(B,dCP)
        ]
        roots = np.roots(dD_coeff)
        real_roots = roots[np.isreal(roots)].real

        # restrict search to roots on t in [0 1]
        restricted_real_roots = np.concatenate([real_roots[(real_roots>=0) & (real_roots<=1)], [0.0, 1.0]])
        # get nearest x,y points
        nearest_points = self.cur_arc.get_bezier(restricted_real_roots)
        # get distances to those points
        dist_to_arc = np.linalg.norm(nearest_points - self.pos[:2], axis=1)
        # solving D'(t) for zero may false positive from local minima, take absolute min distance.
        mindex = np.argmin(dist_to_arc) # min + index = mindex

        return dist_to_arc[mindex] # return distance to B(t*)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # initialize variables
        self.steps = 0 # reset total step count
        self.cur_waypt = 0 # index of the waypoint corresponding to the start of the current arc

        # the current arc is the one that the car measures distance to centerline from
        # the modulo thing is to wrap the last point back to the first point
        # since the current waypoint was just initialized to 0, this technically could be hard coded,
        # but just in case there's only 1 point (circular track), it's implemented w/ variables
        # that said the 2nd order bezier curve calculation probably can't handle the 1 waypoint case anyways
        self.cur_arc = self.arcs[self.cur_waypt]

        # for checkpoints
        self.cur_check = self.cur_arc.check_end_distance(self.pos[:3]) & self.cur_arc.check_end_line(self.pos[:3])
        self.last_check = self.cur_check

        # finish check
        self.is_race_complete = False
        self.is_out = False

        # car at start (initial position)
        # initial position is the first waypoint
        self.pos = np.array(np.concatenate(self.waypoints[self.cur_waypt], [0, 0]), dtype=np.float32)
        self.prev_pos = self.pos

        obs = self._getObs()
        info = {}
        return obs, info


    def step(self, action):
        # update state w/ action, compute reward
        # lets just say tentatively t_s = 0.01s
        # [ x += speed*cos(heading)*ts ]
        # state: [x, y, heading, speed, dist_2_curve]
        # action: [steering, throttle]
        self.pos = np.array([
            self.pos[0] + self.pos[3]*np.cos(self.pos[2])*self.ts,
            self.pos[1] + self.pos[3]*np.sin(self.pos[2])*self.ts,
            self.pos[2] + action[0]*self.ts,
            self.pos[3] + action[1]*self.ts,
            self.d2arc()
        ])
        obs = self._getObs()

        # check to see if car is onto next arc
        # this check = check distance and check pass
        self.cur_check = self.cur_arc.check_end_distance(self.pos[:3]) & self.cur_arc.check_end_line(self.pos[:3])
        # if previous check false and this check true, 
        if self.last_check == False & self.cur_check == True:
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
        if self.steps > 1000000 # arbitrary number
            reward = -1000
            truncated = True

        # throw stuff into info
        info = {}

        #terminated, truncated, info = 
        return obs, reward, terminated, truncated, info
    

    def render(self):
        # plot that shit
        pass