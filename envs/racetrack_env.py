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
        with open("tracks/track_list.yaml") as fpath:
            data = yaml.safe_load(fpath)
        my_track = data.get(track_name) # this might cause type issues. maybe cast?
        self.waypoints = my_track.get("waypoints") # all track waypoints
        self.r_track = my_track.get("radius") # track radius


        # observation space [x, y, speed, heading, distance_from_centerline]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        # action space [throttle, steer]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # some constants
        self.ts = 0.1 # time step
        self.max_spd = 1
        self.max_accel = 1 # max throttle, in essence
        self.max_turn = 1 # both ways
        self.max_dturn = 1 # max heading derivative

    def _getObs(self):
        # returns self.pos. seems kinda redundant but whatever
        return self.pos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # car at start (initial position)
        self.pos = np.array([100, 500, 90, 0, 0], dtype=np.float32)
        obs = self._getObs()
        info = {}
        return obs, info

    def step(self, action):
        # update state w/ action, compute reward
        # lets just say tentatively t_s = 0.01s
        # [ x += speed*cos(heading)*ts ]
        self.pos = np.array([
            self.pos[0] + self.pos[2]*np.cos(self.pos[2])*self.ts
            self.pos[1] + self.pos[2]*np.sin(self.pos[2])*self.ts
            self.pos[2] + action[0]*self.ts
            self.pos[2] + action[1]*self.ts
            
        ])

        # obs = 
        # reward: negative constant at every step, incentivise finishing faster
        reward = -self.ts
        # check for termination conditions
        #terminated, truncated, info = 
        return obs, reward, terminated, truncated, info
    
    def render(self):
        # plot that shit
        pass