import gymnasium as gym
import numpy as np
import yaml

class RacetrackEnv(gym.Env):

    def __init__(self, track_file_path):
        super().__init__()

        # load track data from file
        # a track is defined as a set of ordered vectors that make the centerline and a track
        # radius that gives it width. out of bounds is calculated via distance to centerline
        with open(track_file_path) as fpath:
            data = yaml.safe_load(fpath)
        self.waypoints = data["waypoints"] # all track waypoints
        self.r_track = data["track_radius"] # track radius


        # observation space [x, y, heading, speed, distance_from_centerline]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        # action space [steer, throttle]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # some constants
        self.T_S = 0.1 # time step

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
        
        # obs: can kind of just arbitrarily decide on the time step each step
        # that determines the position change as a result of speed and heading
        # lets just say tentatively t_s = 0.1s
        self.pos = np.array([
            self.pos[0] + self.pos[3]*np.cos()
        ])

        # obs = 
        # reward: negative constant at every step, incentivise finishing faster
        reward = -self.T_S
        # check for termination conditions
        #terminated, truncated, info = 
        return obs, reward, terminated, truncated, info
    
    def render(self):
        # plot that shit
        pass