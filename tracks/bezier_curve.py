import numpy as np

# DIY bezier curve class. Basically exists to call a function that 
# acts as a mathematical function for the bezier curve equation
# Quadratic only (for now)
class BezierCurve():

    def __init__(self, p0, p1, r):

        self.p0 = np.asarray(p0[:2], dtype=float)
        self.p1 = np.asarray(p1[:2], dtype=float)
        self.r = r
        self.p1_head = p1[2]    # for checkpoints
        # rotational transformation matrix for rotating a vector to
        # be checked by square.
        self.TR = np.array([[np.cos(self.p1_head), np.sin(self.p1_head)],
                            [-np.sin(self.p1_head), np.cos(self.p1_head)]])

        # arbitrarily define two headings as "too close"
        # straight line will be drawn instead of curve
        self.is_too_parallel = False
        d_th = abs(p0[2] - p1[2]) % np.pi
        if min(d_th, np.pi-d_th) < np.pi/18:
            self.is_too_parallel = True
            return
        # ***Note***: p2 never gets set if this conditional is true

        # parametrize slope from theta
        d0 = np.array([np.cos(p0[2]), np.sin(p0[2])])
        d1 = np.array([np.cos(p1[2]), np.sin(p1[2])])
        
        # Create 2 parametrized lines, solve for intersection point = P2
        # p0   + t*d0         = p1   + s*d1
        # [x0] + t*[cos(th0)] = [x1] + s*[cos(th1)]
        # [y0]     [sin(th0)]   [y1]     [sin(th1)]
        A = np.column_stack((d0,-d1))
        b = self.p1 - self.p0
        t, s = np.linalg.solve(A,b)
        self.p2 = self.p0 + t*d0
        # 3 (x,y) points

        if not (t > 0 and s < 0):
            raise ValueError("yeah uhh you need a 3rd order spline for this")


    # for t in [0 1]
    def get_bezier(self, t):
        t = np.asarray(t, dtype=float)
        if np.any((t<0) | (t>1)):
            raise ValueError("Bezier curve out of bounds (0 <= t <= 1)")
        
        t = t.reshape(-1, 1)

        if self.is_too_parallel:
            return np.array(self.p0 + t*(self.p1 - self.p0))
        
        return (1-t)*((1-t)*self.p0 + t*self.p1) + t*((1-t)*self.p1 + t*self.p2)

    
    def get_bezier_coeff(self):
        return self.p0 - 2*self.p1 + self.p2, 2*(self.p1-self.p0), self.p0


    def check_end_distance(self, pos):
        # Calculate whether the car is close enough to the checkpoint line
        # "close enough" is arbitrary lol
        # Make a square of length r around the center of the checkpoint
        # Aligns square with end waypoint heading.
        # Returns false for checkpoint checks if car is too far
        if np.max(np.abs(np.matmul(self.TR, pos - self.p1))) > self.r:
            return False
        return True


    # checks which side of the end line the current position is on
    # for incrementing curves. returns the dot product, check happens in racetrack_env
    def check_end_line(self, pos):
        # Calculate whether the car is past the line or not.
        # given the way the dot product works, you can take a normal vector from
        # the heading of the end waypoint, and take the dot with the vector of
        # end waypoint to car position. The sign will be positive if the car is
        # past the line.
        p1_norm = self.TR[0]    # reusing the rotational matrix
        return np.dot(pos - self.p1, p1_norm)