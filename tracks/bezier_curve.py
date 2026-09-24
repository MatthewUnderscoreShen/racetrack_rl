import numpy as np

# DIY bezier curve class. Basically exists to call a function that 
# acts as a mathematical function for the bezier curve equation
# Quadratic only (for now)
class BezierCurve():

    def __init__(self, p0, p1, r):
        # note: angles passed in as radians

        self.p0 = np.asarray(p0[:2], dtype=float)
        self.p1 = np.asarray(p1[:2], dtype=float)
        self.p2 = None
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
    # returns (x, 2)
    def get_bezier(self, t):
        t = np.asarray(t, dtype=float)
        if np.any((t<0) | (t>1)):
            raise ValueError("Bezier curve out of bounds (0 <= t <= 1)")
        
        t = t.reshape(-1, 1)

        if self.is_too_parallel:
            return np.array(self.p0 + t*(self.p1 - self.p0))
        
        # READ THIS NO MATTER WHAT: "p2" is actually point p1 on wikipedia
        return (1-t)*((1-t)*self.p0 + t*self.p2) + t*((1-t)*self.p2 + t*self.p1)

    
    def get_bezier_coeff(self):
        # derivative wrt t
        # p0 = start, p1 = end, p2 = middle
        return self.p0 - 2*self.p2 + self.p1, 2*(self.p2-self.p0), self.p0

    def get_p2(self):
        return self.p2


    # for any x,y point, returns shortest distance to the arc
    # function for distance squared is D(t) = (B(t) - P)^2 where B is the
    # bezier curve and P is the position of the car. The derivative is cubic
    # and thats already solved easy
    # B(t) = At^2 + Bt + C, D(t) = ( At^2 + Bt + C - P )^2
    def d2arc(self, pos):   # pos = (2,) -> (x,y)
        pos = np.array(pos)         # type check
        A, B, C = self.get_bezier_coeff()
        dCP = C - pos

        dD_coeff = [
            4*np.dot(A,A),
            6*np.dot(A,B),
            2*(2*np.dot(A,dCP) + np.dot(B,B)),
            2*np.dot(B,dCP)
        ]
        roots = np.roots(dD_coeff)
        real_roots = roots[np.isreal(roots)].real

        # restrict search to roots on t in [0 1]
        restricted_real_roots = np.concatenate((real_roots[(real_roots>=0) & (real_roots<=1)], [0.0, 1.0]))
        # get nearest x,y points
        nearest_points = self.get_bezier(restricted_real_roots)
        # get distances to those points
        dist_to_arc = np.linalg.norm(nearest_points - pos, axis=1)
        # solving D'(t) for zero may false positive from local minima, take absolute min distance.
        mindex = np.argmin(dist_to_arc) # min + index = mindex

        return dist_to_arc[mindex] # return distance to B(t*)


    def check_end_distance(self, pos):
        pos = np.array(pos)         # type check
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
        pos = np.array(pos)         # type check
        # Calculate whether the car is past the line or not.
        # given the way the dot product works, you can take a normal vector from
        # the heading of the end waypoint, and take the dot with the vector of
        # end waypoint to car position. The sign will be positive if the car is
        # past the line.
        p1_norm = self.TR[0]    # reusing the rotational matrix
        return np.dot(pos - self.p1, p1_norm)