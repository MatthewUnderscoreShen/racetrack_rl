import numpy as np

# DIY bezier curve class. Basically exists to call a function that 
# acts as a mathematical function for the bezier curve equation
# Quadratic only (for now)
class BezierCurve():

    def __init__(self, p0, p1):

        self.p0 = np.asarray(p0[:2], dtype=float)
        self.p1 = np.asarray(p1[:2], dtype=float)

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


    # checks which side of the end line the current position is on
    # for incrementing curves. returns a boolean (past or not past)
    def check_end_line(self, pos):
        # the checkpoint line that separates curves is perpendicular
        # to the heading of the end waypoint. thus, determining whether the car
        # is past the checkpoint can just be done by checking if the angle of 
        # end waypoint to current position is +/- 90deg from the end waypoint
        # heading. some limit is going to need to be set on distance to ensure
        # that a false positive isnt triggered by the car's position being very 
        # far away.
        