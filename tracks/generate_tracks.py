import yaml

# data for all tracks is hard coded in this script. Data consists of a few
# waypoints (x, y, theta), and the rest of the points are interpolated using
# ((insert method here)). That data then gets written to a yaml file for each
# track, which is stored in this folder (/tracks). That data gets loaded by
# racetrack_env.py.

# cancel the interpolation we doing arcs

# cancel the arcs we doing quadratic bezier curves
# try to keep the angles of consecutive nodes above a certain delta
# so shenanigans dont happen

# track data (all of them)
# each track's key is it's filename
# [x, y, theta]
track_list = {
    "test_track": [
        [-200, -100, -45],
        [200, -100, 45],
        [200, 100, 135],
        [-200, 100, -135]
    ]
}

# interpolate points 
# for each track
for name, track in track_list:
    # for each node in a track
    for i in range(1, len(track[1])):
        # draw an arc from this node to the next
        # should be an arc of an ellipse
        pass