from __future__ import division
import tornado.ioloop
import tornado.web
import os
import numpy as np
import redis

# import matplotlib.pyplot as plt

def raw_acc_to_acc(raw_acc):
    acc = np.frombuffer(raw_acc, np.uint16)
    return acc.reshape((len(acc)//3, 3))

def acc_to_pos(acc, t):
    # t is time interval in what units ???
    # acc is a list of 3-vectors in distance / unit time,
    # each vector is an acceleration sample after t seconds

    # compute velocity using Riemann sums:
    pos = []
    vel_so_far = [0.0, 0.0, 0.0]
    pos_so_far = (0.0, 0.0, 0.0)
    for a in acc:
        vel_so_far[0] += a[0]
        vel_so_far[1] += a[1]
        vel_so_far[2] += a[2]
        
        pos_so_far = tuple(pos_so_far[i] + vel_so_far[i] + a[i]/2for i in range(3))
        pos.append(pos_so_far)

    return pos

def f(coeffs, inp):
    return np.dot(coeffs, inp)

def find_plane(pos):
    # pos is "x, y, z" coordinates stored in rows:
    # [[px1, py1, b1],
    #  [px2, py2, b2]
    #  ...            
    #  [pxn, pyn, bn]]

    #        a          x   =   b
    # [[px1, py1, 1],  [xx     [b1 
    #  [px2, py2, 1]    xy  =   b2
    #  ...             ...     ...
    #  [pxn, pyn, 1]]   x0]     bn]

    pos = np.array(pos)
    a = np.vstack([pos[:, :2].T, np.ones(len(pos))]).T
    b = pos[:, 2:]
    x, res, rank, s = np.linalg.lstsq(a, b)
    return x

def get_normal(x):
    # should be the coefficients returned by least squares, see find_plane
    normal = np.copy(x)
    normal *= -1
    normal[-1] = 1
    normal = normal.T[0]
    return normal

def orthonormal_basis(x, anchor):
    bv1 = np.array((1.0, 0.0, f(x.T, np.array((1.0, 0.0, 1.0))))) - anchor
    bv1 = bv1 / np.linalg.norm(bv1, 2)
    # print(np.dot(bv1, normal))
    bv2 = np.array((0.0, 1.0, f(x.T, np.array((0.0, 1.0, 1.0))))) - anchor
    bv2 = bv2 - (np.dot(bv2, bv1) * bv1)
    bv2 = bv2 / np.linalg.norm(bv2, 2)
    # print(np.dot(bv2, normal))
    return bv1, bv2

def project_pos(pos):
    pos = np.array(pos)
    x = find_plane(pos)
    normal = get_normal(x)
    # anchor is the point in our plane that the normal vector passes through
    anchor = (x[2] / (np.linalg.norm(normal, 2)**2)) * normal
    bv1, bv2 = orthonormal_basis(x, anchor)
    new_points = []
    for point in pos:
        point -= anchor
        bv1_c = np.inner(point, bv1)
        bv2_c = np.inner(point, bv2)
        # plt.scatter(bv1_c, bv2_c)
        new_points.append([bv1_c, bv2_c])
    # plt.show()
    return new_points

class BlahCreateHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('<html><body><form action="/blah_new" method="POST">'
                   '<input type="text" name="key">'
                   '<input type="text" name="data">'
                   '<input type="submit" value="Submit">'
                   '</form></body></html>')

    def post(self):
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        r = redis.from_url(redis_url)
        k, v = self.get_body_argument("key"), self.get_body_argument("data")
        r.set(k, v)
        self.write("I just stored " + v + ". Go to /blah_all to see all of them.")

class BlahViewHandler(tornado.web.RequestHandler):
    def get(self):
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        r = redis.from_url(redis_url)
        for k in r.keys():
            self.write(k + " " + r.get(k) + "\n")

class MainHandler(tornado.web.RequestHandler):
#     def get(self):
#         self.write('<html><body><form action="/accelerations" method="POST">'
#                    '<input type="text" name="accelerations">'
#                    '<input type="submit" value="Submit">'
#                    '</form></body></html>')

    def post(self):
        self.set_header("Content-Type", "text/plain")
        raw_acc = self.request.body
        acc = raw_acc_to_acc(raw_acc)
        pos1 = acc_to_pos(acc, 1) # time!!?!?!?!?
        pos2 = project_pos(pos1)
        self.write("Before adjusting: " + str(pos1) + "\nAfter adjusting: " + str(pos2))
        # visualize positions

class ViewHandler(tornado.web.RequestHandler):
    def get(self):
        items = ["item1", "item2???", "item3"]

def make_app():
    return tornado.web.Application([
        (r"/accelerations", MainHandler),
        (r"/blah_new", BlahCreateHandler),
        (r"/blah_all", BlahViewHandler),
        (r"/view", ViewHandler)
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(int(os.environ.get('PORT', 8888)))
    tornado.ioloop.IOLoop.current().start()
