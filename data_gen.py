import os
import random
import numpy as np
import sys
import multiprocessing

def lorenz(pt, s=10, r=28, b=2.667):
    x, y, z = pt[0], pt[1], pt[2]
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

#three dimensional code, but only 2d system so z = 0
def vanderpol(pt, a=1.5, mu=1):
    x, y, z = pt[0], pt[1], pt[2]
    x_dot = y
    y_dot = -x + mu*y*(a-x*x)
    z_dot = 0
    return x_dot, y_dot, z_dot

def dist(x, y):
    s = 0
    for i in range(len(x)):
        s += (x[i]-y[i])**2
    return s**0.5

def varAt(pts, **kwargs):
    """
    computes variance with number of points at every time point
    RETURNS three numbers: varx, vary, varz
    """
    kwargs['totalStep'] = 2
    kwargs['realStep'] = kwargs['lenVar']
    kwargs['multires'] = False
    var = []
    for i in range(kwargs['npts']):
        pn = []
        for iter in pts:
            pn.append(iter + random.random()*kwargs['pbC']-kwargs['pbC']/2.0)
        last, varArr = gen_series(pn, logData = False, **kwargs)
        var.append(last[-1])
    return np.array(np.var(var, axis=0))

def gen_series(init, logData = False, retVar = False, **kwargs):
    """
    kwargs contains
    'dstT', distance between micro points (base)
    'totalStep', total number of actual points
    'realStep', number of micropoints between actual points 
    'pbC', range for pertubations of variance
    'npts', number of variance points to simulate
    'lenVar', length of variance track to simulate
    'multires', multires or not
    'series', series to produce

    RETURNS: list of lists (points and xyz variances if specified)
    """           
    pts = [init]
    var = [[0]*len(init)]

    oldpt = init
    minres = kwargs['dstT']
    totalStep = kwargs['totalStep']
    realStep = kwargs['realStep']
    for i in range(totalStep - 1):
        tdist = 0
        for j in range(realStep):
            if(kwargs['multires']):
                v = varAt(oldpt, **kwargs)**2
                res = max((minres/np.linalg.norm(v))/800, 0.01)#1200000
            else:
                res = minres
            velo = kwargs['series'](oldpt)
            newpt = oldpt + np.dot(res, np.dot(1/np.linalg.norm(velo), velo))
            oldpt = newpt
        pts.append(list(oldpt))
        if(retVar):
            var.append(varAt(oldpt, **kwargs))
    return pts, var

def rrange(a, b):
    return (random.random()*(b-a)+a)

def gen_many(n, res, i):
    for iter in range(n):
        keywords = {'dstT':0.25, 'totalStep':int(sys.argv[1]), 'realStep':40, 'pbC':0.5, 'npts':15, 'multires':bool(int(sys.argv[4])), 'series':lorenz, 'lenVar':25}
        pt, varArr = gen_series([rrange(-8,8), rrange(-8,8), rrange(-8,8)], logData=True, retVar = True, **keywords)
        res.append([pt, varArr])
    np.save("data/" + str(sys.argv[5]) + str(i) + ".npy", np.array(res))

if __name__ == "__main__":
    print("running as main - real datagen")
    jobRes = []
    jobs = []
    n_process = int(sys.argv[2])
    for i in range(n_process):
        p = multiprocessing.Process(target=gen_many, args=(int(int(sys.argv[3])/n_process)+1, jobRes, i))
        jobs.append(p)
        p.start()
        
    for p in jobs:
        p.join()
    
    t = []
    for i in range(n_process):
        a = np.load("data/" + str(sys.argv[5]) + str(i) + ".npy")
        a = list(a)
        t.extend(a)

    adj = []

    mxx = np.min(t)
    mnn = np.max(t)
    print("max", mnn, "min", mxx)
    h = []
    for i in range(len(t)):
        mn = np.min(t[i][0])
        t[i][0] = np.add(-1*np.min(t[i][0]), t[i][0])
        mx = np.max(t[i][0])
        t[i][0] = np.dot(1/np.max(t[i][0]), t[i][0])
        t[i][1] = np.add(-1*np.min(t[i][1]), t[i][1])
        t[i][1] = np.dot(1/np.max(t[i][1]), t[i][1])
        #merge resolution and points
        tmptot = []
        for j in range(len(t[i][0])):
            tmptot.append(np.append(t[i][0][j], t[i][1][j]))
        print(tmptot)
        h.append(tmptot)
        adj.append((mn, mx))
    
    np.save("data/"+str(sys.argv[5])+".npy", np.array(h))
    np.save("data/"+str(sys.argv[5])+"ADJ.npy", np.array(adj))

    for i in range(n_process):
        os.system("rm data/"+str(sys.argv[5])+str(i)+".npy")
    

