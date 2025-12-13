import numpy as np
import matplotlib.pyplot as plt

def gen(xx1, xx2, uu1, uu2, tf, dt, ns, ni):

    def compute_5th_order_coeffs(xx1, xx2, tf):
        A = np.array([
        [1, 0, 0, 0, 0, 0],             # x(0) = x1
        [1, tf, tf**2, tf**3, tf**4, tf**5], # x(T) = x2
        [0, 1, 0, 0, 0, 0],             # dx/dt(0) = 0
        [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4], # dx/dt(T) = 0
        [0, 0, 2, 0, 0, 0],             # d2x/dt2(0) = 0
        [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3] # d2x/dt2(T) = 0
    ])
    
        b = np.array([xx1, xx2, 0, 0, 0, 0])
        coeffs = np.linalg.solve(A, b)
        return coeffs

    TT = int(tf/dt)
    tt = np.linspace(0, tf, TT)

    xx_ref = np.zeros((ns, TT))
    uu_ref= np.zeros((ni, TT))

    for i in range(ns):
        coeffs = compute_5th_order_coeffs(xx1[i], xx2[i], tf)
        x_ref = coeffs[0] + coeffs[1]*tt + coeffs[2]*tt**2 + coeffs[3]*tt**3 + coeffs[4]*tt**4 + coeffs[5]*tt**5
        xx_ref[i,:] = x_ref

    for i in range(ni):
        coeffs = compute_5th_order_coeffs(uu1[i], uu2[i], tf)
        u_ref = coeffs[0] + coeffs[1]*tt + coeffs[2]*tt**2 + coeffs[3]*tt**3 + coeffs[4]*tt**4 + coeffs[5]*tt**5
        uu_ref[i,:] = u_ref

    return xx_ref, uu_ref



