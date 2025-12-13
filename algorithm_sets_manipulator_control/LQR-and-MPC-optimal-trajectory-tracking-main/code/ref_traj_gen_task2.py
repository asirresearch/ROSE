import numpy as np
import matplotlib.pyplot as plt

def gen(xx1, xx2, uu1, uu2, tf, dt, ns, ni, t_stop=0.5):
    def compute_5th_order_coeffs(x1, x2, tf_local):
        A = np.array([
            [1, 0, 0, 0, 0, 0],                 
            [1, tf_local, tf_local**2, tf_local**3, tf_local**4, tf_local**5], 
            [0, 1, 0, 0, 0, 0],                 
            [0, 1, 2*tf_local, 3*tf_local**2, 4*tf_local**3, 5*tf_local**4], 
            [0, 0, 2, 0, 0, 0],                 
            [0, 0, 2, 6*tf_local, 12*tf_local**2, 20*tf_local**3] 
        ])
        b = np.array([x1, x2, 0, 0, 0, 0])
        coeffs = np.linalg.solve(A, b)
        return coeffs

    # Time partitioning
    t_move = (tf - 2 * t_stop) / 2
    TT_move = int(t_move / dt)
    TT_stop = int(t_stop / dt)
    TT = 2 * (TT_move + TT_stop)
    tt_move = np.linspace(0, t_move, TT_move)

    xx_ref = np.zeros((ns, TT))
    uu_ref = np.zeros((ni, TT))

    for i in range(ns):
        # Forward move
        coeffs_fwd = compute_5th_order_coeffs(xx1[i], xx2[i], t_move)
        x_fwd = sum(coeffs_fwd[j] * tt_move**j for j in range(6))

        # Stop at xx2
        x_stop2 = np.ones(TT_stop) * xx2[i]

        # Backward move
        coeffs_bwd = compute_5th_order_coeffs(xx2[i], xx1[i], t_move)
        x_bwd = sum(coeffs_bwd[j] * tt_move**j for j in range(6))

        # Final stop at xx1
        x_stop1 = np.ones(TT_stop) * xx1[i]

        # Concatenate all segments
        xx_ref[i, :] = np.concatenate((x_fwd, x_stop2, x_bwd, x_stop1))

    for i in range(ni):
        coeffs_fwd = compute_5th_order_coeffs(uu1[i], uu2[i], t_move)
        u_fwd = sum(coeffs_fwd[j] * tt_move**j for j in range(6))

        u_stop2 = np.ones(TT_stop) * uu2[i]

        coeffs_bwd = compute_5th_order_coeffs(uu2[i], uu1[i], t_move)
        u_bwd = sum(coeffs_bwd[j] * tt_move**j for j in range(6))

        u_stop1 = np.ones(TT_stop) * uu1[i]

        uu_ref[i, :] = np.concatenate((u_fwd, u_stop2, u_bwd, u_stop1))

    return xx_ref, uu_ref





