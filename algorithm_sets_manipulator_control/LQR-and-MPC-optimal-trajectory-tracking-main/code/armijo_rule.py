import numpy as np
import matplotlib.pyplot as plt
import cost as cst
import fs_dynamics_staticps as fsd

def select_stepsize(stepsize_0, armijo_maxiters, cc, beta, xx_ref, uu_ref, xx_temp, uu, JJ, descent_arm, deltau, KK, sigma, A, B, plot = False):
    TT = uu.shape[1]
    ni = uu.shape[0]

    if A.ndim == 2:
        A = np.repeat(A[:, :, np.newaxis], TT, axis=2)
    if B.ndim == 2:
        B = np.repeat(B[:, :, np.newaxis], TT, axis=2)

    stepsizes = []
    costs_armijo = []
    stepsize = stepsize_0

    for ii in range(armijo_maxiters):
        xx_try = xx_temp.copy()
        uu_try = np.zeros((ni, TT))
        JJ_temp = 0

        for tt in range(TT-1):
            uu_try[:,tt] = uu[:,tt] + stepsize*deltau[:,tt]
            xx_try[:,tt+1] = fsd.dynamics(xx_try[:,tt], uu_try[:,tt])[0]

        for tt in range(TT-1):
            JJ_temp += cst.stagecost(xx_try[:,tt], uu_try[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
        JJ_temp += cst.termcost(xx_try[:,-1], xx_ref[:,-1])[0]

        stepsizes.append(stepsize)
        costs_armijo.append(min(JJ_temp, 100*JJ))

        print(f"  Step {ii}:")
        print(f"    stepsize: {stepsize:.3e}")
        print(f"    current cost: {JJ_temp:.3e}")
        print(f"    target value: {JJ + cc*stepsize*descent_arm:.3e}")

        if JJ_temp > JJ + cc*stepsize*descent_arm:
            stepsize = beta*stepsize
        else:
            print('Armijo stepsize = {:.3e}'.format(stepsize))
            break

        if ii == armijo_maxiters -1:
            print("WARNING: no stepsize was found with Armijo rule!")

    ############################
    # Descent Plot
    ############################
    if plot:
        steps = np.linspace(0, 1.0, 20)
        costs = np.zeros(len(steps))

        # Compute J_0 = J(u^k) to anchor the red/green dashed lines
        J_0 = 0
        for tt in range(TT - 1):
            J_0 += cst.stagecost(xx_temp[:, tt], uu[:, tt], xx_ref[:, tt], uu_ref[:, tt])[0]
        J_0 += cst.termcost(xx_temp[:, -1], xx_ref[:, -1])[0]

        for ii, step in enumerate(steps):
            xx_try = xx_temp.copy()
            uu_try = np.zeros((ni, TT))
            JJ_temp = 0

            for tt in range(TT-1):
                uu_try[:,tt] = uu[:,tt] + step*deltau[:,tt]
                xx_try[:,tt+1] = fsd.dynamics(xx_try[:,tt], uu_try[:,tt])[0]

            for tt in range(TT-1):
                JJ_temp += cst.stagecost(xx_try[:,tt], uu_try[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
            JJ_temp += cst.termcost(xx_try[:,-1], xx_ref[:,-1])[0]
            costs[ii] = JJ_temp

        plt.figure(1)
        plt.clf()
        plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - stepsize*d^k)$')
        plt.plot(steps, J_0 + descent_arm*steps, color='r',
                 label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^\\top d^k$')
        plt.plot(steps, J_0 + cc*descent_arm*steps, color='g', linestyle='dashed',
                 label='$J(\\mathbf{u}^k) - stepsize*c*\\nabla J(\\mathbf{u}^k)^\\top d^k$')
        plt.scatter(stepsizes, costs_armijo, marker='*')
        plt.grid()
        plt.xlabel('stepsize')
        plt.legend()
        plt.draw()
        plt.show()

    return stepsize
