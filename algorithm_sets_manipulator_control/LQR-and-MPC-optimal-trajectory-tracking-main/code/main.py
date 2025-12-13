import numpy as np
import scipy as sp
import control as ctrl
import matplotlib.pyplot as plt
import fs_dynamics_staticps as fsd
from find_equilibria import find_equilibrium_numerical
import ref_traj_gen_task2 as ref_gen2
import ref_traj_gen_task1 as ref_gen1
import cost as cst
import armijo_rule
from cost import QQt, RRt
from solver_LQR import ltv_LQR
from mpc_controller import solver_mpc
from animation import animate_surface

#######################
# Algorithm parameters
#######################

max_iters = 20
fixed_stepsize = 0.5
term_cond = 1e-6

# Flags
Task1 = False # for choosing the reference trajectory generator - True for Task 1 and False for Task 2 and on
plots_task1_2 = False # for plotting the intermediate trajectories in Task 1 and 2
Armijo = True
LQR = True
MPC = False
perturbed = False
Animation = False

# Armijo parameters
stepsize_0 = 1.0  
cc = 0.5 
beta = 0.7 
armijo_maxiters = 20 
visu_descent_plot = False

#######################
# Trajectory parameters
#######################

tf = 2 # final time in seconds
dt = fsd.dt   # get discretization step from dynamics
ns = fsd.ns
ni = fsd.ni
TT = int(tf/dt) # discrete-time sample

####################################
# Task 1 – Trajectory generation (I)
####################################

# Compute equilibrium points
xx_init1 = np.array([0.001, 0.0005, -0.001, 0.0005, 0, 0, 0, 0])
xx_init2 = np.array([-0.001, -0.0005, 0.001 , -0.0005, 0, 0, 0, 0])

uu0_1 = np.array([0.5, -0.5])
uu0_2 = np.array([-0.5, 0.5])

eq1 = find_equilibrium_numerical(xx_init1, uu0_1)
eq2 = find_equilibrium_numerical(xx_init2, uu0_2)

xx_eq1, uu_eq1 = np.array(eq1[0], dtype = float).flatten(), np.array(eq1[1], dtype = float).flatten()
xx_eq2, uu_eq2 = np.array(eq2[0], dtype = float).flatten(), np.array(eq2[1], dtype = float).flatten()

print("Equilibrium 1 states:", xx_eq1)
print("Equilibrium 2 states:", xx_eq2)

perturbation = 0.2*(xx_eq1)

if Task1:
    # Generate reference trajectory for Task 1
    xx_ref, uu_ref = ref_gen1.gen(xx_eq1, xx_eq2, uu_eq1, uu_eq2, tf, dt, ns, ni)
else:
    # Generate reference trajectory for Task 2
    xx_ref, uu_ref = ref_gen2.gen(xx_eq1, xx_eq2, uu_eq1, uu_eq2, tf, dt, ns, ni)

#####################################
# Task 2 – Trajectory generation (II)
#####################################

dfx_ref,dfu_ref = fsd.dynamics(xx_ref[:,TT-1], uu_ref[:,TT-1])[1:3]
AA_ref = dfx_ref.T
BB_ref = dfu_ref.T

QQT = ctrl.dare(AA_ref,BB_ref, QQt, RRt)[0]

# Quasi-static initial trajectory
xx_init = np.repeat(xx_ref[:,0].reshape(-1,1), TT, axis=1)
uu_init = np.repeat(uu_ref[:,0].reshape(-1,1), TT, axis=1)
xx_init = np.repeat(xx_ref[:, 0].reshape(-1, 1), TT, axis=1)  # Keep original initialization
x0 = xx_init[:,0]       # Not for Newton

# Arrays to store data
xx = np.zeros((ns, TT, max_iters))   # state seq.
uu = np.zeros((ni, TT, max_iters))   # input seq.

JJ = np.zeros(max_iters)      # collect cost
deltau = np.zeros((ni, TT, max_iters))
lmbd = np.zeros((ns, TT, max_iters)) # lambdas - costate seq.
dJ = np.zeros((ni,TT, max_iters))     # DJ - gradient of J wrt u
descent = np.zeros(max_iters) # collect descent direction
descent_arm = np.zeros(max_iters) # collect descent direction
deltau = np.zeros((ni, TT, max_iters))

A = np.zeros((ns, ns, TT))
B = np.zeros((ns, ni, TT))
qt = np.zeros((ns,TT))
rt = np.zeros((ni,TT))

KK = np.zeros((ni, ns, TT, max_iters))
sigma = np.zeros((ni, TT, max_iters))
PP = np.zeros((ns, ns, TT, max_iters))
pp = np.zeros((ns, TT, max_iters))

print('-*-*-*-*-*-')

# Initialize trajectories
xx[:,:,0] = xx_init
uu[:,:,0] = uu_init

# Before the main loop, initialize storage for stepsizes
chosen_stepsizes = []

for kk in range(max_iters - 1):

    JJ[kk] = 0

    # Calculate stage cost
    for tt in range(TT - 1):
        temp_cost = cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[0]
        JJ[kk] += temp_cost

    # Terminal cost
    temp_cost = cst.termcost(xx[:, -1, kk], xx_ref[:, -1], QQT)[0]
    JJ[kk] += temp_cost

    # Initialize terminal conditions
    qT = cst.termcost(xx[:, TT-1, kk], xx_ref[:, TT-1], QQT)[1]
    lmbd_temp = cst.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1], QQT)[1]
    lmbd[:,TT-1,kk] = lmbd_temp.copy().squeeze()

    # Backward pass
    for tt in range(TT-2, -1, -1):

        # Compute stage cost gradients
        qt[:,tt], rt[:,tt] = [arr.squeeze() for arr in cst.stagecost(xx[:, tt, kk], 
                                                                uu[:, tt, kk], 
                                                                xx_ref[:, tt], 
                                                                uu_ref[:, tt])[1:3]]
    
        xxp_ref, dfx, dfu = fsd.dynamics(xx_ref[:, tt], uu_ref[:, tt])
        A[:,:,tt] = dfx.T
        B[:,:,tt] = dfu.T
        
        lmbd_temp = A[:,:,tt].T@lmbd[:,tt+1,kk][:,None] + qt[:,tt][:,None]       # costate equation
        dJ_temp = B[:,:,tt].T@lmbd[:,tt+1,kk][:,None] + rt[:,tt][:,None]         # gradient of J wrt u                        
        
        lmbd[:,tt,kk] = lmbd_temp.squeeze()
        dJ[:,tt,kk] = dJ_temp.squeeze()

    KK[:,:,:,kk],sigma[:,:,kk] = ltv_LQR(A[:,:,tt],B[:,:,tt],QQt,RRt,QQT, TT, qt, rt, qT)[:2]

    # Forward pass
    xx_temp = xx[:, :, kk].copy()
    uu_temp = np.zeros((ni, TT))

    delta_x = np.zeros((ns,))
    
    # Computing descent direction and Armijo descent
    for tt in range(TT - 1):
        deltau[:, tt, kk] = KK[:, :, tt, kk]@delta_x + sigma[:, tt, kk]
        delta_x = A[:,:,tt]@delta_x + B[:,:,tt]@deltau[:,tt,kk]

        descent_arm[kk] += dJ[:,tt,kk].T@deltau[:,tt,kk]
        descent[kk] += deltau[:,tt,kk].T@deltau[:,tt,kk] 

    # Debug info
    print(f"Iteration {kk}: Total cost = {JJ[kk]}")
    print(f"Norm of sigma: {np.linalg.norm(sigma[:,:,kk]):.6e}")
    print(f"Iteration {kk}: Total descent armijo = {descent_arm[kk]}")
    print(f"Iteration {kk}: Total descent  = {descent[kk]}")

    if Armijo:
        # Stepsize selection - ARMIJO
        stepsize = armijo_rule.select_stepsize(stepsize_0, armijo_maxiters, cc, beta,
                                            xx_ref, uu_ref, xx_temp, 
                                            uu[:,:,kk], JJ[kk], descent_arm[kk],deltau[:, :, kk],
                                            KK[:,:,:,kk], sigma[:,:,kk], A[:,:,tt], B[:,:,tt], visu_descent_plot)
        chosen_stepsizes.append(stepsize)  # Store the chosen stepsize
    else:
        stepsize = fixed_stepsize

    # Forward integrate of the Newton's method
    for tt in range(TT - 1):
        dx = xx_temp[:, tt] - xx[:, tt, kk]
        du = KK[:, :, tt, kk] @ dx + stepsize*sigma[:, tt, kk]
        uu_temp[:, tt] = uu[:, tt, kk] + du
        xx_temp[:, tt + 1] = fsd.dynamics(xx_temp[:, tt], uu_temp[:, tt])[0]
    
    # Update trajectories for the next iteration
    xx[:, :, kk + 1] = xx_temp
    uu[:, :, kk + 1] = uu_temp

    # The termination condition
    if descent[kk]<= term_cond:
        max_iters = kk
        break

xx_star = xx[:,:,max_iters-1]
uu_star = uu[:,:,max_iters-1]
uu_star[:,-1] = uu_star[:,-2] # for plotting purposes

time = np.linspace(0, tf, TT)

if plots_task1_2 and not Armijo:
    # Intermediate trajectories for the fixed stepsize case
    xx_sec = xx[:,:,1]  # Store the second iteration for comparison
    uu_sec = uu[:,:,1]  
    uu_sec[:,-1] = uu_sec[:,-2] 

    xx_fif = xx[:,:,4]  # Store the fifth iteration for comparison
    uu_fif = uu[:,:,4]  
    uu_fif[:,-1] = uu_fif[:,-2]

    xx_eight = xx[:,:,7]  # Store the eightth iteration for comparison
    uu_eight = uu[:,:,7]  
    uu_eight[:,-1] = uu_eight[:,-2]

    print(xx_fif)

    # Iteration 2
    # Create figure for states
    plt.figure(figsize=(15, 10))
    for i in range(ns):
        ax = plt.subplot(4, 2, i+1)   
        ax.plot(time, xx_sec[i, :], label=f'Optimal State {i + 1}')
        ax.plot(time, xx_ref[i, :], '--', label=f'Reference State {i + 1}')
        ax.grid(True)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'State {i+1}')
        ax.legend()
    plt.tight_layout()
    plt.show()

    # Create figure for inputs
    plt.figure(figsize=(10, 4))  
    for i in range(2): 
        ax = plt.subplot(1, 2, i+1) 
        ax.plot(time, uu_sec[i, :], 'r-', label=f'Optimal Input {i + 1}')
        ax.plot(time, uu_ref[i, :], 'b--', label=f'Reference Input {i + 1}')
        ax.grid(True)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'Input {i+1}') 
        ax.legend()
    plt.tight_layout()
    plt.show

    # Iteration 5
    plt.figure(figsize=(15, 10))
    for i in range(ns):
        ax = plt.subplot(4, 2, i+1)   
        ax.plot(time, xx_fif[i, :], label=f'Optimal State {i + 1}')
        ax.plot(time, xx_ref[i, :], '--', label=f'Reference State {i + 1}')
        ax.grid(True)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'State {i+1}')
        ax.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))  
    for i in range(2): 
        ax = plt.subplot(1, 2, i+1) 
        ax.plot(time, uu_fif[i, :], 'r-', label=f'Optimal Input {i + 1}')
        ax.plot(time, uu_ref[i, :], 'b--', label=f'Reference Input {i + 1}')
        ax.grid(True)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'Input {i+1}') 
        ax.legend()
    plt.tight_layout()
    plt.show

    # Iteration 8
    plt.figure(figsize=(15, 10))
    for i in range(ns):
        ax = plt.subplot(4, 2, i+1)   
        ax.plot(time, xx_eight[i, :], label=f'Optimal State {i + 1}')
        ax.plot(time, xx_ref[i, :], '--', label=f'Reference State {i + 1}')
        ax.grid(True)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'State {i+1}')
        ax.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))  
    for i in range(2): 
        ax = plt.subplot(1, 2, i+1) 
        ax.plot(time, uu_eight[i, :], 'r-', label=f'Optimal Input {i + 1}')
        ax.plot(time, uu_ref[i, :], 'b--', label=f'Reference Input {i + 1}')
        ax.grid(True)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'Input {i+1}') 
        ax.legend()
    plt.tight_layout()
    plt.show

    # Last iteration 
    plt.figure(figsize=(15, 10))
    for i in range(ns):
        ax = plt.subplot(4, 2, i+1)   
        ax.plot(time, xx_star[i, :], label=f'Optimal State {i + 1}')
        ax.plot(time, xx_ref[i, :], '--', label=f'Reference State {i + 1}')
        ax.grid(True)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'State {i+1}')
        ax.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))  
    for i in range(2): 
        ax = plt.subplot(1, 2, i+1) 
        ax.plot(time, uu_star[i, :], 'r-', label=f'Optimal Input {i + 1}')
        ax.plot(time, uu_ref[i, :], 'b--', label=f'Reference Input {i + 1}')
        ax.grid(True)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'Input {i+1}') 
        ax.legend()
    plt.tight_layout()
    plt.show

elif plots_task1_2 and Armijo:
    # Intermediate trajectories for the Armijo case
    xx_sec = xx[:,:,1]  # Store the second iteration for comparison
    uu_sec = uu[:,:,1]
    uu_sec[:,-1] = uu_sec[:,-2]

    xx_third = xx[:,:,2] # Store the third iteration for comparison
    uu_third = uu[:,:,2]
    uu_third[:,-1] = uu_third[:,-2]

    # Iteration 2
    plt.figure(figsize=(15, 10))
    for i in range(ns):
        ax = plt.subplot(4, 2, i+1)   
        ax.plot(time, xx_sec[i, :], label=f'Optimal State {i + 1}')
        ax.plot(time, xx_ref[i, :], '--', label=f'Reference State {i + 1}')
        ax.grid(True)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'State {i+1}')
        ax.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))  
    for i in range(2): 
        ax = plt.subplot(1, 2, i+1) 
        ax.plot(time, uu_sec[i, :], 'r-', label=f'Optimal Input {i + 1}')
        ax.plot(time, uu_ref[i, :], 'b--', label=f'Reference Input {i + 1}')
        ax.grid(True)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'Input {i+1}') 
        ax.legend()
    plt.tight_layout()
    plt.show

    # Iteration 3
    plt.figure(figsize=(15, 10))
    for i in range(ns):
        ax = plt.subplot(4, 2, i+1)   
        ax.plot(time, xx_third[i, :], label=f'Optimal State {i + 1}')
        ax.plot(time, xx_ref[i, :], '--', label=f'Reference State {i + 1}')
        ax.grid(True)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'State {i+1}')
        ax.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))  
    for i in range(2): 
        ax = plt.subplot(1, 2, i+1) 
        ax.plot(time, uu_third[i, :], 'r-', label=f'Optimal Input {i + 1}')
        ax.plot(time, uu_ref[i, :], 'b--', label=f'Reference Input {i + 1}')
        ax.grid(True)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'Input {i+1}') 
        ax.legend()
    plt.tight_layout()
    plt.show

    # Last iteration 
    plt.figure(figsize=(15, 10))
    for i in range(ns):
        ax = plt.subplot(4, 2, i+1)   
        ax.plot(time, xx_star[i, :], label=f'Optimal State {i + 1}')
        ax.plot(time, xx_ref[i, :], '--', label=f'Reference State {i + 1}')
        ax.grid(True)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'State {i+1}')
        ax.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))  
    for i in range(2): 
        ax = plt.subplot(1, 2, i+1) 
        ax.plot(time, uu_star[i, :], 'r-', label=f'Optimal Input {i + 1}')
        ax.plot(time, uu_ref[i, :], 'b--', label=f'Reference Input {i + 1}')
        ax.grid(True)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'Input {i+1}') 
        ax.legend()
    plt.tight_layout()
    plt.show

    # Cost and descent
    plt.figure('Descent direction')
    plt.plot(np.arange(max_iters+1), descent[:max_iters+1])
    plt.xlabel('$k$')
    plt.ylabel('||$\\nabla J(\\mathbf{u}^k)||$')
    plt.yscale('log')
    plt.grid()
    plt.show()

    plt.figure('Cost')
    plt.plot(np.arange(max_iters+1), JJ[:max_iters+1])
    plt.xlabel('$k$')
    plt.ylabel('$J(\\mathbf{u}^k)$')
    plt.yscale('log')
    plt.grid()
    plt.show()

    """plt.figure(figsize=(10, 6))
    plt.plot(range(len(chosen_stepsizes)), chosen_stepsizes, 'bo-')
    plt.xlabel('Iteration')
    plt.ylabel('Chosen Stepsize')
    plt.title('Armijo Stepsizes vs. Iteration')
    plt.grid(True)
    #plt.yscale('log') 
    plt.show()"""

if Task1:
    exit()

#####################################
#Task 3 – Trajectory tracking via LQR
#####################################

if LQR:

    # Recompute gains using classic LQR (no affine terms)
    KK_LQR, _, _, _ = ltv_LQR(AA_ref, BB_ref, QQt, RRt, QQT, TT)

    xx_track = np.zeros_like(xx_star)
    uu_track = np.zeros_like(uu_star)

    if perturbed:
        xx_track[:, 0] = xx_star[:, 0] + perturbation
    else:
        xx_track[:, 0] = xx_star[:, 0] 
    
    for t in range(TT - 1):
        delta_x = xx_track[:, t] - xx_star[:, t]
        uu_track[:, t] = uu_star[:, t] + KK_LQR[:, :, t] @ delta_x
        xx_track[:, t + 1] = fsd.dynamics(xx_track[:, t], uu_track[:, t])[0]

######################################
# Task 4 – Trajectory tracking via MPC
######################################

if MPC:

    T_pred = 5                # Prediction time
    if perturbed:
        T_pred += 2

    # Input and state constraints
    umax = np.array([10.0, 10.0])
    umin = -umax
    xmax = np.array([0.1,0.1,0.1, 0.1,0.1,0.1, 0.1,0.1])
    xmin = -xmax

    # Initialize trajectories
    xx_real_mpc = np.zeros((ns, TT))
    uu_real_mpc = np.zeros((ni, TT))
    xx_mpc = np.zeros((ns, T_pred, TT))

    # Set initial state
    uu_real_mpc[:,0] = uu_eq1.squeeze()
    xx_real_mpc[:,0] = xx_eq1.squeeze()
    if perturbed:
        xx_real_mpc[:,0] += perturbation

    # Simulation loop
    for tt in range(TT-T_pred):
        print(f"t={tt}") 

        # Get optimal control input
        uu_real_mpc[:,tt], xx_mpc[:,:,tt] = solver_mpc(xx_real_mpc[:,tt], xx_star[:,tt:tt+T_pred], uu_star[:,tt:tt+T_pred], umax=umax, umin=umin, xmin=xmin, xmax=xmax, T_pred=T_pred)[:2]

        # Simulate system forward
        xx_real_mpc[:,tt+1] = fsd.dynamics(xx_real_mpc[:,tt], uu_real_mpc[:,tt])[0]

    xx_mpc = xx_real_mpc[:, :TT - T_pred]
    uu_mpc = uu_real_mpc[:, :TT - T_pred]

#########
# PLOTS #
#########

time = np.linspace(0, tf, TT)

if MPC:
    T_trim = len(time) - T_pred  # adjust for prediction horizon
    
    # state trajectories
    plt.figure(figsize=(15, 10))
    for i in range(ns):
        plt.subplot(4, 2, i + 1)
        plt.plot(time[:T_trim], xx_star[i, :T_trim], 'k--', label='Desired Optimal Trajectory')
        plt.plot(time[:T_trim], xx_mpc[i, :T_trim], 'g', label='System Trajectory')
        plt.xlabel("Time [s]")
        plt.ylabel(f"State {i+1}")
        plt.grid(True)
        if i == 0:
            plt.legend()
    plt.tight_layout()
    #plt.savefig("mpc_state_tracking.eps", format='eps', dpi=300)
    plt.show()

    if perturbed:
        # Tracking error plot for position states
        plt.figure(figsize=(15, 10))
        for i in range(4):  
            plt.subplot(4, 1, i + 1)
            plt.plot(time[:T_trim], xx_mpc[i, :T_trim] - xx_star[i, :T_trim])
            plt.xlabel("Time [s]")
            plt.ylabel(f"Error $x_{i+1} - x_{{\\mathrm{{ref}},{i+1}}}$")
            plt.grid(True)
            plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
            if i == 0:
                plt.legend()
        plt.tight_layout()
        #plt.savefig("mpc_position_error.eps", format='eps', dpi=300)
        plt.show()

    # Input trajectories
    plt.figure(figsize=(10, 5))
    for i in range(ni):
        plt.subplot(1, 2, i + 1)
        plt.plot(time[:T_trim], uu_star[i, :T_trim], 'k--', label='Desired Trajectory')
        plt.plot(time[:T_trim], uu_mpc[i, :T_trim], 'r', label='System Trajectory')
        plt.xlabel("Time [s]")
        plt.ylabel(f"Input {i+1}")
        plt.grid(True)
        if i == 0:
            plt.legend()
    plt.tight_layout()
    #plt.savefig("mpc_input_tracking.eps", format='eps', dpi=300)
    plt.show()

if LQR:
    # state trajectories
    plt.figure(figsize=(15, 10))
    for i in range(ns):
        plt.subplot(4, 2, i + 1)
        plt.plot(time, xx_star[i, :], 'k--', label='Desired Optimal Trajectory')
        plt.plot(time, xx_track[i, :], 'g', label='System Trajectory')
        plt.xlabel("Time [s]")
        plt.ylabel(f"State {i+1}")
        plt.grid(True)
        if i == 0:
            plt.legend()
    plt.tight_layout()
    #plt.savefig("lqr_state_tracking.eps", format='eps', dpi=300)
    plt.show()

    if perturbed:
        # Tracking error plot for position states
        plt.figure(figsize=(15, 10))
        for i in range(4):  
            plt.subplot(4, 1, i + 1)
            plt.plot(time, xx_track[i, :] - xx_star[i, :])
            plt.xlabel("Time [s]")
            plt.ylabel(f"Error $x_{i+1} - x_{{\\mathrm{{ref}},{i+1}}}$")
            plt.grid(True)
            plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
            if i == 0:
                plt.legend()
        plt.tight_layout()
        #plt.savefig("lqr_position_error.eps", format='eps', dpi=300)
        plt.show()

    # Input trajectories
    plt.figure(figsize=(10, 5))
    for i in range(ni):
        plt.subplot(1, 2, i + 1)
        plt.plot(time[:-1], uu_star[i, :-1], 'k--', label='Desired Trajectory')
        plt.plot(time[:-1], uu_track[i, :-1], 'r', label='System Trajectory')
        plt.xlabel("Time [s]")
        plt.ylabel(f"Input {i+1}")
        plt.grid(True)
        if i == 0:
            plt.legend()
    plt.tight_layout()
    #plt.savefig("lqr_input_tracking.eps", format='eps', dpi=300)
    plt.show()
    
if Animation:
    animate_surface(xx_star, xx_ref, dt)