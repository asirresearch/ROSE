README.txt

Flexible Surface Optimal Control and Tracking

Description:
------------
This project implements trajectory generation and tracking for a flexible surface model using various optimal control methods: Armijo-based optimization, Linear Quadratic Regulator (LQR), and Model Predictive Control (MPC). It includes simulation, optimization, animation, and visualization tools to compare tracking accuracy and control performance.

Project Structure:
------------------
1. main.py
   - Main script to run trajectory generation and tracking.
   - Selects control method (Armijo, LQR, MPC) via flags.
   - Plots results and optionally animates the surface.

2. animation.py
   - Creates a real-time animation of the flexible surface motion.
   - Shows both the optimal and reference trajectories.

3. armijo_rule.py
   - Implements the Armijo line search for gradient-based step size selection.
   - Includes plotting tools to visualize descent behavior.

4. cost.py
   - Defines stage and terminal cost functions with quadratic penalties.
   - Provides cost gradients used in optimization.

5. find_equilibria.py
   - Solves for equilibrium points numerically using symbolic system definitions.

6. fs_dynamics_staticps.py
   - Core module defining system dynamics and computing their gradients.
   - Uses symbolic definitions and generates callable functions.

7. mpc_controller.py
   - Formulates and solves a constrained MPC problem using CVXPY.
   - Supports state/input constraints and terminal cost.

8. ref_traj_gen.py
   - Generates smooth reference trajectories using quintic polynomials.
   - Produces forward, stop, backward, stop motion patterns.

9. solver_LQR.py
   - Computes time-varying LQR gains via backward Riccati recursion.
   - Supports affine terms for trajectory optimization.

Requirements:
-------------
- Python 3.8+
- NumPy
- SciPy
- SymPy
- Matplotlib
- CVXPY
- Control

Run Instructions:
-----------------
1. Set the desired control method by adjusting flags in `main.py`:
   - `Armijo = True` for iterative optimization.
   - `LQR = True` for feedback tracking.
   - `MPC = True` for model predictive control.
2. Toggle `perturbed = True` to simulate initial state deviation.
3. Set `Animation = True` to enable real-time trajectory visualization.
4. Run the project:
   ```bash
   python main.py
