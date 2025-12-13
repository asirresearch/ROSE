import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
from matplotlib.widgets import Button

def animate_surface(xx_star, xx_ref, dt):
    """
    Animates the surface dynamics
    input parameters:
        - Optimal state trajectory xx_star
        - Reference trajectory xx_ref
        - Sampling time dt
    """
    # Calculate total trajectory time and number of frames
    TT = xx_star.shape[1]
    total_time = dt * TT  # Total trajectory time in seconds
    
    # For real-time playback, we want fewer frames while maintaining smooth animation
    desired_fps = 30  # Standard frame rate for smooth animation
    
    # Calculate frame sampling to achieve real-time playback
    # Number of frames needed for smooth animation over total_time
    n_frames = int(total_time * desired_fps)
    # Create time indices for sampling the trajectory
    frame_indices = np.linspace(0, TT-1, n_frames, dtype=int)

    # System geometry parameters
    d = 0.30
    total_length = 5*d
    x_points = np.array([d, 2*d, 3*d, 4*d])
    x_interp = np.linspace(0, total_length, 200)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate axis limits
    z_min = min(np.min(xx_ref[:4,:]), np.min(xx_star[:4,:]))
    z_max = max(np.max(xx_ref[:4,:]), np.max(xx_star[:4,:]))
    margin = max(abs(z_min), abs(z_max)) * 0.2
    y_limit = max(abs(z_min - margin), abs(z_max + margin))
    
    # Set up plot elements
    ax.set_xlim(-0.1, total_length + 0.1)
    ax.set_ylim(-y_limit, y_limit)
    
    surface_points, = ax.plot([], [], 'bo', label='Measurement Points', markersize=8)
    ref_points, = ax.plot([], [], 'ro', label='Reference Points', markersize=8)
    surface_line, = ax.plot([], [], 'b-', label='Optimal Path', linewidth=2)
    ref_line, = ax.plot([], [], 'r--', label='Reference Surface', linewidth=2)
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
    
    # Add fixed supports
    ax.plot([0], [0], 'k^', markersize=10, label='Fixed Supports')
    ax.plot([total_length], [0], 'k^', markersize=10)

    # Configure plot
    ax.legend(loc='upper right')
    ax.set_title('Flexible Surface Trajectory Tracking')
    ax.set_xlabel('Horizontal Position')
    ax.set_ylabel('Vertical Displacement')
    ax.grid(True)

    def init():
        """Initialize animation"""
        surface_points.set_data([], [])
        ref_points.set_data([], [])
        surface_line.set_data([], [])
        ref_line.set_data([], [])
        time_text.set_text('')
        return surface_points, ref_points, surface_line, ref_line, time_text

    def update(frame_idx):
        """Update function for each frame"""
        # Get actual trajectory index
        t_idx = frame_indices[frame_idx]
        
        # Get current positions
        z_points_opt = xx_star[:4, t_idx]
        z_points_ref = xx_ref[:4, t_idx]

        # Create full arrays with fixed points
        x_full = np.concatenate(([0], x_points, [total_length]))
        z_full_opt = np.concatenate(([0], z_points_opt, [0]))
        z_full_ref = np.concatenate(([0], z_points_ref, [0]))

        # Create smooth curves
        z_interp_opt = interp1d(x_full, z_full_opt, kind='cubic')
        z_interp_ref = interp1d(x_full, z_full_ref, kind='cubic')

        # Update plot elements
        surface_points.set_data(x_points, z_points_opt)
        ref_points.set_data(x_points, z_points_ref)
        surface_line.set_data(x_interp, z_interp_opt(x_interp))
        ref_line.set_data(x_interp, z_interp_ref(x_interp))
        time_text.set_text(f'Time: {t_idx*dt:.2f} s')

        return surface_points, ref_points, surface_line, ref_line, time_text

    # Calculate interval for real-time playback
    interval = (total_time * 1000) / n_frames  # Convert to milliseconds
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=n_frames,
                       init_func=init, blit=True,
                       interval=interval)
    print("total time: ", total_time)
     # PAUSE/RESUME BUTTON
    anim_running = True  # Track animation state

    def toggle_animation(event):
        """Pause or resume animation when button is clicked."""
        nonlocal anim_running
        if anim_running:
            ani.event_source.stop()
            pause_button.label.set_text("Resume")
        else:
            ani.event_source.start()
            pause_button.label.set_text("Pause")
        anim_running = not anim_running

    # Add button for pausing animation
    ax_button = plt.axes([0.8, 0.05, 0.1, 0.075])  # [left, bottom, width, height]
    pause_button = Button(ax_button, "Pause")
    pause_button.on_clicked(toggle_animation)

    plt.show()
    return ani