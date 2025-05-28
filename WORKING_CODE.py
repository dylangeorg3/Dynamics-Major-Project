# Load the necessary libraries
import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.optimize as opt

# System parameters
# For bottom piston (1 is lowest 6 is highest)
m1, m2, m3 = 0.5, 1.5, 0.5
L1, L2 = 0.5, 1.5
I1, I2 = 0.5, 1.5
F0 = 50
LP = 0.5
k = 1
g = 9.81

# For top piston
m4, m5, m6 = 0.5, 1.5, 0.5
L5, L6 = 1.5, 0.5
I5, I6 = 1.5, 0.5

# Spring parameters connecting the pistons
k_spring = 10.0  # Spring stiffness
L0_spring = 0.5  # Natural length of spring connecting pistons

#==============================================================================
# BOTTOM CRANK-SLIDER SYSTEM
#==============================================================================

class BottomCrankSlider:
    def __init__(self):
        # Define symbolic variables for bottom system
        self.t = sp.symbols('t')
        self.x1, self.y1, self.theta1 = dynamicsymbols('x1 y1 theta1')
        self.x2, self.y2, self.theta2 = dynamicsymbols('x2 y2 theta2')
        self.x3, self.y3 = dynamicsymbols('x3 y3')
        
        self.q = sp.Matrix([self.x1, self.y1, self.theta1, 
                           self.x2, self.y2, self.theta2, 
                           self.x3, self.y3])
        self.dq = self.q.diff(self.t)
        
        # Mass matrix for bottom system
        self.M = np.diag([m1, m1, I1, m2, m2, I2, m3, m3])
        self.W = np.linalg.inv(self.M)
        
        # Setup constraints and dynamics
        self._setup_constraints()
        self._setup_dynamics()
    
    def _setup_constraints(self):
        # Rotation matrix
        R = lambda theta: sp.Matrix([[sp.cos(theta), -sp.sin(theta)], 
                                   [sp.sin(theta), sp.cos(theta)]])
        
        # Unit vectors
        i_cap = sp.Matrix([1, 0])
        j_cap = sp.Matrix([0, 1])
        
        # Center of mass positions
        x_com_1 = sp.Matrix([self.x1, self.y1])
        x_com_2 = sp.Matrix([self.x2, self.y2])
        x_com_3 = sp.Matrix([self.x3, self.y3])
        
        # Constraint 1: Pin joint at base (link 1 end at origin)
        constraint_1 = x_com_1 + R(self.theta1) @ sp.Matrix([-L1/2, 0])
        C1 = constraint_1.dot(i_cap)
        C2 = constraint_1.dot(j_cap)
        
        # Constraint 2: Link 1 to Link 2 connection
        constraint_2 = (x_com_1 - x_com_2 + R(self.theta1) @ sp.Matrix([L1/2, 0]) - 
                       R(self.theta2) @ sp.Matrix([-L2/2, 0]))
        C3 = constraint_2.dot(i_cap)
        C4 = constraint_2.dot(j_cap)
        
        # Constraint 3: Link 2 to piston connection
        constraint_3 = x_com_2 + R(self.theta2) @ sp.Matrix([L2/2, 0]) - x_com_3
        C5 = constraint_3.dot(i_cap)
        C6 = constraint_3.dot(j_cap)
        
        # Constraint 4: Piston moves only vertically
        C7 = x_com_3[0]
        
        self.C = sp.Matrix([C1, C2, C3, C4, C5, C6, C7])
        self.J = self.C.jacobian(self.q)
    
    def _setup_dynamics(self):
        # Create lambdified functions
        self.J_fn = sp.lambdify(args=(self.q, self.dq), expr=self.J)
        self.C_fn = sp.lambdify(args=(self.q, self.dq), expr=self.C)
        
        # Calculate derivatives
        dC = self.J @ self.dq
        dJ = dC.jacobian(self.q)
        self.dC_fn = sp.lambdify(args=(self.q, self.dq), expr=dC)
        self.dJ_fn = sp.lambdify(args=(self.q, self.dq), expr=dJ)
        
        # System matrices
        JWJT = self.J @ self.W @ self.J.T
        self.JWJT_fn = sp.lambdify(args=(self.q, self.dq), expr=JWJT)
    
    def get_force_vector(self, q_vals, dq_vals, spring_force):
        """Generate force vector including spring force from top system"""
        Q = np.array([
            0, -m1*g, -k * dq_vals[2],  # body 1 (theta1_dot damping)
            0, -m2*g, 0,                # body 2
            0, F0 * np.cos(q_vals[2]) - m3 * g + spring_force  # piston 3 + spring
        ])
        return Q
    
    def dynamics(self, q_vals, dq_vals, spring_force):
        """Calculate accelerations for bottom system"""
        try:
            Q = self.get_force_vector(q_vals, dq_vals, spring_force)
            
            # Calculate RHS for constraint equation
            RHS = (-self.dJ_fn(q_vals, dq_vals) @ dq_vals - 
                   self.J_fn(q_vals, dq_vals) @ self.W @ Q - 
                   10 * self.C_fn(q_vals, dq_vals).flatten() - 
                   10 * self.dC_fn(q_vals, dq_vals).flatten())
            
            # Solve for Lagrange multipliers
            JWJT_matrix = self.JWJT_fn(q_vals, dq_vals)
            
            # Check for singularity
            if np.linalg.cond(JWJT_matrix) > 1e12:
                return np.zeros(8)
                
            lam = np.linalg.solve(JWJT_matrix, RHS)
            
            # Calculate constraint forces
            Qhat = self.J_fn(q_vals, dq_vals).T @ lam
            
            # Calculate accelerations
            ddq = self.W @ (Q + Qhat)
            return ddq.flatten()
        except:
            return np.zeros(8)

#==============================================================================
# TOP CRANK-SLIDER SYSTEM - PROPERLY MIRRORED VERSION
#==============================================================================

class TopCrankSlider:
    def __init__(self):
        # Define symbolic variables for top system
        self.t = sp.symbols('t')
        self.x4, self.y4 = dynamicsymbols('x4 y4')
        self.x5, self.y5, self.theta3 = dynamicsymbols('x5 y5 theta3')
        self.x6, self.y6, self.theta4 = dynamicsymbols('x6 y6 theta4')
        
        self.q = sp.Matrix([self.x4, self.y4, 
                           self.x5, self.y5, self.theta3, 
                           self.x6, self.y6, self.theta4])
        self.dq = self.q.diff(self.t)
        
        # Mass matrix for top system
        self.M = np.diag([m4, m4, m5, m5, I5, m6, m6, I6])
        self.W = np.linalg.inv(self.M)
        
        # Setup constraints and dynamics
        self._setup_constraints()
        self._setup_dynamics()
    
    def _setup_constraints(self):
        # Rotation matrix
        R = lambda theta: sp.Matrix([[sp.cos(theta), -sp.sin(theta)], 
                                   [sp.sin(theta), sp.cos(theta)]])
        
        # Unit vectors
        i_cap = sp.Matrix([1, 0])
        j_cap = sp.Matrix([0, 1])
        
        # Center of mass positions
        x_com_4 = sp.Matrix([self.x4, self.y4])
        x_com_5 = sp.Matrix([self.x5, self.y5])
        x_com_6 = sp.Matrix([self.x6, self.y6])
        
        # FIXED: Pin joint at top - Link 6 end is fixed at a specific height
        # This is the mirror of the bottom system where link 1 end is at origin
        fixed_height = 4.5  # Fixed top point
        constraint_1 = x_com_6 + R(self.theta4) @ sp.Matrix([L6/2, 0]) - sp.Matrix([0, fixed_height])
        C1 = constraint_1.dot(i_cap)
        C2 = constraint_1.dot(j_cap)
        
        # FIXED: Link 6 to Link 5 connection (mirror of link 1 to link 2)
        # In bottom: link1_center + R(theta1)[L1/2,0] connects to link2_center + R(theta2)[-L2/2,0]
        # In top: link6_center + R(theta4)[-L6/2,0] connects to link5_center + R(theta3)[L5/2,0]
        constraint_2 = (x_com_6 + R(self.theta4) @ sp.Matrix([-L6/2, 0]) - 
                       x_com_5 - R(self.theta3) @ sp.Matrix([L5/2, 0]))
        C3 = constraint_2.dot(i_cap)
        C4 = constraint_2.dot(j_cap)
        
        # FIXED: Link 5 to piston connection (mirror of link 2 to piston)
        # In bottom: link2_center + R(theta2)[L2/2,0] connects to piston
        # In top: link5_center + R(theta3)[-L5/2,0] connects to piston
        constraint_3 = x_com_5 + R(self.theta3) @ sp.Matrix([-L5/2, 0]) - x_com_4
        C5 = constraint_3.dot(i_cap)
        C6 = constraint_3.dot(j_cap)
        
        # Constraint 4: Piston moves only vertically (x-coordinate is zero)
        C7 = x_com_4[0]
        
        self.C = sp.Matrix([C1, C2, C3, C4, C5, C6, C7])
        self.J = self.C.jacobian(self.q)
    
    def _setup_dynamics(self):
        # Create lambdified functions
        self.J_fn = sp.lambdify(args=(self.q, self.dq), expr=self.J)
        self.C_fn = sp.lambdify(args=(self.q, self.dq), expr=self.C)
        
        # Calculate derivatives
        dC = self.J @ self.dq
        dJ = dC.jacobian(self.q)
        self.dC_fn = sp.lambdify(args=(self.q, self.dq), expr=dC)
        self.dJ_fn = sp.lambdify(args=(self.q, self.dq), expr=dJ)
        
        # System matrices
        JWJT = self.J @ self.W @ self.J.T
        self.JWJT_fn = sp.lambdify(args=(self.q, self.dq), expr=JWJT)
    
    def get_force_vector(self, q_vals, dq_vals, spring_force):
        """Generate force vector including spring force from bottom system"""
        # FIXED: Mirror the force pattern from bottom system
        # Bottom has: F0*cos(theta1) on piston, damping on theta1
        # Top should have: F0*cos(theta4) on piston, damping on theta4 (link 6 is the driving link)
        Q = np.array([
            0, F0 * np.cos(q_vals[7]) - m4 * g - spring_force,  # piston 4 with driving force (negative spring force)
            0, -m5*g, 0,                                         # body 5 (link 5)
            0, -m6*g, -k * dq_vals[7]                           # body 6 with damping on theta4 (driving link)
        ])
        return Q
    
    def dynamics(self, q_vals, dq_vals, spring_force):
        """Calculate accelerations for top system"""
        try:
            Q = self.get_force_vector(q_vals, dq_vals, spring_force)
            
            # Calculate RHS for constraint equation
            RHS = (-self.dJ_fn(q_vals, dq_vals) @ dq_vals - 
                   self.J_fn(q_vals, dq_vals) @ self.W @ Q - 
                   10 * self.C_fn(q_vals, dq_vals).flatten() - 
                   10 * self.dC_fn(q_vals, dq_vals).flatten())
            
            # Solve for Lagrange multipliers
            JWJT_matrix = self.JWJT_fn(q_vals, dq_vals)
            
            # Check for singularity
            if np.linalg.cond(JWJT_matrix) > 1e12:
                return np.zeros(8)
                
            lam = np.linalg.solve(JWJT_matrix, RHS)
            
            # Calculate constraint forces
            Qhat = self.J_fn(q_vals, dq_vals).T @ lam
            
            # Calculate accelerations
            ddq = self.W @ (Q + Qhat)
            return ddq.flatten()
        except:
            return np.zeros(8)

#==============================================================================
# COUPLED SYSTEM SOLVER - PROPERLY MIRRORED VERSION
#==============================================================================

class CoupledPistonSystem:
    def __init__(self):
        self.bottom_system = BottomCrankSlider()
        self.top_system = TopCrankSlider()
        
        # Angular velocities for driving links
        self.dtheta1 = 2.0  # Bottom system angular velocity (link 1)
        self.dtheta4 = -2.0  # Top system angular velocity (link 6, negative for opposite rotation)
    
    def calculate_spring_force(self, y3, y4):
        """Calculate spring forces between pistons"""
        spring_length = abs(y4 - y3)
        spring_extension = spring_length - L0_spring
        spring_force_magnitude = k_spring * spring_extension
        
        # Spring force direction (positive when spring is extended)
        if y4 > y3:  # Top piston above bottom piston
            spring_force_on_bottom = spring_force_magnitude    # Upward force on bottom
            spring_force_on_top = -spring_force_magnitude      # Downward force on top
        else:  # Compressed spring
            spring_force_on_bottom = -spring_force_magnitude   # Downward force on bottom
            spring_force_on_top = spring_force_magnitude       # Upward force on top
        
        return spring_force_on_bottom, spring_force_on_top

    def get_initial_conditions(self):
        """Calculate initial conditions for both systems - FIXED VERSION"""
        
        # Bottom system initial positions (unchanged - these work fine)
        initial_pos_bottom = np.array([
            0, L1/2, np.pi/4,        # Link 1 (x1, y1, theta1) - start at 45 degrees
            0, L1 + L2/2, np.pi/2,   # Link 2 (x2, y2, theta2)  
            0, L1 + L2               # Bottom piston (x3, y3)
        ])
        
        # FIXED: Top system initial positions - proper geometric calculation
        fixed_height = 4.5
        
        # Start with desired angles that mirror the bottom system
        theta4_init = 3*np.pi/4  # Link 6 angle (mirror of bottom link 1 at pi/4)
        theta3_init = np.pi/2    # Link 5 angle (same as bottom link 2)
        
        # Link 6 positioning: its right end is fixed at (0, fixed_height)
        # Link 6 center = fixed_point - (L6/2) * [cos(theta4), sin(theta4)]
        x6_init = 0 - (L6/2) * np.cos(theta4_init)
        y6_init = fixed_height - (L6/2) * np.sin(theta4_init)
        
        # Connection point between link 6 and link 5 (left end of link 6)
        # Left end of link 6 = link6_center - (L6/2) * [cos(theta4), sin(theta4)]
        connection_x = x6_init - (L6/2) * np.cos(theta4_init)
        connection_y = y6_init - (L6/2) * np.sin(theta4_init)
        
        # Link 5 positioning: its right end connects to link 6's left end
        # Link 5 center = connection_point - (L5/2) * [cos(theta3), sin(theta3)]
        x5_init = connection_x - (L5/2) * np.cos(theta3_init)
        y5_init = connection_y - (L5/2) * np.sin(theta3_init)
        
        # Top piston positioning: connects to left end of link 5
        # Left end of link 5 = link5_center - (L5/2) * [cos(theta3), sin(theta3)]
        x4_init = 0  # Constrained to x = 0
        y4_init = y5_init - (L5/2) * np.sin(theta3_init)
        
        initial_pos_top = np.array([
            x4_init, y4_init,                    # Top piston (x4, y4)
            x5_init, y5_init, theta3_init,       # Link 5 (x5, y5, theta3)
            x6_init, y6_init, theta4_init        # Link 6 (x6, y6, theta4)
        ])
        
        # Debug print to check positions
        print(f"Bottom piston initial y: {initial_pos_bottom[7]:.3f}")
        print(f"Top piston initial y: {y4_init:.3f}")
        print(f"Initial spring length: {abs(y4_init - initial_pos_bottom[7]):.3f}")
        print(f"Link 6 center: ({x6_init:.3f}, {y6_init:.3f})")
        print(f"Link 5 center: ({x5_init:.3f}, {y5_init:.3f})")
        
        # Initial velocities - only driving links have initial angular velocity
        initial_vel_bottom = np.array([0, 0, self.dtheta1, 0, 0, 0, 0, 0])
        initial_vel_top = np.array([0, 0, 0, 0, 0, 0, 0, self.dtheta4])
        
        return (np.concatenate([initial_pos_bottom, initial_vel_bottom]),
                np.concatenate([initial_pos_top, initial_vel_top]))
    
    def coupled_dynamics(self, t, state):
        """Combined dynamics function for both systems"""
        # Split state into bottom and top systems
        n_bottom = len(self.bottom_system.q)
        n_top = len(self.top_system.q)
        
        state_bottom = state[:2*n_bottom]
        state_top = state[2*n_bottom:]
        
        # Extract positions and velocities
        q_bottom, dq_bottom = np.split(state_bottom, 2)
        q_top, dq_top = np.split(state_top, 2)
        
        # Get piston positions for spring force calculation
        y3 = q_bottom[7]  # Bottom piston y-position
        y4 = q_top[1]     # Top piston y-position
        
        # Calculate spring forces
        spring_force_bottom, spring_force_top = self.calculate_spring_force(y3, y4)
        
        try:
            # Calculate accelerations for both systems
            ddq_bottom = self.bottom_system.dynamics(q_bottom, dq_bottom, spring_force_bottom)
            ddq_top = self.top_system.dynamics(q_top, dq_top, spring_force_top)
            
            # Return combined derivative
            return np.concatenate([dq_bottom, ddq_bottom, dq_top, ddq_top])
        except Exception as e:
            # Return zero derivatives to prevent integration failure
            return np.zeros_like(state)

#==============================================================================
# SIMULATION AND VISUALIZATION
#==============================================================================

# Create the coupled system
print("Creating coupled system...")
system = CoupledPistonSystem()

# Get initial conditions
print("Calculating initial conditions...")
x0_bottom, x0_top = system.get_initial_conditions()
x0 = np.concatenate([x0_bottom, x0_top])

# Run simulation
print("Running simulation...")
t_span = (0, 5)
t_eval = np.linspace(*t_span, 250)

sol = solve_ivp(system.coupled_dynamics, t_span, x0, 
                atol=1e-6, rtol=1e-6, method='RK45', t_eval=t_eval,
                max_step=0.01)

if sol.success:
    print(f"Simulation completed successfully! Final time: {sol.t[-1]:.2f} seconds")
    
    # Extract results for visualization
    n_bottom = len(system.bottom_system.q)

    # Bottom system results
    x1, y1, theta1 = sol.y[0], sol.y[1], sol.y[2]
    x2, y2, theta2 = sol.y[3], sol.y[4], sol.y[5]  
    x3, y3 = sol.y[6], sol.y[7]

    # Top system results (offset by bottom system size)
    offset = 2 * n_bottom
    x4, y4 = sol.y[offset], sol.y[offset+1]
    x5, y5, theta3 = sol.y[offset+2], sol.y[offset+3], sol.y[offset+4]
    x6, y6, theta4 = sol.y[offset+5], sol.y[offset+6], sol.y[offset+7]

    # Print some debugging info
    print(f"Bottom piston y-range: {np.min(y3):.3f} to {np.max(y3):.3f}")
    print(f"Top piston y-range: {np.min(y4):.3f} to {np.max(y4):.3f}")
    print(f"Spring length range: {np.min(np.abs(y4-y3)):.3f} to {np.max(np.abs(y4-y3)):.3f}")
    print(f"Link 1 theta1 range: {np.min(theta1):.3f} to {np.max(theta1):.3f} rad")
    print(f"Link 6 theta4 range: {np.min(theta4):.3f} to {np.max(theta4):.3f} rad")

    # Visualization classes for animation
    class Box:
        def __init__(self, width, height, color='b'):
            self.width = width
            self.height = height
            self.color = color
            self.offset = -np.array([width/2, height/2])

        def first_draw(self, ax):
            corner = np.array([0, 0])
            self.patch = plt.Rectangle(corner, self.width, self.height, 
                                     angle=0, color=self.color, alpha=0.7)
            ax.add_patch(self.patch)
            self.ax = ax
            return self.patch
        
        def set_data(self, x, y, theta):
            self.x = x
            self.y = y
            self.theta = theta

        def update(self, i):
            x, y, theta = self.x[i], self.y[i], self.theta[i]
            
            # Calculate rotation and translation
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            # Corner position after rotation and translation
            offset_rotated = np.array([
                self.offset[0] * cos_theta - self.offset[1] * sin_theta,
                self.offset[0] * sin_theta + self.offset[1] * cos_theta
            ])
            corner = np.array([x, y]) + offset_rotated
            
            # Update patch
            self.patch.set_xy(corner)
            self.patch.set_angle(np.rad2deg(theta))
            return self.patch

    class Spring:
        def __init__(self, n_coils=4, amplitude=0.05, color='k', linewidth=2):
            self.n_coils = n_coils
            self.amplitude = amplitude
            self.color = color
            self.linewidth = linewidth

        def first_draw(self, ax):
            self.line, = ax.plot([], [], color=self.color, lw=self.linewidth)
            return self.line

        def set_data(self, x1, y1, x2, y2):
            self.x1 = x1
            self.y1 = y1
            self.x2 = x2
            self.y2 = y2

        def update(self, i):
            x_start = self.x1[i]
            y_start = self.y1[i]
            x_end = self.x2[i]
            y_end = self.y2[i]

            num_points = self.n_coils * 2 + 2
            t = np.linspace(0, 1, num_points)
            base_x = (1 - t) * x_start + t * x_end
            base_y = (1 - t) * y_start + t * y_end

            dx = x_end - x_start
            dy = y_end - y_start
            length = np.hypot(dx, dy)
            if length == 0:
                direction = np.array([1, 0])
            else:
                direction = np.array([dx, dy]) / length
            normal = np.array([-direction[1], direction[0]])

            zigs = np.concatenate(([0], np.tile([1, -1], self.n_coils), [0]))
            offsets = self.amplitude * zigs

            spring_x = base_x + offsets * normal[0]
            spring_y = base_y + offsets * normal[1]

            self.line.set_data(spring_x, spring_y)
            return self.line

    # Create animation
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(8, 10))

    ax.set_ylim(-0.6, 5.1)
    ax.set_xlim(-1, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Create visual components
    theta_pistons = np.zeros_like(x3)

    box1 = Box(L1, 0.02, 'b')       # Link 1
    box2 = Box(L2, 0.02, 'r')       # Link 2
    box3 = Box(0.1, 0.3, 'g')       # Piston 3 (bottom)
    box4 = Box(0.1, 0.3, 'g')       # Piston 4 (top)
    box5 = Box(L5, 0.02, 'r')       # Link 5
    box6 = Box(L6, 0.02, 'b')       # Link 6

    spring = Spring(n_coils=6, amplitude=0.08, color='orange', linewidth=3)

    # Set animation data
    box1.set_data(x1, y1, theta1)
    box2.set_data(x2, y2, theta2)
    box3.set_data(x3, y3, theta_pistons)
    box4.set_data(x4, y4, theta_pistons)
    box5.set_data(x5, y5, theta3)
    box6.set_data(x6, y6, theta4)
    spring.set_data(x3, y3, x4, y4)

    boxes = [box3, box4, box1, box2, box5, box6]

    def init():
        ax.set_title("Properly Mirrored Dual Piston System: t=0.00 sec", fontsize=15)
        patches = []
        for box in boxes:
            patches.append(box.first_draw(ax))
        patches.append(spring.first_draw(ax))
        return patches

    def animate(frame):
        ax.set_title(f"Properly Mirrored Dual Piston System: t={sol.t[frame]:.2f} sec", fontsize=15)
        patches = []
        for box in boxes:
            patches.append(box.update(frame))
        patches.append(spring.update(frame))
        return patches

    # Create and show animation
    anim = FuncAnimation(fig, animate, frames=len(sol.t), init_func=init, 
                        blit=True, interval=50, repeat=True)

    plt.show()

    # Save animation
    from matplotlib.animation import PillowWriter
    print("Saving animation...")
    anim.save("properly_mirrored_dual_piston_system.gif", writer=PillowWriter(fps=20))
    print("Animation saved successfully!")

else:
    print(f"Simulation failed: {sol.message}")

print("\nProperly mirrored dual piston system simulation complete!")
