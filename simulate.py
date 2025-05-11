#----------------------------------------------------------------------#
# Title: Spring-Coupled Opposed Pistons Simulation
# Author: Dylan George
# Date: 2025-05-09
# Description: 
#   This script models the dynamics of a planar mechanism comprising two 
#   opposed pistons connected by a spring. Each piston is linked to a 
#   crankshaft via a connecting rod, and the system includes multiple 
#   translational and rotational degrees of freedom. The spring coupling 
#   serves to dampen oscillations and smooth out periodic forcing.
#
#   Symbolic methods (SymPy) are used to define generalized coordinates, 
#   derive constraint equations, and express the motion of the system. 
#   The system is then numerically integrated using SciPy to simulate 
#   time-domain behavior and evaluate system response under various conditions.
#
#   Further analysis includes:
#     - Force transmission and reaction forces
#     - Energy conservation and dissipation
#     - Effect of spring stiffness on motion smoothness (to be added)
#
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
# Packages
#----------------------------------------------------------------------#
# Load libraries
import numpy as np                                  # For numerical operations
import sympy as sp                                  # For symbolic mathematics
from sympy.physics.mechanics import dynamicsymbols  # For dynamics symbols
import matplotlib.pyplot as plt                     # For plotting
from scipy.integrate import solve_ivp               # For numerical integration
import scipy.optimize as opt                        # For optimization
from matplotlib.animation import FuncAnimation      # For animation
from IPython.display import HTML                    # For displaying animations in Jupyter




#----------------------------------------------------------------------#
# System parameters
#----------------------------------------------------------------------#
m1r, m2r, m3p, m4p, m5r, m6r = 1, 1, 0.5, 0.5, 1, 1    # Masses of components (kg) - Left to right
L1, L2, L3, L4 = 1, 1, 1, 1                            # Lengths of piston rods (m) - Left to right
I1, I2, I3, I4 = 1, 1, 1, 1                            # Inertia of components (kg*m^2) - Left to right
PL = 0.5                                               # Length of pistons (m)
F0 = 50                       # Max force of forcing function (N)
r1, r2, r3, r4 = 1, 1, 1, 1   # Resistance for torque vectors acting on shafts (N*m/rad)
k = 50                        # Spring constant (N/m)
g = 9.81                      # Gravity (ms^-2)


#----------------------------------------------------------------------#
# Symbolic variables
#----------------------------------------------------------------------#
t = sp.symbols('t')

# Rods centers of mass positions
x1r, x2r, x5r, x6r = dynamicsymbols('x1r x2r x5r x6r')
y1r, y2r, y5r, y6r = dynamicsymbols('y1r y2r y5r y6r')                  

# Piston center of mass positions
x3p, x4p = dynamicsymbols('x3p x4p')
y3p, y4p = dynamicsymbols('y3p y4p')

# Rotor angular positions
theta1, theta2, theta3, theta4 = dynamicsymbols('theta1 theta2 theta3 theta4')           

#----------------------------------------------------------------------#
# State, position, mass, inertia, external force vectors
#----------------------------------------------------------------------#
# State vector
SV = sp.Matrix([
    x1r, y1r,
    x2r, y2r,
    x3p, y3p,
    x4p, y4p,
    x5r, y5r,
    x6r, y6r,
    theta1, theta2, theta3, theta4
])

# State vector derivatives
dSV = SV.diff(t)

# Centers of mass positions
x_com_1 = sp.Matrix([x1r, y1r])     # Left piston rod 1
x_com_2 = sp.Matrix([x2r, y2r])     # Left piston rod 2
x_com_3 = sp.Matrix([x3p, y3p])     # Left piston 
x_com_4 = sp.Matrix([x4p, y4p])     # Right piston
x_com_5 = sp.Matrix([x5r, y5r])     # Right piston rod 5
x_com_6 = sp.Matrix([x6r, y6r])     # Right piston rod 6

# Rotation matrix function
R = lambda theta: sp.Matrix([[sp.cos(theta), -sp.sin(theta)], [sp.sin(theta), sp.cos(theta)]])

# Mass matrix M (16x16) — mass for translation, inertia for rotation
M = sp.diag(
    m1r, m1r,       # x1r, y1r (Rod 1)
    m2r, m2r,       # x2r, y2r (Rod 2)
    m3p, m3p,       # x3p, y3p (Piston 1)
    m4p, m4p,       # x4p, y4p (Piston 2)
    m5r, m5r,       # x5r, y5r (Rod 3)
    m6r, m6r,       # x6r, y6r (Rod 4)
    I1, I2, I3, I4  # theta1–theta4 (rotational inertia of rods)
)

# Inverse mass matrix
#W = np.linalg.inv(M)
W = np.linalg.inv(np.array(M).astype(float))



L0 = PL  # Natural length of spring (m)
spring_force = k * ((x4p - x3p) - L0)

# External force and torques vector
FV = sp.Matrix([
    0, -m1r * g,                        # Gravity on rod 1
    0, -m2r * g,                        # Gravity on rod 2
    -F0 * sp.cos(theta1) + spring_force, 0,  # Piston 3: forcing + spring
    F0 * sp.cos(theta4) - spring_force, 0,  # Piston 4: forcing - spring
    0, -m5r * g,                        # Gravity on rod 5
    0, -m6r * g,                        # Gravity on rod 6
    -r1 * theta1.diff(t),               # Torque resistance on shaft 1
    -r2 * theta2.diff(t),               # Torque resistance on bearings between rods 1 & 2
    -r3 * theta3.diff(t),               # Torque resistance on bearings between rods 3 & 4
    -r4 * theta4.diff(t),               # Torque resistance on shaft 2
])



#----------------------------------------------------------------------#
# System constraints
#----------------------------------------------------------------------#

# X and Y unit vectors
i_cap = sp.Matrix([1, 0])
j_cap = sp.Matrix([0, 1])

# VERY UNSURE ABOUT ALL OF THESE CONSTRAINTS, I DON'T KNOW IF IT'S THESE
# OR THE EQUATIONS OF MOTION THAT ARE INCORRECT THOUGH

# Constraints of pin 1 - rod 1
constraint_1 = x_com_1 + R(theta1) @ sp.Matrix([-L1/2, 0])  
CiR1 = constraint_1.dot(i_cap) 
CjR1 = constraint_1.dot(j_cap) 

# Constraints of rod 1 - rod 2
constraint_2 = x_com_1 + R(theta1) @ sp.Matrix([L1/2, 0]) - x_com_2  - R(theta2) @ sp.Matrix([-L2/2, 0])

CiR2 = constraint_2.dot(i_cap) 
CjR2 = constraint_2.dot(j_cap) 

# Constraints of rod 2 - piston 3
constraint_3 = x_com_2 + R(theta2) @ sp.Matrix([L2/2, 0]) - x_com_3
CiP3 = constraint_3.dot(i_cap) 
CjP3 = constraint_3.dot(j_cap) 

# Prismatic constraints (pistons must stay horizontal)
CjP3_slide = x_com_3.dot(j_cap)  # y-position of piston 3 
CjP4_slide = x_com_4.dot(j_cap)  # y-position of piston 4 

# Constraints of piston 4 - rod 5
#constraint_4 = x_com_4 - R(theta3) @ sp.Matrix([0, L2/2]) - x_com_5
constraint_4 = x_com_5 + R(theta3) @ sp.Matrix([0, L3/2]) - x_com_4
CiP4 = constraint_4.dot(i_cap)
CjP4 = constraint_4.dot(j_cap)

# Constraints of rod 5 - rod 6
constraint_5 = x_com_6 - x_com_5 + R(theta4) @ sp.Matrix([L4/2, 0]) - R(theta3) @ sp.Matrix([-L3/2, 0])
CiR5 = constraint_5.dot(i_cap)
CjR5 = constraint_5.dot(j_cap)

# Constraints of rod 6 - pin 2
constraint_6 = x_com_6 - R(theta4) @ sp.Matrix([L4/2, 0])  
CiR6 = constraint_6.dot(i_cap)
CjR6 = constraint_6.dot(j_cap)



# Constraints vector
CV = sp.Matrix([
    CiR1, CjR1,
    CiR2, CjR2,
    CiP3, CjP3,
    CjP3_slide,
    CjP4, CiP4, 
    CjP4_slide,
    CiR5, CjR5,
    CiR6, CjR6
])

sp.pprint(CV)

#----------------------------------------------------------------------#
# Constraint Jacobian and Time Derivatives
#----------------------------------------------------------------------#

J = CV.jacobian(SV)     # Jacobian of constraint vector
dSV = SV.diff(t)        # Derivative of state vector (velocity) 
dC = J @ dSV            # Derivative of constraint vector
dJ = dC.jacobian(SV)    # Derivative of Jacobian

JWJT = J @ W @ J.T      # Jacobian times inverse mass matrix times transpose of Jacobian

# NOT WURE WHICH LINE IS THE CORRRECT EQUATION
#RHS = -dJ @ dSV - J @ W @ FV - 1 * CV - 1 * dC  # Right-hand side of the equation of motion
RHS = -dJ @ dSV - J @ W @ FV
# FIX THE EQUATION OF MOTION HERE DEPENDING ON WHATEVER IS THE CORRECT ONE

JWJT_fn = sp.lambdify(args=(SV, dSV), expr=JWJT)
RHS_fn = sp.lambdify(args=(SV, dSV), expr=RHS)
CV_fn = sp.lambdify(args=(SV, dSV), expr=CV)    
J_fn = sp.lambdify(args=(SV, dSV), expr=J)   
dC_fn = sp.lambdify(args=(SV, dSV), expr=dC)  
dJ_fn = sp.lambdify(args=(SV, dSV), expr=dJ)
FV_fn = sp.lambdify(args=(SV, dSV), expr=FV)



#----------------------------------------------------------------------#
# Initial Conditions (Horizontal Arrangement)
#----------------------------------------------------------------------#

# I just have three things to say
# God bless our troops,
# God bless America, 
# and gentlemen, 
# START
# YOUR
# ENGINEEESSSSSSSSSSSSSSSSSSS!!!!!!!!!!!

# Initial shaft rotation
dtheta1 = 0.5   # Starter motor 1
dtheta4 = 0.5   # Starter motor 2


# Initial positions of left engine
initial_position_rod1 = np.array([L1/2, 0, 0])
initial_position_rod2 = np.array([L1 + L2/2, 0, 0])
initial_position_piston3 = np.array([L1 + L2, 0])

# Intial positions of right engine
initial_position_piston4 = np.array([L1 + L2 + PL, 0])
initial_position_rod5 = np.array([L1 + L2 + PL + L3/2, 0, 0])
initial_position_rod6 = np.array([L1 + L2 + PL + L3 + L4/2, 0, 0])

# Initial velocity of left engine
initial_velocity_rod1 = np.array([0, 0, dtheta1]) # To start the engine
initial_velocity_rod2 = np.array([0, 0, 0])
initial_velocity_piston3 = np.array([0, 0])

# Initial velocity of right engine
initial_velocity_piston4 = np.array([0, 0])
initial_velocity_rod5 = np.array([0, 0, dtheta4]) # To start the engine
initial_velocity_rod6 = np.array([0, 0, 0])


# Initial state vector
x0 = np.concatenate((
    initial_position_rod1,
    initial_position_rod2,
    initial_position_piston3,
    initial_position_piston4,
    initial_position_rod5,
    initial_position_rod6,
    initial_velocity_rod1,
    initial_velocity_rod2,
    initial_velocity_piston3,
    initial_velocity_piston4,
    initial_velocity_rod5,
    initial_velocity_rod6
))



#----------------------------------------------------------------------#
# Test convergence and constraints satisfied 
#----------------------------------------------------------------------#

# Split x0 into positions and discard the velocities
x, _ = np.split(x0, 2)

# Optimiser to project valid initial velocity that satisfies dC = 0
def optimiser(b):
    dx1r, dy1r, dx2r, dy2r, dx3p, dy3p, dx4p, dy4p, dx5r, dy5r, dx6r, dy6r, dtheta2, dtheta3 = b
    dq = np.array([
        dx1r, dy1r,
        dx2r, dy2r,
        dx3p, dy3p,
        dx4p, dy4p,
        dx5r, dy5r,
        dx6r, dy6r,
        dtheta1, dtheta2, dtheta3, dtheta4
    ])
    return dC_fn(x, dq).flatten()

# Initial guess for unknown velocity components (excluding dtheta1 and dtheta4)
initial_guess = np.zeros(14)
result = opt.root(optimiser, initial_guess)

# Extract solution and build full velocity vector
b = result.x
dx = np.array([
    b[0],  b[1],   # dx1r, dy1r
    b[2],  b[3],   # dx2r, dy2r
    b[4],  b[5],   # dx3p, dy3p
    b[6],  b[7],   # dx4p, dy4p
    b[8],  b[9],   # dx5r, dy5r
    b[10], b[11],  # dx6r, dy6r
    dtheta1, b[12], b[13], dtheta4  # Angular velocities
])

# Evaluate constraint satisfaction
C_val = CV_fn(x, dx)
dC_val = dC_fn(x, dx)

print(f'Position constraint residual: {np.round(C_val, 4)}')
print(f'Velocity constraint residual: {np.round(dC_val, 4)}')

#assert np.allclose(C_val, 0, atol=1e-6), "Initial position constraint violated"
#assert np.allclose(dC_val, 0, atol=1e-6), "Initial velocity constraint violated"

# Final corrected initial state vector
x0 = np.concatenate((x, dx))


#----------------------------------------------------------------------#
# Piston engine function
#----------------------------------------------------------------------#

def piston_engine(t, state):
    '''
    This function returns the derivative of the state vector for the system

    Parameters:
    t: float
        The current time
    state: numpy array
        The current state of the system
        The vector is arranged as [SV, dSV]
        where q is the position vector and dq is the derivative of the position vector
    '''

    SV, dSV= np.split(state, 2)

    # Solve for lambda 
    lam = np.linalg.solve(JWJT_fn(SV,dSV), RHS_fn(SV,dSV))

    # Solve for constraint forces 
    Qhat = J_fn(SV, dSV).T @ lam

    # Calculate accelerations
    ddSV = W @ (FV_fn(SV, dSV) + Qhat)
    ddSV = ddSV.flatten()

    return np.concatenate((dSV, ddSV))

# Test run
piston_engine(0, x0)


# Solve the system of equations using the solve_ivp function
t_span = (0, 30)
t_eval = np.linspace(*t_span, 500)
sol = solve_ivp(piston_engine, t_span, x0, atol=1e-7, rtol=1e-7, method='BDF', t_eval=t_eval)


#%%
#----------------------------------------------------------------------#
# Box plotting class
#----------------------------------------------------------------------#
class Box:
    def __init__(self, width, height, color='b'):
        self.width = width
        self.height = height
        self.color = color
        self.offset = -np.array([width/2, height/2])

    def first_draw(self, ax):
        corner = np.array([0, 0])
        self.patch = plt.Rectangle(corner, 0, 0, angle=0, 
                        rotation_point='center', color=self.color, animated=True)
        ax.add_patch(self.patch)
        self.ax = ax
        return self.patch
    
    def set_data(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def update(self, i):
        x, y, theta = self.x[i], self.y[i], self.theta[i]
        theta = np.rad2deg(theta)

        # The rectangle is drawn from the left bottom corner
        # So, we need to calculate the corner position
        corner = np.array([x, y]) + self.offset

        # Update the values for the rectangle
        self.patch.set_width(self.width)
        self.patch.set_height(self.height)
        self.patch.set_xy(corner)
        self.patch.set_angle(theta)
        return self.patch
    

#----------------------------------------------------------------------#
# Spring plotting class
#----------------------------------------------------------------------#
class Spring:
    def __init__(self, n_coils=4, amplitude=0.05, color='k', linewidth=2):
        self.n_coils = n_coils      # Number of zigzag coils
        self.amplitude = amplitude  # Height of coils
        self.color = color
        self.linewidth = linewidth

    def first_draw(self, ax):
        self.line, = ax.plot([], [], color=self.color, lw=self.linewidth, animated=True)
        return self.line

    def set_data(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def update(self, i):
        # Get the endpoints at frame i
        x_start = self.x1[i]
        y_start = self.y1[i]
        x_end   = self.x2[i]
        y_end   = self.y2[i]

        # Generate zigzag spring shape
        num_points = self.n_coils * 2 + 2
        t = np.linspace(0, 1, num_points)
        base_x = (1 - t) * x_start + t * x_end
        base_y = (1 - t) * y_start + t * y_end

        # Unit direction and normal vectors
        dx = x_end - x_start
        dy = y_end - y_start
        length = np.hypot(dx, dy)
        if length == 0:
            direction = np.array([1, 0])
        else:
            direction = np.array([dx, dy]) / length
        normal = np.array([-direction[1], direction[0]])

        # Zigzag pattern
        zigs = np.concatenate(([0], np.tile([1, -1], self.n_coils), [0]))
        offsets = self.amplitude * zigs

        # Apply perpendicular offsets
        spring_x = base_x + offsets * normal[0]
        spring_y = base_y + offsets * normal[1]

        self.line.set_data(spring_x, spring_y)
        return self.line


#%%
#----------------------------------------------------------------------#
# Plotting & Animation
#----------------------------------------------------------------------#

# Create a figure and axis for the animation
fig, ax = plt.subplots()
plt.close()

# Set bounding limits (horizontal plot)
ax.set_xlim(-0.6, 5)
ax.set_ylim(-0.6, 0.6)
ax.set_aspect('equal')

# Get the position and angle of the 6 bodies
x1, y1, theta1 = sol.y[:3]          # Left piston rod 1
x2, y2, theta2 = sol.y[3:6]         # Left piston rod 2
x3, y3 = sol.y[6:8]                 # Left piston
theta_01 = np.zeros_like(x3)      
x4, y4 = sol.y[8:10]                # Right piston
x5, y5, theta3 = sol.y[10:13]       # Right piston rod 5
x6, y6, theta4 = sol.y[13:16]       # Right piston rod 6
theta_02 = np.zeros_like(x5)          

# Create the boxes for the pistons and rods
box1 = Box(L1, 0.01, 'b')
box2 = Box(L2, 0.01, 'r')
box3 = Box(PL, 0.3, 'g') # Piston 1
box4 = Box(PL, 0.3, 'g') #piston 2
box5 = Box(L3, 0.01, 'r')
box6 = Box(L4, 0.01, 'b')

box3.set_data(x3, y3, theta_01)
box4.set_data(x4, y4, theta_02)

box1.set_data(x1, y1, theta1)
box2.set_data(x2, y2, theta2)

box5.set_data(x5, y5, theta3)
box6.set_data(x6, y6, theta4)

# Create the spring
spring = Spring(n_coils=5, amplitude=0.05, color='black', linewidth=2)
spring.set_data(x3, y3, x4, y4)  # between pistons

boxes = [box3, box4, box1, box2, box5, box6] # Drawing pistons first so you can see rods above them

# Add a text annotation for time and q values
time_text = ax.text(0.75, 2.5, '', fontsize=12, color='black', ha='left', transform=ax.transAxes)
q_text_lines = [
    ax.text(0.75, 2.3 - 0.15 * i, '', fontsize=10, color='black', ha='left', transform=ax.transAxes)
    for i in range(8)
]
def init():
    ax.set_title("t=0.00 sec", fontsize=15)
    for box in boxes:
        box.first_draw(ax)
    spring_line = spring.first_draw(ax)
    patches = [box.patch for box in boxes]
    time_text.set_text('')
    for q_text in q_text_lines:
        q_text.set_text('')
    return patches + [time_text] + q_text_lines + [spring_line]


def animate(i):
    ''' Draw the i-th frame of the animation'''
    
    ax.set_title(f"t={sol.t[i]:.2f} sec", fontsize=15)
    
    for box in boxes:
        box.update(i)
    patches = [box.patch for box in boxes]

    spring_patch = spring.update(i)

    # Update the time text
    time_text.set_text(f"Time: {sol.t[i]:.2f} sec")
    
    # Update the q values text in a table format
    q_values = np.round(sol.y[:, i], 2)
    q_labels = ['x1', 'y1', 'θ1', 'x2', 'y2', 'θ2', 'x3', 'y3']
    for j, (label, value) in enumerate(zip(q_labels, q_values)):
        q_text_lines[j].set_text(f"{label}: {value}")
    
    return patches + [time_text] + q_text_lines + [spring_patch]


# Set the interval between frames
dt = sol.t[1] - sol.t[0]

# Create the animation
anim = FuncAnimation(fig, animate, frames=len(sol.t), init_func=init, blit=False, interval=1000*dt)

# Save the animation as a video file
# HTML(anim.to_html5_video())

# For plotting GIF
from matplotlib.animation import PillowWriter
anim.save("spring_piston.gif", writer=PillowWriter(fps=30))






