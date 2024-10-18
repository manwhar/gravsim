import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.animation import FuncAnimation
import random

G = 6.6743e-11

def rand_hex():
    r = lambda: random.randint(0,255)
    return '#%02X%02X%02X' % (r(),r(),r())

class Position:
    def __init__(self, x, y, z):
        # unit: m
        self.x = x
        self.y = y
        self.z = z


class Velocity:
    def __init__(self, vx=0, vy=0, vz=0):
        # unit: m/s
        self.vx = vx
        self.vy = vy
        self.vz = vz


class Force:
    def __init__(self, fx=0, fy=0, fz=0):
        # unit: N
        self.fx = fx
        self.fy = fy
        self.fz = fz
    
    def __add__(self, other):
        if type(other) != Force:
            raise ValueError("other must be a Force object")
        return Force(self.fx + other.fx, self.fy + other.fy, self.fz + other.fz)


class SpaceRock:
    def __init__(self, pos, vel=None, mass=10, data=None, id="", size=1, color=None):
        # size is purely cosmetic and does not affect the simulation
        # size is also not to scale with real life
        if type(pos) == tuple:
            pos = Position(*pos)
        elif type(pos) != Position:
            raise ValueError("pos must be a tuple or Position object")

        if type(vel) == tuple:
            vel = Velocity(*vel)
        elif (type(vel) != Velocity) and (type(vel) != type(None)):
            raise ValueError("vel must be a tuple or Velocity object; found type: " + str(type(vel)))
        
        self.pos = pos
        self.vel = vel if vel is not None else Velocity()
        self.data = data if data is not None else []
        self.id = id
        self.size = size
        self.color = rand_hex() if color is None else color

        # mass unit: kg
        self.mass = mass

    def calculate_grav_force(self, other):
        if type(other) != SpaceRock:
            raise ValueError("other must be a SpaceRock object")

        # 3 dimensional distance calculation
        r = np.sqrt((self.pos.x - other.pos.x)**2 + (self.pos.y - other.pos.y)**2 + (self.pos.z - other.pos.z)**2)
        
        # gravitational force calculation based on Newton's law of universal gravitation
        f = (G * self.mass * other.mass) / r**2
        
        fx = f * (other.pos.x - self.pos.x) / r
        fy = f * (other.pos.y - self.pos.y) / r
        fz = f * (other.pos.z - self.pos.z) / r
        return Force(fx, fy, fz)
    
    def update_position(self, force, dt):
        # dt unit: s
        # force unit: N
        # velocity unit: m/s

        # update velocity
        self.vel.vx += (force.fx * dt) / self.mass
        self.vel.vy += (force.fy * dt) / self.mass
        self.vel.vz += (force.fz * dt) / self.mass

        # update position
        self.pos.x += self.vel.vx * dt
        self.pos.y += self.vel.vy * dt
        self.pos.z += self.vel.vz * dt


# Generate data for the animation
def generate_data(space_rocks, steps=1000, dt=1, output_every=1):
    # output_every minus 1 is the number of steps to skip between data points in the output
    # we still calculate the ones in between for precision since this isn't calculus-based
    for step in range(steps):
        for rock in space_rocks:
            force = Force()
            for other in space_rocks:
                if rock != other:
                    force += rock.calculate_grav_force(other)
            rock.update_position(force, dt)
            if step % output_every == 0:
                rock.data.append((rock.pos.x, rock.pos.y, rock.pos.z))

def generate_stars(count=5000, minimum_distance=1e20, maximum_distance=1e25):
    stars_x = []
    stars_y = []
    stars_z = []
    for _ in range(count):
        distance = random.uniform(minimum_distance, maximum_distance)
        angle1 = random.uniform(0, 2 * np.pi)
        angle2 = random.uniform(0, np.pi)
        x = distance * np.sin(angle2) * np.cos(angle1)
        y = distance * np.sin(angle2) * np.sin(angle1)
        z = distance * np.cos(angle2)
        stars_x.append(x)
        stars_y.append(y)
        stars_z.append(z)
    return (stars_x, stars_y, stars_z)

SUN_START_POS = (0, 0, 0)
SUN_MASS = 1.989e30

EARTH_START_POS = (1.498e11, 0, 0)
EARTH_START_VEL = (0, 2.9784e4, 0)
EARTH_MASS = 5.972e24

MOON_MASS = 7.34767309e22
MOON_DIST_FROM_EARTH = 3.84e8
MOON_VEL_TO_EARTH = 1022

# New user-defined constants
SIMULATION_TIME = 615360000  # total length of time simulated in seconds (e.g., one year)
RENDER_TIME = 10   # total length of the outputted render in seconds
FPS = 30  # frames per second of the animation
TIME_STEP = 360000  # seconds; dictates accuracy of simulation

# Dependent variables based on user input
TOTAL_STEPS = int(SIMULATION_TIME / TIME_STEP)  # total number of simulation steps
FRAMES_COUNT = int(FPS * RENDER_TIME)  # total number of frames in the animation
SKIP_IN_RENDER = max(1, int(TOTAL_STEPS / FRAMES_COUNT))  # skip rendering every n frames

SPACE_MODE = False  # whether to include stars in the background of the animation to make it less confusing

obj1 = SpaceRock(
    pos=(0, 0, 0),
    mass=300*SUN_MASS,
    vel=(0, 0, 100000),
    size=4,
    color='orange'
)

obj2 = SpaceRock(
    pos=(1.5e11, 0, 0),
    mass=200*SUN_MASS,
    vel=(100000, 0, -100000),
    size=4,
    color='red'
)

obj3 = SpaceRock(
    pos=(0, -5e11, 0),
    mass=SUN_MASS,
    vel=(-100000, 100000, 900),
    size=4,
    color='blue'
)

sun = SpaceRock(
    pos=SUN_START_POS, 
    mass=SUN_MASS, 
    size=10,
    color="#ffcc54",
    id="sun")

earth = SpaceRock(
    pos=EARTH_START_POS, 
    vel=EARTH_START_VEL, 
    mass=EARTH_MASS,
    size=5,
    color="#54acff",
    id="earth")

moon = SpaceRock(
    pos=(EARTH_START_POS[0] + MOON_DIST_FROM_EARTH, 0, 0), 
    vel=(0, EARTH_START_VEL[1] + MOON_VEL_TO_EARTH, 0), 
    mass=MOON_MASS, 
    size=2,
    color="#9e9e9e",
    id="moon")

mercury = SpaceRock(
    pos=(5.791e10, 0, 0),
    mass=3.285e23,
    vel=(0, 4.787e4, 0),
    size=3,
    color="#ff5454",
    id="mercury")

mars = SpaceRock(
    pos=(2.2794e11, 0, 0),
    mass=6.39e23,
    vel=(0, 2.4077e4, 0),
    size=3,
    color="#ff5454",
    id="mars")

venus = SpaceRock(
    pos=(1.08e11, 0, 0),
    mass=4.867e24,
    vel=(0, 3.502e4, 0),
    size=4,
    color="#ff8f54",
    id="venus")

saturn = SpaceRock(
    pos=(1.429e12, 0, 0),
    mass=5.683e26,
    vel=(0, 9.69e3, 0),
    size=7,
    color="#ff54ff",
    id="saturn")

jupiter = SpaceRock(
    pos=(7.785e11, 0, 0),
    mass=1.898e27,
    vel=(0, 1.307e4, 0),
    size=6,
    color="#54ff54",
    id="jupiter")

uranus = SpaceRock(
    pos=(2.871e12, 0, 0),
    mass=8.681e25,
    vel=(0, 6.8e3, 0),
    size=5,
    color="#54ffff",
    id="uranus")

neptune = SpaceRock(
    pos=(4.498e12, 0, 0),
    mass=1.024e26,
    vel=(0, 5.43e3, 0),
    size=5,
    color="#5454ff",
    id="neptune")


# space_rocks = [obj1, obj2]
space_rocks = [earth, moon, mars, neptune, jupiter, uranus, saturn, mercury, venus, sun]


# center can be coordinates or a SpaceRock object, in which case it will follow that object
# SIM_CENTER = EARTH_START_POS  
SIM_CENTER = earth
SIM_SIZE = 5e11 # side lengths of cube of rendered space


generate_data(space_rocks, steps=TOTAL_STEPS, dt=TIME_STEP, output_every=SKIP_IN_RENDER)  # generate data for the animation
fig = plt.figure()  # setup figure
ax = fig.add_subplot(111, projection='3d')  # setup 3d axis
ax.legend([space_rocks[i].id for i in range(len(space_rocks))])
if SPACE_MODE:
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    stars_x, stars_y, stars_z = generate_stars()
    ax.scatter(stars_x, stars_y, stars_z, s=0.1, color='white')
    ax.grid(False)
    ax.xaxis.pane.set_alpha(0)  # make the graph borders transparent
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
if type(SIM_CENTER) == tuple:
    ax.set_xlim([SIM_CENTER[0] - SIM_SIZE/2, SIM_CENTER[0] + SIM_SIZE/2])  # set sim bounds
    ax.set_ylim([SIM_CENTER[1] - SIM_SIZE/2, SIM_CENTER[1] + SIM_SIZE/2])
    ax.set_zlim([SIM_CENTER[2] - SIM_SIZE/2, SIM_CENTER[2] + SIM_SIZE/2])
elif type(SIM_CENTER) == SpaceRock:
    ax.set_xlim([SIM_CENTER.pos.x - SIM_SIZE/2, SIM_CENTER.pos.x + SIM_SIZE/2])
    ax.set_ylim([SIM_CENTER.pos.y - SIM_SIZE/2, SIM_CENTER.pos.y + SIM_SIZE/2])
    ax.set_zlim([SIM_CENTER.pos.z - SIM_SIZE/2, SIM_CENTER.pos.z + SIM_SIZE/2])
else:
    raise ValueError("SIM_CENTER must be a tuple or SpaceRock object")
data = [list(zip(*rock.data)) for rock in space_rocks]  # extract data for animation for all space rocks
points = [ax.plot([], [], [], marker='o', markersize=space_rocks[i].size, color=space_rocks[i].color)[0] for i in range(len(space_rocks))]  # create plot placeholders for each space rock
lines = [Line3D([], [], [], color=space_rocks[i].color, alpha=0.5) for i in range(len(space_rocks))]
for line in lines:
    ax.add_line(line)

# Initialization function for animation
def init():
    for point, line in zip(points, lines):
        point.set_data([], [])
        point.set_3d_properties([])
        line.set_data([], [])
        line.set_3d_properties([])
    return points + lines


# Animation update function
camera_target_index = space_rocks.index(SIM_CENTER) if type(SIM_CENTER) == SpaceRock else None
def animate(i):
    for j, (point, line) in enumerate(zip(points, lines)):
        x, y, z = data[j]
        point.set_data([x[i]], [y[i]])
        point.set_3d_properties([z[i]])
        line.set_data(x[:i+1], y[:i+1])
        line.set_3d_properties(z[:i+1])
    
    if type(SIM_CENTER) == SpaceRock:
        # Update the view limits to lock the camera onto the selected planet
        target_x, target_y, target_z = data[camera_target_index][0][i], data[camera_target_index][1][i], data[camera_target_index][2][i]
        ax.set_xlim([target_x - SIM_SIZE / 2, target_x + SIM_SIZE / 2])
        ax.set_ylim([target_y - SIM_SIZE / 2, target_y + SIM_SIZE / 2])
        ax.set_zlim([target_z - SIM_SIZE / 2, target_z + SIM_SIZE / 2])
    
    return points + lines



# Create animation
ani = FuncAnimation(fig, animate, init_func=init, frames=len(data[0][0]), interval=20, repeat=False)

# Show plot
plt.show()