import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter

N = 2

def greens_functions(q0):
	#produces the greens
	delta_x = np.outer(q0 , np.ones(N) ) - np.outer( np.ones(N),q0)
	sq_mag =  delta_x**2
	
	G = np.exp( - sq_mag / 2 )
	dG =  - delta_x * G
	d2G = (sq_mag - 1)*G
	d3G = (3*delta_x - delta_x**3)*G
	'''
	G = np.exp( - np.abs( delta_x) )
	dG = -np.sign( delta_x ) * G
	d2G = -G
	d3G = np.sign( delta_x ) * G
	'''
	return G,dG,d2G,d3G

def assemble_state(q0,p0,q1,p1):
	state = np.zeros( 4*N )
	state[0:N] = q0
	state[N:2*N] = p0
	state[2*N:3*N] = q1
	state[3*N:4*N] = p1
	return state

def decompose_state( state ):
	q0 = state[0:N]
	p0 = state[N:2*N]
	q1 = state[2*N:3*N]
	p1 = state[3*N:4*N]
	return q0,p0,q1,p1

def get_velocity(state):
	q0,p0,q1,p1 = decompose_state( state )
	G,dG,d2G,d3G = greens_functions( q0 )
	dq0 = np.inner(G,p0) - np.inner(dG , p1*q1) 
	dq1 = q1*np.inner(dG,p0) - q1*np.inner( d2G, p1*q1 )
	dp0 = -0.5*p0*np.inner(dG, p0) + p0*np.inner( d2G , p1*q1) + 0.5*p1*q1*np.inner( d3G,p1*q1)
	dp1 = -p1*np.inner( dG , p0) + p1*np.inner( d2G , p1*q1)
	return assemble_state(dq0,dp0,dq1,dp1)

def energy( state ):
	q0,p0,q1,p1 = decompose_state( state )
	G,dG,d2G,d3G = greens_functions( q0 )
	term1 = 0.5 * np.inner( p0 , np.inner( G , p0 ) ) 
	term2 = -np.inner( p0 , np.inner( dG , p1*q1 ) )
	term3 = -0.5 * np.inner( p1*q1 , np.inner( d2G , p1*q1 ) )
	return term1 + term2 + term3
'''
q0 = np.linspace(-2,2,N)
p0 = 1.0*np.ones(N)
q1 = 1.0*np.ones(N)
p1 = 0.0*np.ones(N)
'''
q0 = np.array( [ -2.0 , 2.0] )
p0 = np.array( [ 1.0 , -1.0] )
q1 = np.array( [ 1.0 , 1.0 ] )
p1 = np.array( [ 0.0 , 0.0 ] )

state = assemble_state( q0, p0, q1, p1)
N_time_steps = 20
dt = 0.05
state_hist = np.zeros( [ N_time_steps , 4*N ] )

print energy( state )

for t_index in range(0,N_time_steps):
	state_hist[t_index] = state
	k1 = get_velocity( state )
	k2 = get_velocity( state + 0.5*k1*dt )
	k3 = get_velocity( state + 0.5*k2*dt )
	k4 = get_velocity( state + k3*dt )
	state = state + dt*( k1 + 2*k2 + 2*k3 + k4)/6.
print energy( state )
fig = plt.figure()
ax = fig.gca(projection='3d')
cc = lambda arg: colorConverter.to_rgba(arg, alpha = 0.6)
xs = np.linspace(-3.0,3.0,50)
verts = []
zs = np.linspace(0,dt*(N_time_steps-1),N_time_steps)
for z in zs:
	ys = np.exp( 0.1*xs )
	ys[0] = 0.0
	ys[-1] = 0.0
	verts.append(list(zip(xs,ys)))

poly = PolyCollection(verts, facecolors = [cc('w')])
poly.set_alpha(0.3)
ax.add_collection3d( poly , zs=zs , zdir = 'y')
ax.set_xlabel('X')
ax.set_xlim3d(-3,3)
ax.set_ylabel('time')
ax.set_ylim3d(0,1)
ax.set_zlabel('u')
ax.set_zlim3d(0,1)
plt.show()