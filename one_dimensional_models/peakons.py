import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.collections import PolyCollection
#from matplotlib.colors import colorConverter

N = 2

def greens_functions(q):
#produces the greens
	delta_x = np.outer(q , np.ones(N) ) - np.outer( np.ones(N),q)
	sq_mag =  delta_x**2	
	G = np.exp( - np.abs( delta_x) )
	dG = -np.sign( delta_x ) * G
	for k in range(0,N):
		dG[k,k] = 0.
	return G,dG

def assemble_state(q,p):
	state = np.zeros( 2*N )
	state[0:N] = q
	state[N:2*N] = p
	return state

def decompose_state( state ):
	q = state[0:N]
	p = state[N:2*N]
	return q,p

def get_velocity(state):
	q,p = decompose_state(state)
	G,dG = greens_functions(q)
	dq = np.inner(G,p) 
	dp = -0.5*p*np.inner(dG, p)
	return assemble_state(dq,dp)

def velocity_field( state , x ):
	q,p = decompose_state( state )
	store = 0.0
	for k in range(0,N):
		store = store + p[k]*np.exp( -np.abs(x - q[k]) )
	return store

def energy( state ):
	q,p= decompose_state( state )
	G,dG = greens_functions( q )
	return 0.5 * np.inner( p , np.inner( G , p ) ) 
'''
q0 = np.linspace(-2,2,N)
p0 = 1.0*np.ones(N)
q1 = 1.0*np.ones(N)
p1 = 0.0*np.ones(N)
'''

q = np.array( [ -2.0 , 2.0] )
p = np.array( [ 1.0 ,-1.0] )

'''
q0 = np.array( [0.0] )
p0 = np.array( [1.0] )
q1 = np.array( [1.0] )
p1 = np.array( [0.0] )
'''
state = assemble_state( q, p )
N_time_steps = 2000
dt = 0.01
state_hist = np.zeros( [ N_time_steps , 2*N ] )

print energy( state )

for t_index in range(0,N_time_steps):
	state_hist[t_index] = state
	k1 = get_velocity( state )
	k2 = get_velocity( state + 0.5*k1*dt )
	k3 = get_velocity( state + 0.5*k2*dt )
	k4 = get_velocity( state + k3*dt )
	state = state + dt*( k1 + 2*k2 + 2*k3 + k4)/6.
print energy( state )
fig,ax = plt.subplots()
x = np.linspace(-4,4,200)
for t_index in range(0,N_time_steps,100):
	y = 2.0*velocity_field( state_hist[t_index] , x )
	ax.plot(x,y+dt*t_index,'k-')
plt.xlabel('x')
plt.ylabel('t')
plt.show()
