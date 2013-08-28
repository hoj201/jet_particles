from numpy import array, concatenate, exp, sqrt, ones, zeros, outer, empty, einsum, linspace
from numpy.random import randn
from scipy.integrate import odeint 
from numpy.linalg import cond, solve
import time,pickle
from jet_integrator_lib import *

# Initial Configuration
x = 1.0*randn( DIM , N_PART )
g1 = zeros([ DIM , DIM , N_PART ])
for d in range(0,DIM):
	g1[d,d,:] = ones( N_PART )
g2 = zeros([ DIM , DIM , DIM , N_PART ])
K,dK,d2K,d3K,d4K,d5K = set_kernel_tensors(x)

# Initial Momenta
#dx = zeros([ DIM , N_PART ])
#xi = zeros([ DIM , DIM , N_PART ])
#theta = zeros([DIM , DIM , DIM , N_PART])
#theta[0, 0, 0, 0] = 1.0
#a,b,c = greek_to_arabic(x, dx, xi, theta)

a = 1.0*randn( DIM , N_PART )
b = 0.10*randn( DIM , DIM , N_PART )
c = 0.00*randn( DIM , DIM , DIM , N_PART )

#c[0,0,0,0] = 1.0
#a[0,0] = 1./SIGMA**2

state = jet_data_to_array(x,g1,g2,a,b,c)
state_hist = empty( [N_timestep, 2*(DIM + DIM**2 + DIM**3)*N_PART])
energy_hist = empty( N_timestep )
N_tracer = 20**DIM
tracer = empty( [ DIM, N_tracer] )
tracer_hist = empty( [ N_timestep, DIM , N_tracer ] )
units = ones( 20 )
tracer[0] = outer( linspace(-2.,2.,20) , units).reshape(N_tracer)
tracer[1] = outer( units , linspace(-2.,2.,20) ).reshape(N_tracer)

t0 = time.clock()
for step in range(0, N_timestep ):	
	K,dK,d2K,d3K,d4K,d5K = set_kernel_tensors(x)
	state_hist[step] = state
	tracer_hist[step] = tracer
	energy_hist[step] = get_energy( state )
	state = odeint( ode_func,state, array([0.,dt]))[1]
	tracer = tracer + dt*evaluate_velocity( state, tracer)
	x = state[0:DIM*N_PART].reshape([DIM,N_PART])
t1 = time.clock()
print energy_hist
print 'integration complete'
file = open('state.pydat','w')
pickle.dump(state_hist, file )
file = open('tracer.pydat','w')
pickle.dump(tracer_hist,file )
file = open('energy.pydat','w')
pickle.dump(energy_hist, file )
file.close()

