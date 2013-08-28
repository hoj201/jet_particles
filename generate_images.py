from jet_integrator_lib import *
import pickle
import matplotlib.pyplot as plt
import numpy as np

file = open('state.pydat','r')
state_hist = pickle.load(file)
file = open('tracer.pydat','r')
tracer_hist = pickle.load(file)
file = open('energy.pydat','r')
energy_hist = pickle.load(file)
file.close()

N_vert = 20
circle_verts = np.zeros( [ DIM , N_vert + 1 ] )
theta = np.linspace(0,2*np.pi, N_vert )
circle_verts[0,0:N_vert] = SIGMA*np.cos(theta)
circle_verts[1,0:N_vert] = SIGMA*np.sin(theta)
verts = np.empty_like(circle_verts)
units = np.ones( N_vert + 1)

fig = plt.figure()
time = linspace(0,1,energy_hist.size)
ax = fig.add_subplot(111)
ax.plot(time, energy_hist )
ax.set_xlabel( 'time' )
ax.set_ylabel( 'energy' )
fig.savefig('energy_plot.pdf')
fig.clf()

for k in range(0,N_timestep):
	tracer = tracer_hist[k]
	x,g1,g2,a,b,c = array_to_jet_data( state_hist[k] )
	fig = plt.figure()
	ax = fig.add_subplot(111,aspect='equal',autoscale_on=False, xlim=( -2,2 ) , ylim=( -2,2 ) )
	ax.plot( x[0] , x[1], 'bo', tracer[0],tracer[1],'ro' )
	for p in range( 0 , N_PART ):
		verts = np.dot(g1[:,:,p], circle_verts ) + np.outer(x[:,p],units)
		ax.plot(verts[0],verts[1],'b-')
	fname = './images/fig_'+str(k)+'.png'
	fig.savefig( fname )
	fig.clf()
print 'done'
