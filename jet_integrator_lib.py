from numpy import array, concatenate, exp, sqrt, ones, zeros, outer, empty, einsum, linspace
from numpy.random import rand, randn
from numpy.linalg import cond, solve
N_PART = 10
DIM = 2
SIGMA = 1./sqrt(N_PART)
N_timestep = 60
dt = 0.1/60

K = zeros( [N_PART , N_PART])
dK = zeros( [DIM , N_PART , N_PART] )
d2K = zeros( [DIM , DIM , N_PART , N_PART] )
d3K = zeros( [DIM , DIM , DIM , N_PART , N_PART] )
d4K = zeros( [DIM , DIM , DIM , DIM , N_PART , N_PART] )
d5K = zeros( [DIM , DIM , DIM , DIM , DIM , N_PART , N_PART] )

def set_kernel_tensors( x ):
	global K, dK, d2K, d3K, d4K, d5K
	delta_x = empty( [DIM , N_PART , N_PART])
	for d in range(0,DIM):
		delta_x[d] = outer(x[d] ,ones( N_PART) ) - outer( ones( N_PART ),x[d])

	sq_mag = zeros( [N_PART , N_PART])
	for d in range(0,DIM):
		sq_mag = sq_mag + delta_x[d]**2

	delta = array([[1 , 0] , [0 , 1]])

	# Generate the Kernel tensors
	K = exp( - sq_mag / (2*SIGMA**2) )
	dK = (-1./(SIGMA**2)) * einsum('jab,ab->jab',delta_x,K)
	d2K = (-1./(SIGMA**2)) * (einsum('jk,ab->jkab', delta , K) + einsum('jab,kab->jkab',delta_x,dK))
	d3K = (-1./(SIGMA**2))*(einsum('jk,lab->jklab',delta,dK) + einsum('jl,kab->jklab',delta,dK) + einsum('jab,klab->jklab',delta_x,d2K))
	d4K = (-1./(SIGMA**2))*(einsum('jk,lmab->jklmab',delta,d2K)+einsum('jl,kmab->jklmab',delta,d2K)+einsum('jm,klab->jklmab',delta,d2K)+einsum('jab,klmab->jklmab',delta_x,d3K))
	d5K = (-1./(SIGMA**2))*(einsum('jk,lmnab->jklmnab',delta,d3K)+einsum('jl,kmnab->jklmnab',delta,d3K)+einsum('jm,klnab->jklmnab',delta,d3K)+einsum('jn,klmab->jklmnab',delta,d3K) + einsum('jab,klmnab->jklmnab',delta_x,d4K))
	return K,dK,d2K,d3K,d4K,d5K

def array_to_jet_data( state ):
	#takes a tuble of numbers and returns jet data
	begin = 0
	end = DIM*N_PART
	x = state[begin:end].reshape([DIM,N_PART])
	
	begin = end
	end = end + DIM**2 * N_PART
	g1 = state[begin:end].reshape([DIM, DIM, N_PART])

	begin = end
	end = end + DIM**3 * N_PART
	g2 = state[begin:end].reshape([DIM,DIM,DIM,N_PART])

	begin = end
	end = end + DIM*N_PART
	a = state[begin:end].reshape([DIM,N_PART])
	
	begin = end
	end = end + DIM**2 * N_PART
	b = state[begin:end].reshape([DIM,DIM,N_PART])
	
	begin = end
	end = end + DIM**3 * N_PART
	c = state[begin:end].reshape([DIM,DIM,DIM,N_PART])
	return x, g1, g2, a , b, c

def jet_data_to_array( x , g1 , g2 , a , b , c):
	#takes a tuple of jet data and outputs a tuple of numbers.
	out = empty(2*(DIM + DIM**2 + DIM**3)*N_PART )
	begin = 0
	end = DIM*N_PART
	out[begin:end] = x.reshape(DIM*N_PART)
	
	begin = end
	end = end + DIM**2 * N_PART
	out[begin:end] = g1.reshape(DIM**2 * N_PART)

	begin = end
	end = end + DIM**3 * N_PART
	out[begin:end] = g2.reshape(DIM**3 * N_PART)

	begin = end
	end = end + DIM*N_PART
	out[begin:end] = a.reshape(DIM*N_PART)
	
	begin = end
	end = end + DIM**2 * N_PART
	out[begin:end] = b.reshape(DIM**2 * N_PART)
	
	begin = end
	end = end + DIM**3 * N_PART
	out[begin:end] = c.reshape(DIM**3 * N_PART)
	return out


def arabic_to_greek(x,a,b,c):
	#solves for xi,theta given x,a,b,c
	global K, dK, d2K, d3K, d4K, d5K

	#derive velcoties from the momenta
	dx = einsum('ib,ab->ia',a,K) + einsum('ijb,jab->ia',b,dK) + einsum('ijkb,jkab->ia',c,d2K)
	xi = einsum('ib,jab->ija',a,dK) + einsum('ikb,jkab->ija',b,d2K) + einsum('iklb,jklab->ija',c,d3K)
	theta = einsum('ib,jkab->ijka',a,d2K) + einsum('ilb,jklab->ijka',b,d3K) + einsum('ilmb,jklmab->ijka',c,d4K) 
	return dx,xi,theta

def greek_to_arabic(x,dx,xi,theta):
	dof = (DIM + DIM**2 + DIM**3)*N_PART
	Operator = empty( [dof,dof] )
	e_k = zeros(dof)
	g1 = rand(DIM,DIM,N_PART)
	g2 = rand(DIM,DIM,DIM,N_PART)
	store = jet_data_to_array(x,g1,g2,dx,xi,theta)
	config = store[0:dof]
	for k in range(0,dof):
		e_k[k] = 1.
		input = concatenate([config,e_k])
		xo,g1o,g2o,ao,bo,co = array_to_jet_data(input)
		dxo, xio, thetao = arabic_to_greek(xo,ao,bo,co)
		output = jet_data_to_array(xo,g1o,g2o,dxo,xio,thetao)
		Operator[:,k] = output[dof:2*dof]
		e_k[k] = 0
	print Operator
	print 'condition number = ' + str( cond(Operator))
	store = jet_data_to_array(x,g1,g2,dx,xi,theta)
	store = solve( Operator , store[dof:2*dof] )	
	x,g1,g2,a,b,c = array_to_jet_data( concatenate([config,store]) )
	return a,b,c

def get_velocity(x,g1,g2,a,b,c): 
	global K, dK, d2K, d3K, d4K, d5K
	delta_x = empty( [DIM , N_PART , N_PART])
	for d in range(0,DIM):
		delta_x[d] = outer(x[d] ,ones( N_PART) ) - outer( ones( N_PART ),x[d])

	sq_mag = zeros( [N_PART , N_PART])
	for d in range(0,DIM):
		sq_mag = sq_mag + delta_x[d]**2

	delta = array([[1 , 0] , [0 , 1]])
	dx,xi,theta = arabic_to_greek(x,a,b,c)
	chi = einsum('ib,jklab->ijkla',a,d3K) + einsum('imb,jklmab->ijkla',b,d4K) + einsum('imnb,jklmnab->ijkla',c,d5K)	

	#calculate group velocities
	dg1 = einsum('ika,kja->ija',xi,g1)
	dg2 = einsum('ila,ljka->ijka',xi,g2) + einsum('ila,ljka->ijka',g1,theta)

	#calculate the time derivative of the momenta from the velocity
	da = einsum('ja,jia->ia',-a,xi) + einsum('jka,jkia->ia',b,theta) - einsum('jkla,jikla->ia',c,chi)
	db = einsum('ika,jka->ija',b,xi) - einsum('kja,kia->ija',b,xi) - einsum('ikla,jkla->ija',c,theta) + einsum('klja,kila->ija',c,theta) + einsum('kjla,kila->ija',c,theta)
	dc = einsum('ija,ka->ijka',b,dx) + einsum('ika,ja->ijka',b,dx) + einsum('ilka,jla->ijka',c,xi) + einsum('ikla,jla->ijka',c,xi) - einsum('ljka,lia->ijka',c,xi)
	#print einsum('ijka,la->ijkla',c,dx) - einsum('ikla,ja->ijkla',c,dx)
	return dx, dg1, dg2, da, db, dc

def ode_func( state , t ):
	#a function to put into an ODE solver
	x,g1,g2,a,b,c = array_to_jet_data( state )
	dx,dg1,dg2,da,db,dc = get_velocity( x,g1,g2,a,b,c )
	return jet_data_to_array( dx,dg1,dg2,da,db,dc)

def evaluate_velocity( state , nodes ):
	# u = evaluate_velocity( state, nodes) outpute velocities at spatial loactions stored in nodes
	N_nodes = nodes.shape[1]
	x,g1,g2,a,b,c = array_to_jet_data( state )
	delta_x = empty( [DIM , N_PART , N_nodes])
	for d in range(0,DIM):
		delta_x[d] = outer( ones( N_PART ), nodes[d] ) - outer( x[d] ,ones( N_nodes) ) 

	sq_mag = zeros( [N_PART , N_nodes])
	for d in range(0,DIM):
		sq_mag = sq_mag + delta_x[d]**2

	delta = array([[1 , 0] , [0 , 1]])

	# Generate the Kernel tensors
	J = exp( - sq_mag / (2*SIGMA**2) )
	dJ = (-1./(SIGMA**2)) * einsum('jab,ab->jab',delta_x,J)
	d2J = (-1./(SIGMA**2)) * (einsum('jk,ab->jkab', delta , J) + einsum('jab,kab->jkab',delta_x,dJ))
	d3J = (-1./(SIGMA**2))*(einsum('jk,lab->jklab',delta,dJ) + einsum('jl,kab->jklab',delta,dJ) + einsum('jab,klab->jklab',delta_x,d2J))
	d4J = (-1./(SIGMA**2))*(einsum('jk,lmab->jklmab',delta,d2J)+einsum('jl,kmab->jklmab',delta,d2J)+einsum('jm,klab->jklmab',delta,d2J)+einsum('jab,klmab->jklmab',delta_x,d3J))
	d5J = (-1./(SIGMA**2))*(einsum('jk,lmnab->jklmnab',delta,d3J)+einsum('jl,kmnab->jklmnab',delta,d3J)+einsum('jm,klnab->jklmnab',delta,d3J)+einsum('jn,klmab->jklmnab',delta,d3J) + einsum('jab,klmnab->jklmnab',delta_x,d4J))
	u = einsum('ib,ba ->ia', a , J) + einsum('ijb,jba->ia',b,dJ) + einsum('ijkb,jkba->ia',c,d2J)
	return u

def get_energy( state ):
	x,g1,g2,a,b,c = array_to_jet_data( state )
	term = einsum('ia,ib,ab',a,a,K)
	term += einsum('ib,ija,jba',a,b,dK)
	term += einsum('ib,ijka,jkba',a,c,d2K)
	term -= einsum('ijb,ia,jba',b,a,dK)
	term -= einsum('ijb,ika,jkba',b,b,d2K)
	term -= einsum('ijb,ikla,jklba',b,c,d3K)
	term += einsum('ijkb,ia,jkba',c,a,d2K)
	term += einsum('ijkb,ila,jklba',c,b,d3K)
	term += einsum('ijkb,ilma,jklmba',c,c,d4K)
	return 0.5*term 
