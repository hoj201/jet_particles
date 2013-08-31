from numpy import array, concatenate, exp, sqrt, ones, zeros, outer, empty, einsum, linspace
from numpy.random import rand, randn
from numpy.linalg import cond, solve
from scipy.integrate import odeint
N_PART = 5
DIM = 2
SIGMA = 1./sqrt(N_PART)

def get_kernel_tensors( x ):
	delta_x = empty( [DIM , N_PART , N_PART])
	for d in range(0,DIM):
		delta_x[d] = outer(x[d] ,ones( N_PART) ) - outer( ones( N_PART ),x[d])

	sq_mag = zeros( [N_PART , N_PART])
	for d in range(0,DIM):
		sq_mag = sq_mag + delta_x[d]**2

	delta = array([[1 , 0] , [0 , 1]])

	# Generate the Kernel tensors
	K = exp( - sq_mag / (2*SIGMA**2) )
	dK = (-1./(SIGMA**2)) * einsum('aij,ij->aij',delta_x,K)
	term1 = einsum('ab,ij->abij',delta,K)
	term2 = einsum('aij,bij->abij',delta_x, dK)
	d2K = (-1./SIGMA**2) *( term1 + term2)
	term1 = einsum('ab,cij->abcij',delta,dK)
	term2 = einsum('ac,bij->abcij',delta,dK)
	term3 = einsum('aij,bcij->abcij',delta_x,d2K)
	d3K = (-1./SIGMA**2) * (term1 + term2 + term3 )
	return K,dK,d2K,d3K

def state_decomp( state ):
	q = empty( [DIM , N_PART] )
	p = empty( [DIM , N_PART] )
	mu = empty( [DIM, DIM , N_PART] )
	begin = 0
	end = DIM*N_PART
	q = state[begin:end].reshape( [DIM , N_PART] )
	begin = end
	end = end + DIM*N_PART
	p = state[begin:end].reshape( [DIM , N_PART] )
	begin = end
	end = end + DIM * DIM * N_PART
	mu = state[ begin:end].reshape( [DIM , DIM , N_PART] )
	return q,p,mu

def state_reconstruction( q , p , mu ):
	state = empty( DIM*N_PART + DIM*N_PART + DIM*DIM*N_PART )
	begin = 0
	end = DIM*N_PART
	state[begin:end] = q.reshape( DIM*N_PART )
	begin = end
	end = end + DIM*N_PART
	state[begin:end] = p.reshape( DIM*N_PART )
	begin = end
	end = end + DIM*DIM*N_PART
	state[begin:end] = mu.reshape( DIM*DIM*N_PART )
	return state

def get_energy( state ):
	#something is wrong with this energy function
	q,p,mu = state_decomp( state )
	K,dK,d2K,d3K = get_kernel_tensors(q)
	term = 1*einsum('ai,aj,ij',p,p,K)
	term = term + 2* einsum('abi,aj,bij',mu,p,dK)
	term = term - 1* einsum('abi,agj,bgij',mu,mu,d2K)
	return 0.5*term 

def get_state_velocity( state , t ):
	q,p,mu = state_decomp( state )
	K,dK,d2K,d3K = get_kernel_tensors( q )
	dq = einsum( 'aj,ij->ai',p,K) + einsum( 'abj,bij',mu,dK )
	xi = einsum( 'aj,bij->abi',p,dK) - einsum('agj,bgij->abi',mu,d2K)
	theta = einsum( 'aj,bgij->abgi',p,d2K) - einsum('adj,bgdij->abgi',mu,d3K)
	dp = - einsum('bi,bai->ai',p,xi) - einsum( 'bgi,bgai->ai',mu,theta)
	dmu = einsum('agi,bgi->abi',mu,xi) - einsum('gbi,gai->abi',mu,xi)
	out = state_reconstruction( dq, dp, dmu)
	return out

q = rand(DIM,N_PART)
p = rand(DIM,N_PART)
mu = 0.01*rand(DIM,DIM, N_PART)

state = state_reconstruction( q , p , mu )

t_end = 100
time = linspace(0.0,2.0,t_end)
state_hist = odeint( get_state_velocity , state , time )
print state_hist.shape
energy = empty(t_end)
for t in range(0,t_end):
	energy[t] = get_energy( state_hist[t] )
print energy

