#from scipy.spatial.distance import pdist , squareform
import matplotlib.pyplot as plt
import numpy as np

DIM = 2
N = 2
SIGMA = 1.0

def Hermite( k , x):
    #Calculate the 'statisticians' Hermite polynomials
    if k==0:
        return 1.
    elif k==1:
        return x
    elif k==2:
        return x**2 -1
    elif k==3:
        return x**3 - 3*x
    elif k==4:
        return x**4 - 6*x**2 + 3
    elif k==5:
        return x**5 - 10*x**3 + 15*x
    else:
        print 'error in Hermite function, unknown formula for k=' + str(k)

def derivatives_of_Gaussians( nodes , q ):
    #given x_i , x_j returns G(x_ij) for x_ij = x_i - x_j
    N_nodes = nodes.shape[0]
    dx = np.zeros([N_nodes,N,DIM])
    for i in range(0,N_nodes):
        for j in range(0,N):
            dx[i,j,:] = nodes[i,:] - q[j,:]

    rad_sq = np.einsum('ija,ija->ij',dx,dx)
    delta = np.eye(DIM)
    G = np.exp( -rad_sq / (2*SIGMA**2) ) 
    DG = np.zeros([ N_nodes , N, DIM ]) #indices 0 and 1 are particle-indices
    D2G = np.zeros([ N_nodes , N, DIM , DIM ]) #indices 0 and 1 are particle-indices
    D3G = np.zeros([ N_nodes , N, DIM , DIM , DIM ]) #indices 0 and 1 are particle-indices

    DG = np.einsum('ija,ij->ija',-dx/(SIGMA**2) , G )
    D2G = (1./SIGMA**2)*(np.einsum('ab,ij->ijab',-delta , G ) - np.einsum('ija,ijb->ijab',dx , DG ) )
    D3G = (1./SIGMA**2)*( np.einsum('ab,ijc->ijabc',-delta,DG) - np.einsum('ac,ijb->ijabc',delta,DG) - np.einsum('ija,ijcb->ijabc',dx,D2G) )

#    for a in range(0,DIM):
#        alpha = np.zeros(DIM)
#        alpha[a] = 1
#        DG[:,:,a] = (-1./SIGMA) * G * Hermite( 1 , dx[:,:,a] )
#        for b in range(0,DIM):
#            alpha[b] = alpha[b] + 1
#            store = np.ones([N_nodes,N])
#            for i in range(0,DIM):
#                store = store*Hermite( alpha[i] , dx[:,:,i] )
#            D2G[:,:,a,b] = (-1./SIGMA)**2 * G * store
#            for c in range(0,DIM):
#                alpha[c] = alpha[c] + 1
#                store = np.ones([N_nodes,N])
#                for i in range(0,DIM):
#                    store = store*Hermite(alpha[i] , dx[:,:,i])
#                D3G[:,:,a,b,c] = (-1./SIGMA)**3 * G * store
#                alpha[c] = alpha[c] - 1
#            alpha[b] = alpha[b] - 1
    return G , DG, D2G, D3G

def derivatives_of_kernel( nodes , q ):
    #given x_i and x_j the K = Kernel( x_ij) and derivatives with x_ij = x_i - x_j.
    N_nodes = nodes.shape[0]
    x = np.zeros([N_nodes , N , DIM ])
    r_sq = np.zeros([N_nodes, N])
    #The code is written such that we evaluate at the nodes, and entry (i,j) is the contribution at node i due to particle j.
    for i in range(0,N_nodes):
        for j in range(0,N):
            x[i,j,:] = nodes[i,:] - q[j,:]
            r_sq[i,j] = np.dot(x[i,j] , x[i,j])

    delta = np.identity( DIM )

    S = np.einsum('ija,ijb->ijab',x,x) / (SIGMA**2) \
        + np.einsum('ij,ab->ijab',np.ones([N_nodes,N]) - r_sq/(SIGMA**2) , delta )
    DS = ( np.einsum('ac,ijb->ijabc',delta,x) \
               + np.einsum('ija,bc->ijabc',x,delta) \
               - 2.*np.einsum('ijc,ab->ijabc',x,delta) ) / (SIGMA**2)
    D2S = ( np.einsum('ac,bd->abcd',delta,delta) \
                + np.einsum('ad,bc->abcd',delta,delta) \
                - 2.*np.einsum('ab,cd->abcd',delta,delta) ) / (SIGMA**2)

    G,DG,D2G,D3G = derivatives_of_Gaussians( nodes , q )
    K = np.einsum('ij,ijab->ijab',G,S)
    DK = np.einsum('ijc,ijab->ijabc',DG,S) + np.einsum('ij,ijabc->ijabc',G,DS)
    D2K = np.einsum('ijcd,ijab->ijabcd',D2G,S) \
        + np.einsum('ijc,ijabd->ijabcd',DG,DS) \
        + np.einsum('ijd,ijabc->ijabcd',DG,DS) \
        + np.einsum('ij,abcd->ijabcd',G,D2S)
    D3K = np.einsum('abcd,ije->ijabcde',D2S,DG) \
        + np.einsum('abec,ijd->ijabcde',D2S,DG) \
        + np.einsum('abde,ijc->ijabcde',D2S,DG) \
        + np.einsum('ijabc,ijde->ijabcde',DS,D2G) \
        + np.einsum('ijabd,ijec->ijabcde',DS,D2G) \
        + np.einsum('ijabe,ijcd->ijabcde',DS,D2G) \
        + np.einsum('ijab,ijcde->ijabcde',S,D3G)
    #EXAMPLE OF INDEX CONVENTION 'ijabc' refers to the c^th derivative of the ab^th entry of K(q_i - q_j)
    return K, DK, D2K, D3K

def Hamiltonian( q , p , mu ):
    #return the Hamiltonian.  Serves as a safety to check our equations of motion are correct.
    K,DK,D2K,D3K = derivatives_of_kernel(q,q)
    term_00 = 0.5*np.einsum('ia,ijab,jb',p,K,p)
    term_01 = - np.einsum('ia,ijabc,jbc',p,DK,mu)
    term_11 = - 0.5*np.einsum('iac,ijabcd,jbd',mu,D2K,mu)
    return term_00 + term_01 + term_11
    
def ode_function( state , t ):
    q , p , mu = state_to_weinstein_darboux( state )
    K,DK,D2K,D3K = derivatives_of_kernel( q , q )
    dq = np.einsum('ijab,jb->ia',K,p) - np.einsum('ijabc,jbc->ia',DK,mu)
    xi = np.einsum('ijacb,jc->iab',DK,p) - np.einsum('ijacbd,jcd->iab',D2K,mu)
    chi = np.einsum('ijadbc,jd->iabc',D2K,p) - np.einsum('ijaebcd,jed->iabc',D3K,mu)
    dp = - np.einsum('ib,jc,ijbca->ia',p,p,DK) \
        + np.einsum('id,jbc,ijdbca->ia',p,mu,D2K) \
        - np.einsum('jd,ibc,ijdbca->ia',p,mu,D2K) \
        + np.einsum('icb,jed,ijceabd->ia',mu,mu,D3K)
    dmu = np.einsum('iac,ibc->iab',mu,xi) - np.einsum('icb,ica->iab',mu,xi)
    dstate = weinstein_darboux_to_state( dq , dp , dmu )
    return dstate

def state_to_weinstein_darboux( state ):
    q = np.reshape( state[0:(N*DIM)] , [N,DIM] )
    p = np.reshape( state[(N*DIM):(2*N*DIM)] , [N,DIM] )
    mu = np.reshape( state[(2*N*DIM):(2*N*DIM + N*DIM*DIM)] , [N,DIM,DIM] )
    return q , p , mu

def weinstein_darboux_to_state( q , p , mu ):
    state = np.zeros( 2*N*DIM + N*DIM*DIM )
    state[0:(N*DIM)] = np.reshape( q , N*DIM )
    state[(N*DIM):(2*N*DIM)] = np.reshape( p , N*DIM )
    state[(2*N*DIM):(2*N*DIM+N*DIM*DIM)] = np.reshape( mu , N*DIM*DIM)
    return state

def display_velocity_field( q , p ,mu ):
 W = 5*SIGMA
 res = 30
 N_nodes = res**DIM
 store = np.outer( np.linspace(-W,W , res), np.ones(res) )
 nodes = np.zeros( [N_nodes , DIM] )
 nodes[:,0] = np.reshape( store , N_nodes )
 nodes[:,1] = np.reshape( store.T , N_nodes )
 K,DK,D2K,D3K = derivatives_of_kernel( nodes , q )
 vel_field = np.einsum('ijab,jb->ia',K,p) - np.einsum('ijabc,jbc->ia',DK,mu)
 U = vel_field[:,0]
 V = vel_field[:,1]

 plt.figure()
 plt.quiver( nodes[:,0] , nodes[:,1] , U , V )
 plt.plot(q[:,0],q[:,1],'ro')
 for i in range(0,N):
     plt.arrow(q[i,0], q[i,1], 0.2*p[i,0], 0.2*p[i,1], head_width=0.05, head_length=0.1, fc='b', ec='b')
 plt.axis([- W, W,- W, W ])
 return plt.gcf()

def test_functions( trials ):
    #checks that each function does what it is supposed to
    
    #testing derivatives of Gaussians
    h = 10e-7
    q = SIGMA*np.random.randn(N,DIM)
    p = SIGMA*np.random.randn(N,DIM)
    mu = np.random.randn(N,DIM,DIM)
    G,DG,D2G,D3G = derivatives_of_Gaussians(q,q)
    q_a = np.copy(q)
    q_b = np.copy(q)
    q_c = np.copy(q)

    for i in range(0,N):
        for a in range(0,DIM):
            error_max = 0.
            q_a[i,a] = q[i,a]+h
            G_a , DG_a , D2G_a , D3G_a = derivatives_of_Gaussians(q_a, q) 
            for j in range(0,N):
                error = (G_a[i,j] - G[i,j])/h - DG[i,j,a]
                error_max = np.maximum( np.absolute( error ) , error_max )
            print 'max error for DG was ' + str( error_max )
            error_max = 0.
            for b in range(0,DIM):
                q_b[i,b] = q[i,b] + h
                G_b , DG_b , D2G_b , D3G_b = derivatives_of_Gaussians( q_b , q ) 
                for j in range(0,N):
                    error = (DG_b[i,j,a] - DG[i,j,a])/h - D2G[i,j,a,b]
                    error_max = np.maximum( np.absolute( error ) , error_max )
                print 'max error for D2G was ' + str( error_max )
                error_max = 0.
                for c in range(0,DIM):
                    q_c[i,c] = q_c[i,c] + h
                    G_c , DG_c , D2G_c , D3G_c = derivatives_of_Gaussians( q_c , q ) 
                    for j in range(0,N):
                        error = (D2G_c[i,j,a,b] - D2G[i,j,a,b])/h - D3G[i,j,a,b,c]
                        error_max = np.maximum( np.absolute(error) , error_max )
                    print 'max error for D3G was ' + str( error_max )
                    q_c[i,c] = q_c[i,c] - h
                q_b[i,b] = q_b[i,b] - h
            q_a[i,a] = q_a[i,a] - h
    

    K,DK,D2K,D3K = derivatives_of_kernel(q,q)
    delta = np.identity(DIM)
    error_max = 0.
    for i in range(0,N):
        for j in range(0,N):
            x = q[i,:] - q[j,:]
            r_sq = np.inner( x , x )
            for a in range(0,DIM):
                for b in range(0,DIM):
                    G = np.exp( -r_sq / (2.*SIGMA**2) )
                    K_ij_ab = ( (x[a]*x[b])/(SIGMA**2) + (1. - r_sq/(SIGMA**2) )*delta[a,b] )*G
                    error = K_ij_ab - K[i,j,a,b]
                    error_max = np.maximum( np.absolute(error) , error_max )

    print 'error_max for K = ' + str( error_max )
    if (error_max > 100*h):
        print 'WARNING:  COMPUTATION OF K APPEARS TO BE INACCURATE'

    error_max = 0.
    for i in range(0,N):
        for a in range(0,DIM):
            q_a[i,a] = q[i,a] + h
            K_a,DK_a,D2K_a,D3K_a = derivatives_of_kernel(q_a,q)
            for j in range(0,N):
                der = ( K_a[i,j,:,:] - K[i,j,:,:] ) / h
                error = np.linalg.norm(  der - DK[i,j,:,:,a] )
                error_max = np.maximum(error, error_max)
            q_a[i,a] = q[i,a]
    print 'error_max for DK = ' + str( error_max )
    if (error_max > 100*h):
        print 'WARNING:  COMPUTATION OF DK APPEARS TO BE INACCURATE'

    error_max = 0.
    q_b = np.copy(q)
    for i in range(0,N):
        for a in range(0,DIM):
            for b in range(0,DIM):
                q_b[i,b] = q[i,b] + h
                K_b,DK_b,D2K_b,D3K_b = derivatives_of_kernel(q_b,q)
                for j in range(0,N):
                    der = (DK_b[i,j,:,:,a] - DK[i,j,:,:,a] )/h
                    error = np.linalg.norm( der - D2K[i,j,:,:,a,b] )
                    error_max = np.maximum( error, error_max )
                q_b[i,b] = q[i,b]

    print 'error_max for D2K = ' + str( error_max )
    if (error_max > 100*h):
        print 'WARNING:  COMPUTATION OF D2K APPEARS TO BE INACCURATE'

    error_max = 0.
    q_c = np.copy(q)
    for i in range(0,N):
        for a in range(0,DIM):
            for b in range(0,DIM):
                for c in range(0,DIM):
                    q_c[i,c] = q[i,c] + h
                    K_c,DK_c,D2K_c,D3K_c = derivatives_of_kernel(q_c,q)
                    for j in range(0,N):
                        der = (D2K_c[i,j,:,:,a,b] - D2K[i,j,:,:,a,b] )/h
                        error = np.linalg.norm( der - D3K[i,j,:,:,a,b,c] )
                        error_max = np.maximum( error, error_max )
                    q_c[i,c] = q[i,c]

    print 'error_max for D3K = ' + str( error_max )

    if (error_max > 100*h):
        print 'WARNING:  COMPUTATION OF D3K APPEARS TO BE INACCURATE'

    print 'TESTING SYMMETRIES'
    print 'Is K symmetric with respect to ij?'
    error_max = 0
    for i in range(0,N):
        for j in range(0,N):
            error = np.linalg.norm( K[i,j,:,:] - K[j,i,:,:] )
            error_max = np.maximum( error, error_max )
    print 'max for K_ij - K_ji = ' + str( error_max )

    print 'Is DK anti-symmetric with respect to ij?'
    error_max = 0
    for i in range(0,N):
        for j in range(0,N):
            for a in range(0,DIM):
                error = np.linalg.norm( DK[i,j,:,:,a] + DK[j,i,:,:,a] )
                error_max = np.maximum( error, error_max )
    print 'max for DK_ij + DK_ji = ' + str( error_max )



    s = weinstein_darboux_to_state( q , p , mu)
    ds = ode_function( s , 0 )
    dq,dp_coded,dmu = state_to_weinstein_darboux( ds ) 

    print 'a test of the ode:'
    print 'dp_coded =' + str(dp_coded)

    Q = np.copy(q)
    dp_estim = np.zeros([N,DIM])
    for i in range(0,N):
        for a in range(0,DIM):
            Q[i,a] = q[i,a] + h
            dp_estim[i,a] = - ( Hamiltonian(Q,p,mu) - Hamiltonian(q,p,mu) ) / h 
            Q[i,a] = Q[i,a] - h
    print 'dp_estim =' + str(dp_estim)
    print 'dp_error =' + str(dp_estim - dp_coded)

    return 'what do you think?'
