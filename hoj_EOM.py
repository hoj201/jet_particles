#from scipy.spatial.distance import pdist , squareform
#import matplotlib.pyplot as plt
import numpy as np

DIM = 2
N = 2
SIGMA = 1.0

def Gaussian_monomial( x , n ):
# computes x/sigma^n * G(x)
    y = x / SIGMA
    store = y * np.exp( -(0.5/n) * y**2 )
    return store**n

def diff_1D_Gaussian( x , k ):
# returns the kth derivative of a 1 dimensional Guassian
    G = np.exp( -0.5 * (x / SIGMA)**2 )
    if k == 0:
        return G
    elif k==1:
        return -1.*Gaussian_monomial(x,1)
    elif k==2:
        return Gaussian_monomial(x,2) - G
    elif k==3:
        return -1.*( Gaussian_monomial(x,3) - 3.*Gaussian_monomial(x,1))
    elif k==4:
        return Gaussian_monomial(x,4) - 6.*Gaussian_monomial(x,2) + 3.*G
    elif k==5:
        return -1.*(Gaussian_monomial(x,5) - 10.*Gaussian_monomial(x,3) + 15.*Gaussian_monomial(x,1) )
    elif k==6:
        return Gaussian_monomial(x,6) - 15.*Gaussian_monomial(x,4) + 45.*Gaussian_monomial(x,2) -15.*G
    else:
        print 'error in diff_1D_Guassian:  k='+str(k)
        return 'error'

def derivatives_of_Gaussians( nodes , q ):
    N_nodes = nodes.shape[0]
    r_sq = np.zeros( [ N_nodes , N ] )
    dx = np.zeros( [N_nodes,N,DIM] )
    for a in range(0,DIM):
        dx[:,:,a] = np.outer( nodes[:,a] , np.ones(N) ) - np.outer( np.ones( N_nodes ), q[:,a] )
        r_sq[:,:] = dx[:,:,a]**2 + r_sq[:,:]
    G = np.exp( - r_sq / (2.*SIGMA**2) )
    DG = np.ones( [N_nodes,N,DIM] )
    D2G = np.ones( [N_nodes,N,DIM,DIM] )
    D3G = np.ones( [N_nodes,N,DIM,DIM,DIM] )
    D4G = np.ones( [N_nodes,N,DIM,DIM,DIM,DIM] )
    D5G = np.ones( [N_nodes,N,DIM,DIM,DIM,DIM,DIM] )
    alpha = np.int_(np.zeros(DIM))
    #one derivative
    for a in range(0,DIM):
        alpha[a] = 1
        for b in range(0,DIM):
            DG[:,:,a] = DG[:,:,a]*diff_1D_Gaussian( dx[:,:,b] , alpha[b] )
        alpha[a] = 0
    
    #two derivatives
    for a in range(0,DIM):
        alpha[a] = 1
        for b in range(0,DIM):
            alpha[b] = alpha[b] + 1
            for c in range(0,DIM):
                D2G[:,:,a,b] = D2G[:,:,a,b]*diff_1D_Gaussian( dx[:,:,c] , alpha[c] )
            alpha[b] = alpha[b] - 1
        alpha[a] = 0

    #three derivatives
    for a in range(0,DIM):
        alpha[a] = 1
        for b in range(0,DIM):
            alpha[b] = alpha[b] + 1
            for c in range(0,DIM):
                alpha[c] = alpha[c] + 1
                for d in range(0,DIM):
                    D3G[:,:,a,b,c] = D3G[:,:,a,b,c]*diff_1D_Gaussian( dx[:,:,d] , alpha[d] )
                alpha[c] = alpha[c] - 1
            alpha[b] = alpha[b] - 1
        alpha[a] = 0

    #four derivatives
    for a in range(0,DIM):
        alpha[a] = 1
        for b in range(0,DIM):
            alpha[b] = alpha[b] + 1
            for c in range(0,DIM):
                alpha[c] = alpha[c] + 1
                for d in range(0,DIM):
                    alpha[d] = alpha[d] + 1
                    for e in range(0,DIM):
                        D4G[:,:,a,b,c,d] = D4G[:,:,a,b,c,d]*diff_1D_Gaussian( dx[:,:,e] , alpha[e] )
                    alpha[d] = alpha[d] - 1
                alpha[c] = alpha[c] - 1
            alpha[b] = alpha[b] - 1
        alpha[a] = 0

    #five derivatives
    for a in range(0,DIM):
        alpha[a] = 1
        for b in range(0,DIM):
            alpha[b] = alpha[b] + 1
            for c in range(0,DIM):
                alpha[c] = alpha[c] + 1
                for d in range(0,DIM):
                    alpha[d] = alpha[d] + 1
                    for e in range(0,DIM):
                        alpha[e] = alpha[e] + 1
                        for f in range(0,DIM):
                            D5G[:,:,a,b,c,d,e] = D5G[:,:,a,b,c,d,e]*diff_1D_Gaussian( dx[:,:,f] , alpha[f] )
                        alpha[e] = alpha[e] - 1
                    alpha[d] = alpha[d] - 1
                alpha[c] = alpha[c] - 1
            alpha[b] = alpha[b] - 1
        alpha[a] = 0
    return G, DG, D2G, D3G, D4G, D5G

def derivatives_of_kernel( nodes , q ):
    #given x_i and x_j the K = Kernel( x_ij) and derivatives with x_ij = x_i - x_j.
    #The code is written such that we evaluate at the nodes, and entry (i,j) is the contribution at node i due to particle j.
    delta = np.identity( DIM )
    G,DG,D2G,D3G,D4G,D5G = derivatives_of_Gaussians( nodes , q )
    K = np.einsum('ij,ab->ijab',G,delta)
    DK = np.einsum('ijc,ab->ijabc',DG,delta)
    D2K = np.einsum('ijcd,ab->ijabcd',D2G,delta)
    D3K = np.einsum('ijcde,ab->ijabcde',D3G,delta)
    D4K = np.einsum('ijcdef,ab->ijabcdef',D4G,delta)
    D5K = np.einsum('ijcdefg,ab->ijabcdefg',D5G,delta)
    #EXAMPLE OF INDEX CONVENTION 'ijabc' refers to the c^th derivative of the ab^th entry of K(q_i - q_j)
    return K, DK, D2K, D3K , D4K , D5K 

def Hamiltonian( q , p , mu_1 , mu_2 ):
    #return the Hamiltonian.  Serves as a safety to check our equations of motion are correct.
    K,DK,D2K,D3K,D4K,D5K = derivatives_of_kernel(q,q)
    term_00 = 0.5*np.einsum('ia,ijab,jb',p,K,p)
    term_01 = - np.einsum('ia,ijabc,jbc',p,DK,mu_1)
    term_11 = -0.5*np.einsum('iad,ijabcd,jbc',mu_1,D2K,mu_1)
    term_02 = np.einsum('ia,jbcd,ijabcd',p,mu_2,D2K)
    term_12 = np.einsum('iae,jbcd,ijabecd',mu_1,mu_2,D3K)
    term_22 = 0.5*np.einsum('iaef,jbcd,ijabcdef',mu_2,mu_2,D4K)
    return term_00 + term_01 + term_11 + term_02 + term_12 + term_22
    
def ode_function( state , t ):
    q , p , mu_1 , mu_2 = state_to_weinstein_darboux( state )
    K,DK,D2K,D3K,D4K,D5K = derivatives_of_kernel( q , q )
    dq = np.einsum('ijab,jb->ia',K,p) - np.einsum('ijabc,jbc->ia',DK,mu_1) + np.einsum('jbcd,ijabcd->ia',mu_2,D2K)
    T00 = -np.einsum('ic,jb,ijcba->ia',p,p,DK)
    T01 = np.einsum('id,jbc,ijdbac->ia',p,mu_1,D2K) - np.einsum('jd,ibc,ijdbac->ia',p,mu_1,D2K)
    #THERE IS AN ERROR IN ONE OF THE NEXT FOUR LINES
    T02 = -np.einsum('ie,jbcd,ijebacd->ia',p,mu_2,D3K)-np.einsum('je,ibcd,ijebacd->ia',p,mu_2,D3K)
    T12 = -np.einsum('ife,jbcd,ijfbacde->ia',mu_1,mu_2,D4K)+np.einsum('jfe,ibcd,ijfbacde->ia',mu_1,mu_2,D4K)
    T11 = np.einsum('ied,jbc,ijebacd->ia',mu_1,mu_1,D3K)
    T22 = np.einsum('izef,jbcd,ijzbafcde->ia',mu_2,mu_2,D5K)
    xi_1 = np.einsum('ijacb,jc->iab',DK,p) - np.einsum('ijacbd,jcd->iab',D2K,mu_1) + np.einsum('jecd,ijaebcd->iab',mu_2,D3K)
    xi_2 = np.einsum('ijadbc,jd->iabc',D2K,p) - np.einsum('ijaebcd,jed->iabc',D3K,mu_1) + np.einsum('jefd,ijeabcfd->iab',mu_2,D4K)
    dp = T00 + T01 + T02 + T12 + T11 + T22
    dmu_1 = np.einsum('iac,ibc->iab',mu_1,xi_1)\
        - np.einsum('icb,ica->iab',mu_1,xi_1)\
        + np.einsum('iadc,ibdc->iab',mu_2,xi_2)\
        - np.einsum('idbc,idac->iab',mu_2,xi_2)\
        - np.einsum('idcb,idca->iab',mu_2,xi_2)
    dmu_2 = np.einsum('iadc,ibd->iabc',mu_2,xi_1)\
        + np.einsum('iacd,ibd->iabc',mu_2,xi_1)\
        - np.einsum('idbc,ida->iabc',mu_2,xi_1)
    dstate = weinstein_darboux_to_state( dq , dp , dmu_1 , dmu_2 )
    return dstate

def state_to_weinstein_darboux( state ):
    i = 0
    q = np.reshape( state[i:(i+N*DIM)] , [N,DIM] )
    i = i + N*DIM
    p = np.reshape( state[i:(i+N*DIM)] , [N,DIM] )
    i = i + N*DIM
    mu_1 = np.reshape( state[i:(i + N*DIM*DIM)] , [N,DIM,DIM] )
    i = i + N*DIM*DIM
    mu_2 = np.reshape( state[i:(i + N*DIM*DIM*DIM)] ,[N,DIM,DIM,DIM] ) 
    return q , p , mu_1 , mu_2

def weinstein_darboux_to_state( q , p , mu_1, mu_2 ):
    state = np.zeros( 2*N*DIM + N*DIM*DIM + N*DIM*DIM*DIM )
    i = 0
    state[i:(N*DIM)] = np.reshape( q , N*DIM )
    i = i + N*DIM 
    state[i:(i + N*DIM)] = np.reshape( p , N*DIM )
    i = i + N*DIM
    state[i:(i+N*DIM*DIM)] = np.reshape( mu_1 , N*DIM*DIM)
    i = i + N*DIM*DIM
    state[i:(i+N*DIM*DIM*DIM)] = np.reshape( mu_2 , N*DIM*DIM*DIM ) 
    return state

def display_velocity_field( q , p ,mu_1 , mu_2 ):
 W = 5*SIGMA
 res = 30
 N_nodes = res**DIM
 store = np.outer( np.linspace(-W,W , res), np.ones(res) )
 nodes = np.zeros( [N_nodes , DIM] )
 nodes[:,0] = np.reshape( store , N_nodes )
 nodes[:,1] = np.reshape( store.T , N_nodes )
 K,DK,D2K,D3K = derivatives_of_kernel( nodes , q )
 vel_field = np.einsum('ijab,jb->ia',K,p) - np.einsum('ijabc,jbc->ia',DK,mu_1) + np.einsum('ijabcd,jbcd->ia',D2K,mu_2)
 U = vel_field[:,0]
 V = vel_field[:,1]

 plt.figure()
 plt.quiver( nodes[:,0] , nodes[:,1] , U , V )
 plt.plot(q[:,0],q[:,1],'ro')
 for i in range(0,N):
     plt.arrow(q[i,0], q[i,1], 0.2*p[i,0], 0.2*p[i,1], head_width=0.05, head_length=0.1, fc='b', ec='b')
 plt.axis([- W, W,- W, W ])
 return plt.gcf()

def test_Gaussians( q ):
    h = 1e-7
    G,DG,D2G,D3G,D4G,D5G = derivatives_of_Gaussians(q,q)
    q_a = np.copy(q)
    q_b = np.copy(q)
    q_c = np.copy(q)
    q_d = np.copy(q)
    q_e = np.copy(q)
    for i in range(0,N):
        for a in range(0,DIM):
            error_max = 0.
            q_a[i,a] = q[i,a]+h
            G_a , DG_a , D2G_a , D3G_a, D4G_a , D5G_a = derivatives_of_Gaussians(q_a, q) 
            for j in range(0,N):
                error = (G_a[i,j] - G[i,j])/h - DG[i,j,a]
                error_max = np.maximum( np.absolute( error ) , error_max )
            print 'max error for DG was ' + str( error_max )
            error_max = 0.
            for b in range(0,DIM):
                q_b[i,b] = q[i,b] + h
                G_b , DG_b , D2G_b , D3G_b , D4G_b , D5G_b  = derivatives_of_Gaussians( q_b , q ) 
                for j in range(0,N):
                    error = (DG_b[i,j,a] - DG[i,j,a])/h - D2G[i,j,a,b]
                    error_max = np.maximum( np.absolute( error ) , error_max )
                print 'max error for D2G was ' + str( error_max )
                error_max = 0.
                for c in range(0,DIM):
                    q_c[i,c] = q_c[i,c] + h
                    G_c , DG_c , D2G_c , D3G_c , D4G_c , D5G_c = derivatives_of_Gaussians( q_c , q ) 
                    for j in range(0,N):
                        error = (D2G_c[i,j,a,b] - D2G[i,j,a,b])/h - D3G[i,j,a,b,c]
                        error_max = np.maximum( np.absolute(error) , error_max )
                    print 'max error for D3G was ' + str( error_max )
                    error_max = 0.
                    for d in range(0,DIM):
                        q_d[i,d] = q[i,d] + h
                        G_d, DG_d , D2G_d , D3G_d, D4G_d , D5G_d = derivatives_of_Gaussians( q_d , q )
                        for j in range(0,N):
                            error = (D3G_d[i,j,a,b,c] - D3G[i,j,a,b,c])/h - D4G[i,j,a,b,c,d]
                            error_max = np.maximum( np.absolute(error) , error_max )
                        print 'max error for D4G was '+ str(error_max)
                        error_max = 0.
                        for e in range(0,DIM):
                            q_e[i,e] = q[i,e] + h
                            G_e, DG_e , D2G_e , D3G_e, D4G_e, D5G_e = derivatives_of_Gaussians( q_e , q )
                            for j in range(0,N):
                                error = (D4G_e[i,j,a,b,c,d] - D4G[i,j,a,b,c,d])/h - D5G[i,j,a,b,c,d,e]
                                error_max = np.maximum( np.absolute(error) , error_max )
                            print 'max error for D5G was '+ str(error_max)
                            error_max = 0.
                            q_e[i,e] = q_e[i,e] - h
                        q_d[i,d] = q_d[i,d] - h
                    q_c[i,c] = q_c[i,c] - h
                q_b[i,b] = q_b[i,b] - h
            q_a[i,a] = q_a[i,a] - h
    return 1

def test_kernel_functions( q ):
    h = 1e-7
#    G,DG,D2G,D3G,D4G,D5G = derivatives_of_Gaussians(q,q)
    q_a = np.copy(q)
    q_b = np.copy(q)
    q_c = np.copy(q)
    q_d = np.copy(q)
    q_e = np.copy(q)
    K,DK,D2K,D3K,D4K,D5K = derivatives_of_kernel(q,q)
    delta = np.identity(DIM)
    error_max = 0.
    for i in range(0,N):
        for j in range(0,N):
            x = q[i,:] - q[j,:]
            r_sq = np.inner( x , x )
            for a in range(0,DIM):
                for b in range(0,DIM):
                    G = np.exp( -r_sq / (2.*SIGMA**2) )
                    K_ij_ab = G*delta[a,b]
                    error = K_ij_ab - K[i,j,a,b]
                    error_max = np.maximum( np.absolute(error) , error_max )

    print 'error_max for K = ' + str( error_max )
    if (error_max > 100*h):
        print 'WARNING:  COMPUTATION OF K APPEARS TO BE INACCURATE'

    error_max = 0.
    for i in range(0,N):
        for a in range(0,DIM):
            q_a[i,a] = q[i,a] + h
            K_a,DK_a,D2K_a,D3K_a,D4K_a,D5K_a = derivatives_of_kernel(q_a,q)
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
                K_b,DK_b,D2K_b,D3K_b,D4K_b,D5K_b = derivatives_of_kernel(q_b,q)
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
                    K_c,DK_c,D2K_c,D3K_c,D4K_c,D5K_c = derivatives_of_kernel(q_c,q)
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
    return 1


def test_functions( trials ):
    #checks that each function does what it is supposed to
    h = 10e-7
    q = SIGMA*np.random.randn(N,DIM)
    p = SIGMA*np.random.randn(N,DIM)
    mu_1 = np.random.randn(N,DIM,DIM)
#    mu_1 = np.zeros([N,DIM,DIM])
#    mu_2 = np.zeros([N,DIM,DIM,DIM])
    mu_2 = np.random.randn(N,DIM,DIM,DIM)
    
#    test_Gaussians( q )
#    test_kernel_functions( q )

    s = weinstein_darboux_to_state( q , p , mu_1 , mu_2 )
    ds = ode_function( s , 0 )
    dq,dp_coded,dmu_1,dmu_2 = state_to_weinstein_darboux( ds ) 

    print 'a test of the ode:'
    print 'dp_coded =' + str(dp_coded)
    Q = np.copy(q)
    dp_estim = np.zeros([N,DIM])
    for i in range(0,N):
        for a in range(0,DIM):
            Q[i,a] = q[i,a] + h
            dp_estim[i,a] = - ( Hamiltonian(Q,p,mu_1,mu_2) - Hamiltonian(q,p,mu_1,mu_2) ) / h 
            Q[i,a] = Q[i,a] - h
    print 'dp_estim =' + str(dp_estim)
    print 'dp_error =' + str(dp_estim - dp_coded)
    return 1

test_functions(1)
