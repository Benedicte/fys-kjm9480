import numpy as np
import matplotlib.pyplot as plt


state = np.array([[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]])
state_CISD = np.array([[1,2],[1,3],[1,4],[2,3],[2,4]])

determinant = np.zeros((6, 6))
determinant_CISD = np.zeros((5, 5))

x_axis=np.array([np.arange(-1.0, 1.1, 0.1)]).transpose()
y_axis = np.zeros((1))
y_axis_FCI = np.zeros((6))
y_axis_CISD = np.zeros((1))
y_axis_rspt = np.zeros((3))

def single_particle_contribution(p, q, r, s):
	
	sp_contribution = 2*kroniker_delta(p,r)*kroniker_delta(q,s)*(r+s-2)
		
	return sp_contribution

def hamiltonian(p, q, r, s, g):
	
	single_particle_contribution = 2*kroniker_delta(p,r)*kroniker_delta(q,s)\
		*(r+s-2)

	double_particle_contribution = -0.5*g*((kroniker_delta(p,r)+kroniker_delta(q,s)\
	+ kroniker_delta(p,s)+kroniker_delta(q,r)))
	
	total_contribution = single_particle_contribution + double_particle_contribution
	
	
	return total_contribution
		
def kroniker_delta(a,b):
	if(a == b):
		return 1
	
	else:
		return 0				

def fill_determinant_matrix(g):
	
	for i in range (0, 6):
		for j in range (0, 6):
			determinant[i,j] = hamiltonian(state[i,0],state[i,1], state[j,0], state[j,1],g)
			
def fill_determinant_matrix_CISD(g):
	
	for i in range (0, 5):
		for j in range (0, 5):
			determinant_CISD[i,j] = hamiltonian(state_CISD[i,0],state_CISD[i,1], state_CISD[j,0], state_CISD[j,1],g)		

def gamma(i,j):
	
	#1/(e12 - eij)
	
	gamma = 1.0/(2 - single_particle_contribution(i, j, i, j))
	return gamma

def RSPT(g, eigval_CI):
	E0 = 2
	E1 = -1*g
	E2 = -0.5*g * ((gamma(1,3)*eigval_CI[0,1] + gamma(1,4)*eigval_CI[0,2] \
                 + gamma(2,3)*eigval_CI[0,3] + gamma(2,4)*eigval_CI[0,4]))

	E31 = (g*g/4)*(gamma(1,3)**2*(2*eigval_CI[0,0] + eigval_CI[0,1] + eigval_CI[0,2] + eigval_CI[0,4])\
                 + gamma(1,4)**2*(2*eigval_CI[0,1] + eigval_CI[0,0] + eigval_CI[0,3] + eigval_CI[0,4])\
                 + gamma(2,3)**2*(2*eigval_CI[0,2] + eigval_CI[0,3] + eigval_CI[0,0] + eigval_CI[0,4])\
                 + gamma(2,4)**2*(2*eigval_CI[0,3] + eigval_CI[0,1] + eigval_CI[0,2] + eigval_CI[0,4])) 
	E32 = -1*(g*g/2)*(gamma(1,3)**2*eigval_CI[0,0] + gamma(1,4)**2*eigval_CI[0,1]\
                 + gamma(2,3)**2*eigval_CI[0,2] + gamma(1,4)**2*eigval_CI[0,3])
	E3 = E31 + E32
    

	E1_total = E0 + E1
	E2_total = E0 + E1 + E2
	E3_total = E0 + E1 + E2 + E3
    
	rsp_energies = np.array([E1_total, E2_total, E3_total])
    
	return rsp_energies

for g in np.arange(-1.0, 1.1, 0.1):
    
    fill_determinant_matrix(g)
    eigval, eigvec = np.linalg.eigh(determinant) 
     
    y_axis = np.vstack((y_axis, np.sum(np.multiply(eigvec[:1,:1],eigvec[:1,:1]))))
    rspt_energies = RSPT(g, determinant)
    
    y_axis_rspt = np.vstack((y_axis_rspt, rspt_energies))
    y_axis_FCI = np.vstack((y_axis_FCI, eigval))
 
def F(p, epsilon):
    if (p == 1 or p == 2):
        return epsilon[p-1] - 0.5*g
    elif (p == 3 or p == 4):
        return epsilon[p-1]

def CCamplitudes(sigma, epsilon, T_old, g):

    T = [[0 for i in range(2)] for j in range(2)]

    for i in range(1,3):
        for a in range(3,5):
            I=i-1
            A=a-3
            denom=2.0*(F(i,epsilon) - F(a,epsilon)) + sigma
            numerator = sigma*T_old[I][A]
            numerator -= 0.5*g
            for b in range(3,5):
                numerator -= 0.5*g*T_old[I][b-3]
            for j in range(1,3):
                numerator -= 0.5*g*T_old[j-1][A]
            numerator -= 0.5*g*T_old[0][0]*T_old[1][1]
            numerator -= 0.5*g*T_old[0][1]*T_old[1][0]
            summ = 0.0
            for b in range(3,5):
                for j in range(1,3):
                    summ += T_old[j-1][b-3]
            summ *= T_old[I][A]
            numerator += 0.5*g*summ
            T[I][A] = numerator/denom
    return T

epsilon=[]

for i in range(1,5):
    epsilon.append((i-1))

maxiter = 100
sigma = -0.5
y_axis_ccd=[]
for g in np.arange(-1.0, 1.1, 0.1):
    T0 = [[0 for i in range(2)] for j in range(2)]
    T = [[0 for i in range(2)] for j in range(2)]
    
    for i in range(0,maxiter):
        T = CCamplitudes(sigma, epsilon, T0, g)
        T0 = T
    
    sumofall=0.0
    for i in range(2):
        for a in range(2):
            sumofall += T[i][a]
    y_axis_ccd.append(2.0 - g -g*0.5*sumofall)  

##########Plots##########

fill_determinant_matrix(-1)
print(determinant)

print single_particle_contribution(1,3,1,3)
print(gamma(1,3))


######FCI#######

plt.figure(1)
plt.plot(x_axis, y_axis[1:,0], 'k')
plt.xlabel('g', fontsize=12)
plt.ylabel('$f(g)$', fontsize=12)
plt.savefig("FCI_probability")

#####RSPT######

plt.figure(2)

plt.plot(x_axis, y_axis_rspt[1:,2], 'g',  label=r'RSPT')
plt.xlabel('g', fontsize=12)
plt.ylabel('$Energy$', fontsize=12)
plt.legend(loc="upper right", fontsize=10)
plt.savefig("plot_RSPT")

#####CCD######

plt.figure(3)

plt.plot(x_axis, y_axis_ccd, 'b',  label= r'$E_{CCD}$')
plt.xlabel('g', fontsize=12)
plt.ylabel('$Energy$', fontsize=12)
plt.legend(loc="upper right", fontsize=10)
plt.savefig("plot_CCD")


####All in one########

plt.figure(4)

plt.plot(x_axis,y_axis_FCI[1:,0], 'b',  label=r'FCI')
plt.plot(x_axis, y_axis_rspt[1:,2], 'r',  label=r'RSPT')
plt.plot(x_axis, y_axis_ccd, 'y', linestyle= "dashed", label=r'CCD')
plt.xlabel('g', fontsize=12)
plt.ylabel('$Energy$', fontsize=12)
plt.legend(loc="upper right", fontsize=10)
plt.savefig("plot_all")

print("Sucessfully ended")


			
	
	

	
	
