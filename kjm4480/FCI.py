import numpy as np
import matplotlib.pyplot as plt


state = np.array([[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]])
state_CISD = np.array([[1,2],[1,3],[1,4],[2,3],[2,4]])

determinant = np.zeros((6, 6))
determinant_CISD = np.zeros((5, 5))

x_axis=np.array([np.arange(-1.0, 1.1, 0.1)]).transpose()
y_axis = np.zeros((6))
y_axis_CISD = np.zeros((5))

def hamiltonian(p, q, r, s, g):
	
	single_particle_contribution = 2*kroniker_delta(p,r)*kroniker_delta(q,s)\
		*(r+s-2)

	#print("single particle contribution")
	#print (single_particle_contribution)
	
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

for g in np.arange(-1.0, 1.1, 0.1):
    fill_determinant_matrix(g)
    fill_determinant_matrix_CISD(g)
    
    eigval, eigvec = np.linalg.eigh(determinant)
    eigval_CISD, eigvec_CISD = np.linalg.eigh(determinant_CISD) 
    
    y_axis = np.vstack((y_axis, eigval))
    y_axis_CISD = np.vstack((y_axis_CISD, eigval_CISD))

    

eigval, eigvec = np.linalg.eigh(determinant)

eigval_sorted = np.sort(eigval)


plt.figure(0)
plt.plot(x_axis, y_axis[1:,0], 'b',  label=r'$k=1$')
plt.plot(x_axis, y_axis[1:,1], 'r',  label=r'$k=2$')
plt.plot(x_axis, y_axis[1:,2], 'y',  label=r'$k=3$')
plt.plot(x_axis, y_axis[1:,3], 'k', linestyle= "dashed",  label=r'$k=4$')
plt.plot(x_axis, y_axis[1:,4], 'g',  label=r'$k=5$')
plt.plot(x_axis, y_axis[1:,5], 'm',  label=r'$k=6$')


plt.xlabel('g', fontsize=12)
plt.ylabel('$E_k$', fontsize=12)
plt.legend(loc="upper right", fontsize=10)


plt.savefig("plot_1")

plt.figure(1)

plt.plot(x_axis, y_axis_CISD[1:,0], 'b',  label=r'$k=1$')
plt.plot(x_axis, y_axis_CISD[1:,1], 'r',  label=r'$k=2$')
plt.plot(x_axis, y_axis_CISD[1:,2], 'y',  label=r'$k=3$')
plt.plot(x_axis, y_axis_CISD[1:,3], 'k', linestyle= "dashed",  label=r'$k=4$')
plt.plot(x_axis, y_axis_CISD[1:,4], 'g',  label=r'$k=5$')
plt.xlabel('g', fontsize=12)
plt.ylabel('$E_k$', fontsize=12)
plt.legend(loc="upper right", fontsize=10)
plt.savefig("plot_CID")


			
	
	

	
	
