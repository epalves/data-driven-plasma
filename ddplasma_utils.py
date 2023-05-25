# #########################
# data driven plasma utils
# #########################

import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import KFold
import numba
import pickle

def variance_of_model_form_with_mterms(regression_results):
    K = regression_results['K']
    range_numnonzeroterms = range(0, int(regression_results['num_non_zero_terms'].max()) + 1)
    
    num_nonzero_terms = []
    variance_of_model_form = []
    for num in range_numnonzeroterms:
        idxs_with_models_with_numterms = (regression_results['num_non_zero_terms'] == num)
        # if np.sum(regression_results['num_non_zero_terms'] == num) >2:
        if np.sum(regression_results['num_non_zero_terms'] == num) >1:
            sparsity_pattern_of_models_with_numterms = (np.abs(regression_results['coeff_evol'][idxs_with_models_with_numterms,:])>0)
            variance_of_model_form.append(np.sum(np.std(sparsity_pattern_of_models_with_numterms, axis = 0)))
            num_nonzero_terms.append(num)
        
    return np.array(num_nonzero_terms), np.array(variance_of_model_form)


def rebin_errors_vs_numnonzeroterms(regression_results, train_test = 'mse_test', if_display_modelformvariance = False):
	

	K = regression_results['K']

	range_numnonzeroterms = range(0, int(regression_results['num_non_zero_terms'].max()) + 1)
	
	ave_error = []
	min_error = []
	max_error = []
	numnonzeroterms_array = []
	
	for num in range_numnonzeroterms:
		# normalized test errors for models with num terms
		normalized_errors = (regression_results[train_test]/regression_results['error_norm']).flatten()[regression_results['num_non_zero_terms'].flatten() == num]
		# if len(normalized_errors) > 2:
		if len(normalized_errors) > 1:
			ave_error.append(np.mean(normalized_errors))
			min_error.append(np.min(normalized_errors))
			max_error.append(np.max(normalized_errors))
			numnonzeroterms_array.append(num)

	ave_error = np.array(ave_error)
	min_error = np.array(min_error)
	max_error = np.array(max_error)
	numnonzeroterms_array = np.array(numnonzeroterms_array)

	if if_display_modelformvariance == True:
		n_nonzero_terms, model_form_var = variance_of_model_form_with_mterms(regression_results)

		idx_zero_modelformvar = (model_form_var == 0)

		if train_test == 'mse_test':
			plt.errorbar(numnonzeroterms_array[idx_zero_modelformvar], ave_error[idx_zero_modelformvar], 
						 yerr= [min_error[idx_zero_modelformvar], max_error[idx_zero_modelformvar]] , label='both limits (default)', fmt = 'o', c = 'C0') # fmt='o', c = 'C0')
			plt.errorbar(numnonzeroterms_array[~idx_zero_modelformvar], ave_error[~idx_zero_modelformvar], 
						 yerr= [min_error[~idx_zero_modelformvar], max_error[~idx_zero_modelformvar]] , label='both limits (default)', fmt = '^', c = 'C0') # fmt='o', c = 'C0')
		else:
			plt.scatter(numnonzeroterms_array[idx_zero_modelformvar], ave_error[idx_zero_modelformvar], c = 'C1', marker = 'o')
			plt.scatter(numnonzeroterms_array[~idx_zero_modelformvar], ave_error[~idx_zero_modelformvar], c = 'C1', marker = '^')
		plt.yscale('log')
		plt.ylim(ymax = 5)

	else:

		if train_test == 'mse_test':
			plt.errorbar(numnonzeroterms_array, ave_error, 
	                     yerr= [min_error, max_error] , label='both limits (default)', fmt='o', c = 'C0')
		else:
			plt.scatter(numnonzeroterms_array, ave_error, c = 'C1', marker = 'o')
		plt.yscale('log')
		plt.ylim(ymax = 5)

	return ave_error, min_error, max_error, numnonzeroterms_array

def Kfoldaveraged_test_train_errors(regression_results, l0_penalty = 0, get_lambda_opt = True):
    Kfoldaveraged_test_error = np.mean(regression_results['mse_test'],axis = 0) + l0_penalty * np.mean(regression_results['num_non_zero_terms'],axis=0)
    Kfoldaveraged_train_error = np.mean(regression_results['mse_train'],axis = 0)
    
    plt.loglog(regression_results['lamdas'], Kfoldaveraged_train_error/regression_results['error_norm'], label = r'$<MSE_{train}>$')
    plt.loglog(regression_results['lamdas'], Kfoldaveraged_test_error/regression_results['error_norm'], label = r'$<MSE_{test}> + \alpha ||\xi||_0$')
    # plt.ylim(1e-8,)
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'MSE')
    plt.legend()
    plt.tight_layout()

    if get_lambda_opt == True:
        lambda_opt = regression_results['lamdas'][np.argmin(Kfoldaveraged_test_error)]
        print('lambda_opt = ', lambda_opt)
        return lambda_opt

def save_regression_results(regression_results, name ):
    with open('./regression_results__'+ name + '.pkl', 'wb') as f:
        pickle.dump(regression_results, f, pickle.HIGHEST_PROTOCOL)

def load_pickle_obj(pickle_file):
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)

def full_correlations_diagnostic(X_data, term_labels, correlation_threshold = 0.95, label_size = 'small', figsize = (5,4)):
    # calculate Pearson correlation
    corr = np.corrcoef(X_data.T)
    n_terms = corr.shape[0]
    below_diagonal_indices = np.indices((n_terms,n_terms))[0]<np.indices((n_terms,n_terms))[1]

    # correlation_threshold = 0.95 # above this threshold, two variables are considered strongly correlated
    corr = np.nan_to_num(corr)
    corr_variables = np.abs(corr*below_diagonal_indices)>correlation_threshold

    plt.figure(figsize = figsize)
    plt.subplot(1,3,1)
    plt.title('Pearson Correlation Matrix')
    plt.imshow(corr, origin ='lower', vmin = -1, vmax = 1, cmap='bwr')
    term_labels_2 = [r'$'+term +'$' for term in term_labels]
    plt.xticks(range(len(term_labels)), term_labels_2, size = label_size, rotation='vertical')
    plt.yticks(range(len(term_labels)), term_labels_2, size = label_size)
    plt.colorbar()

    plt.subplot(1,3,2)
    plt.title('Below diagonal coefficients')
    plt.imshow(corr*below_diagonal_indices, origin ='lower', vmin = -1, vmax = 1, cmap='bwr')
    plt.xticks(range(len(term_labels)), term_labels_2, size = label_size, rotation='vertical')
    plt.yticks(range(len(term_labels)), term_labels_2, size = label_size)
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.title('Strongly correlated variables')
    plt.imshow(corr_variables, origin ='lower', vmin = -1, vmax = 1, cmap='bwr')
    plt.xticks(range(len(term_labels)), term_labels_2, size = label_size, rotation='vertical')
    plt.yticks(range(len(term_labels)), term_labels_2, size = label_size)
    plt.colorbar()

    plt.tight_layout()
    
    corr_args = np.argwhere(corr_variables>0)
    corr_var_list = []
    for corr_arg in corr_args:
        corr_var_list.append([term_labels[corr_arg[0]], term_labels[corr_arg[1]]])
    
    return corr_var_list
    
def correlations_diagnostic(X_data, term_labels, correlation_threshold = 0.95, label_size = 'small', figsize = (5,4)):
    # calculate Pearson correlation
    corr = np.corrcoef(X_data.T)
    n_terms = corr.shape[0]
    below_diagonal_indices = np.indices((n_terms,n_terms))[0]<np.indices((n_terms,n_terms))[1]

    # correlation_threshold = 0.95 # above this threshold, two variables are considered strongly correlated
    corr_variables = np.abs(corr*below_diagonal_indices)>correlation_threshold

    plt.figure(figsize=figsize)
    plt.title('Strongly correlated variables')
    plt.imshow(corr_variables, origin ='lower', vmin = -1, vmax = 1, cmap='bwr')
    plt.xticks(range(len(term_labels)), term_labels, size = label_size, rotation='vertical')
    plt.yticks(range(len(term_labels)), term_labels, size = label_size)

    plt.tight_layout()


def FFTDiff(u, x):
    u_fft = np.fft.fft(u)
    kx = 2*np.pi*np.fft.fftfreq(len(x), x[1]-x[0])
    
    return np.fft.ifft(1.0j * kx * u_fft).real

@numba.njit()
def FiniteDiff(u, dx, d):
    """
    Takes dth derivative data using 2nd order finite difference method (up to d=3)
    Works but with poor accuracy for d > 3
    
    Input:
    u = data to be differentiated
    dx = Grid spacing.  Assumes uniform spacing
    """
    
    n = u.size
    ux = np.zeros(n, dtype=np.float64)
    
    if d == 1:
        for i in range(1,n-1):
            ux[i] = (u[i+1]-u[i-1]) / (2*dx)
        
        ux[0] = (-3.0/2*u[0] + 2*u[1] - u[2]/2) / dx
        ux[n-1] = (3.0/2*u[n-1] - 2*u[n-2] + u[n-3]/2) / dx
        return ux
    
    if d == 2:
        for i in range(1,n-1):
            ux[i] = (u[i+1]-2*u[i]+u[i-1]) / dx**2
        
        ux[0] = (2*u[0] - 5*u[1] + 4*u[2] - u[3]) / dx**2
        ux[n-1] = (2*u[n-1] - 5*u[n-2] + 4*u[n-3] - u[n-4]) / dx**2
        return ux
    
    if d == 3:
        for i in range(2,n-2):
            ux[i] = (u[i+2]/2-u[i+1]+u[i-1]-u[i-2]/2) / dx**3
        
        ux[0] = (-2.5*u[0]+9*u[1]-12*u[2]+7*u[3]-1.5*u[4]) / dx**3
        ux[1] = (-2.5*u[1]+9*u[2]-12*u[3]+7*u[4]-1.5*u[5]) / dx**3
        ux[n-1] = (2.5*u[n-1]-9*u[n-2]+12*u[n-3]-7*u[n-4]+1.5*u[n-5]) / dx**3
        ux[n-2] = (2.5*u[n-2]-9*u[n-3]+12*u[n-4]-7*u[n-5]+1.5*u[n-6]) / dx**3
        return ux
    
    if d > 3:
        return FiniteDiff(FiniteDiff(u,dx,3), dx, d-3)

@numba.njit()
def FiniteDiff_O4(u, dx, d):
    """
    Takes dth derivative data using 4th order finite difference method (up to d=3)
    Works but with poor accuracy for d > 3
    
    Input:
    u = data to be differentiated
    dx = Grid spacing.  Assumes uniform spacing
    """
    
    n = u.size
    ux = np.zeros(n, dtype=np.float64)
    
    if d == 1:
        for i in range(2,n-2):
            ux[i] = (u[i-2]-8*u[i-1] + 8*u[i+1]-u[i+2]) / (12*dx)
        
        # Still using 2nd order estimation for boundaries
        ux[0] = (-3.0/2*u[0] + 2*u[1] - u[2]/2) / dx
        ux[1] = (u[2] -u[0] ) / (2*dx)
        ux[n-1] = (3.0/2*u[n-1] - 2*u[n-2] + u[n-3]/2) / dx
        ux[n-2] = (u[n-1] -u[n-3] ) / (2*dx)
        return ux
    
    if d == 2:
        for i in range(2,n-2):
            ux[i] = (-u[i-2]+16*u[i-1] -30*u[i] + 16*u[i+1]-u[i+2]) / (12*dx**2)
        
        # Still using 2nd order estimation for boundaries
        ux[0] = (2*u[0] - 5*u[1] + 4*u[2] - u[3]) / dx**2
        ux[1] = (u[1+1]-2*u[1]+u[1-1]) / dx**2
        ux[n-1] = (2*u[n-1] - 5*u[n-2] + 4*u[n-3] - u[n-4]) / dx**2
        ux[n-2] = (u[n-2+1]-2*u[n-2]+u[n-2-1]) / dx**2
        return ux

@numba.njit()
def FiniteDiff_O6(u, dx, d):
    """
    Takes dth derivative data using 4th order finite difference method (up to d=3)
    Works but with poor accuracy for d > 3
    
    Input:
    u = data to be differentiated
    dx = Grid spacing.  Assumes uniform spacing
    """
    
    n = u.size
    ux = np.zeros(n, dtype=np.float64)
    
    if d == 1:
        for i in range(3,n-3):
            ux[i] = (-1./60.*u[i-3] + 3./20.*u[i-2] -3./4.*u[i-1] + 3./4.*u[i+1] - 3./20.*u[i+2] + 1./60.*u[i+3]) / dx
        
        # Still using 2nd order estimation for boundaries
        ux[0] = (-3.0/2*u[0] + 2*u[1] - u[2]/2) / dx
        ux[1] = (u[2] -u[0] ) / (2*dx)
        ux[2] = (u[3]-u[1]) / (2*dx)
        ux[n-1] = (3.0/2*u[n-1] - 2*u[n-2] + u[n-3]/2) / dx
        ux[n-2] = (u[n-1] -u[n-3] ) / (2*dx)
        ux[n-3] = (u[n-2] -u[n-4] ) / (2*dx)
        return ux
    
    if d == 2:
        for i in range(3,n-3):
            ux[i] = (1./90.*u[i-3] - 3./20.*u[i-2] +3./2.*u[i-1] -49./18*u[i] + 3./2.*u[i+1] - 3./20.*u[i+2] + 1./90.*u[i+3]) / (dx**2)
        
        # Still using 2nd order estimation for boundaries
        ux[0] = (2*u[0] - 5*u[1] + 4*u[2] - u[3]) / dx**2
        ux[1] = (u[1+1]-2*u[1]+u[1-1]) / dx**2
        ux[2] = (u[2+1]-2*u[2]+u[2-1]) / dx**2
        ux[n-1] = (2*u[n-1] - 5*u[n-2] + 4*u[n-3] - u[n-4]) / dx**2
        ux[n-2] = (u[n-2+1]-2*u[n-2]+u[n-2-1]) / dx**2
        ux[n-3] = (u[n-3+1]-2*u[n-3]+u[n-3-1]) / dx**2
        return ux

@numba.njit()
def FiniteDiff_O8(u, dx, d):
    """
    Takes dth derivative data using 8th order finite difference method (up to d=3)
    Works but with poor accuracy for d > 3
    
    Input:
    u = data to be differentiated
    dx = Grid spacing.  Assumes uniform spacing
    """
    
    n = u.size
    ux = np.zeros(n, dtype=np.float64)
    
    if d == 1:
        for i in range(4,n-4):
            ux[i] = (1./280.*u[i-4] - 4./105.*u[i-3] + 1./5.*u[i-2] -4./5.*u[i-1] + 4./5.*u[i+1] - 1./5.*u[i+2] + 4./105.*u[i+3] - 1./280.*u[i+4]) / dx
        
        # Still using 2nd order estimation for boundaries
        ux[0] = (-3.0/2*u[0] + 2*u[1] - u[2]/2) / dx
        ux[1] = (u[2] -u[0] ) / (2*dx)
        ux[2] = (u[3]-u[1]) / (2*dx)
        ux[3] = (u[4]-u[2]) / (2*dx)
        ux[n-1] = (3.0/2*u[n-1] - 2*u[n-2] + u[n-3]/2) / dx
        ux[n-2] = (u[n-1] -u[n-3] ) / (2*dx)
        ux[n-3] = (u[n-2] -u[n-4] ) / (2*dx)
        ux[n-4] = (u[n-3] -u[n-5] ) / (2*dx)
        return ux
    
    if d == 2:
        for i in range(4,n-4):
            ux[i] = (-1./560.*u[i-4] + 8./315.*u[i-3] - 1./5.*u[i-2] + 8./5.*u[i-1] - 205./72.*u[i] + 8./5.*u[i+1] - 1./5.*u[i+2] + 8./315.*u[i+3] - 1./560.*u[i+4]) / (dx**2)
        
        # Still using 2nd order estimation for boundaries
        ux[0] = (2*u[0] - 5*u[1] + 4*u[2] - u[3]) / dx**2
        ux[1] = (u[1+1]-2*u[1]+u[1-1]) / dx**2
        ux[2] = (u[2+1]-2*u[2]+u[2-1]) / dx**2
        ux[3] = (u[3+1]-2*u[3]+u[3-1]) / dx**2
        ux[n-1] = (2*u[n-1] - 5*u[n-2] + 4*u[n-3] - u[n-4]) / dx**2
        ux[n-2] = (u[n-2+1]-2*u[n-2]+u[n-2-1]) / dx**2
        ux[n-3] = (u[n-3+1]-2*u[n-3]+u[n-3-1]) / dx**2
        ux[n-4] = (u[n-4+1]-2*u[n-4]+u[n-4-1]) / dx**2
        return ux

@numba.njit()
def FiniteDiffPoint(u, i_x, dx, d, O = 2):
    if O == 2:
        if d == 1:
            return FiniteDiff(u[i_x-1:i_x+2], dx=dx, d=d)[1]
        if d == 2:
            return FiniteDiff(u[i_x-2:i_x+3], dx=dx, d=d)[2]
    elif O == 4:
        return FiniteDiff_O4(u[i_x-2:i_x+3], dx=dx, d=d)[2]
    elif O == 6:
        return FiniteDiff_O6(u[i_x-3:i_x+4], dx=dx, d=d)[3]
    elif O == 8:
        return FiniteDiff_O8(u[i_x-4:i_x+5], dx=dx, d=d)[4]

@numba.njit()
def FiniteDiffPoint2(u, i_x, dx, ds, O = 2):
    derivs = []
    for d in ds:
        if O == 2:
            if d == 1:
                derivs.append( FiniteDiff(u[i_x-1:i_x+2], dx=dx, d=d)[1] )
            if d == 2:
                derivs.append( FiniteDiff(u[i_x-2:i_x+3], dx=dx, d=d)[2] )
        elif O == 4:
            derivs.append( FiniteDiff_O4(u[i_x-2:i_x+3], dx=dx, d=d)[2] )
        elif O == 6:
            derivs.append( FiniteDiff_O6(u[i_x-3:i_x+4], dx=dx, d=d)[3] )
        elif O == 8:
            derivs.append( FiniteDiff_O8(u[i_x-4:i_x+5], dx=dx, d=d)[4] ) 

    return derivs



def build_Theta_and_Y(features, y_quants, X_data_descr, y_data_descr, power = 2, integration_type = None, weak_weights = None):
    
    # ###############################################################################################
    # Theta is an mxp matrix, where m is the number of measurements, and p is the number of features
    # ###############################################################################################
    # Getting m
    num_vols, num_points_per_vol = features[list(features.keys())[0]].shape[:2]
    if integration_type == None:
        m = num_vols * num_points_per_vol
    else:
        m = num_vols

    # Getting p: compute all combinations of variables in X_data_descr up to given power
    poly_terms = []
    for x in itertools.combinations_with_replacement(X_data_descr, power):
        poly_terms.append(list(x))
    p = len(poly_terms)
    
    # Initialize Y
    y = np.zeros((m, 1))
    # Initialize Theta
    theta = np.ones((m, p))

    # Get shuffle in case we want to use random integration
    shuffle_idxs = np.arange(num_vols * num_points_per_vol)
    np.random.shuffle(shuffle_idxs)

    # Fill in columns of Theta (the first column is supposed to be ones):
    print('Building Theta matrix...')
    for i in range(1, p): # ignore the first term which is a column of ones
        poly_term = poly_terms[i]
    
        column = 1
        for term in poly_term:
            if term != '1':
                column *= features[term]

        if integration_type == None:
            column = column.reshape(num_vols*num_points_per_vol)
        elif integration_type == 'vol':
            column = np.mean(column.reshape(num_vols, num_points_per_vol), axis=1)
        elif integration_type == 'weak':
            column = np.mean(column.reshape(num_vols, num_points_per_vol)*weak_weights, axis=1)/np.mean(weak_weights)
        elif integration_type == 'random':
            column = column.reshape(num_vols * num_points_per_vol)[shuffle_idxs]
            column = np.mean(column.reshape(num_vols, num_points_per_vol), axis=1)

        theta[:,i] = column

    print('Calculating Y...')
    if integration_type == None:
        y = y_quants[y_data_descr].reshape(num_vols*num_points_per_vol,1)
    elif integration_type == 'vol':
        y = np.mean(y_quants[y_data_descr].reshape(num_vols, num_points_per_vol), axis=1).reshape(num_vols,1)
    elif integration_type == 'weak':
        y = np.mean(y_quants[y_data_descr].reshape(num_vols, num_points_per_vol)*weak_weights, axis=1).reshape(num_vols,1)/np.mean(weak_weights)
    elif integration_type == 'random':
        y = y_quants[y_data_descr].reshape(num_vols * num_points_per_vol)[shuffle_idxs]
        y = np.mean(y.reshape(num_vols, num_points_per_vol), axis=1).reshape(num_vols,1)


    # Getting description of columns of Theta
    poly_terms[0] = '1'
    for i in range(1,len(poly_terms)):
        if '1' in poly_terms[i]:
            poly_terms[i].remove('1')
        poly_terms[i] = ''.join(poly_terms[i])
    description = poly_terms

    return theta, y, description



def STLS_1by1(X, y, xi = None):
    """ Sequentially thresholded least-squares"""
    m, d = X.shape

    if np.all(xi == None):
        xi = np.linalg.lstsq(X,y, rcond=None)[0]


    lam = np.min(np.abs(xi[np.nonzero(xi)]))

    big_coefs = (np.abs(xi) > lam)
    small_coefs = np.logical_not(big_coefs)
    xi[small_coefs] = 0
    
    num_relevant = sum(big_coefs)
    if num_relevant == 0:
        return np.zeros(d), lam

    else:
        xi[big_coefs] = np.linalg.lstsq(X[:, big_coefs],y, rcond=None)[0]
        return xi, lam


def Kfold_STLS_regression__1by1(Theta, Y, K_folds = 10, normalize_Theta = True):
    if normalize_Theta:
        Theta_Norm = np.sqrt(np.var(Theta, axis=0))
        Theta_Norm[0] = 1
    else:
        Theta_Norm = 1
        
    # error_Norm = np.mean(Y**2) # normalize errors to typical values of Y
    error_Norm = np.var(Y) # normalize errors to typical values of Y

    kf = KFold(n_splits=K_folds, random_state=8, shuffle=True)
    kf.get_n_splits(Theta)

    # Range of regularization parameters for STLS regression
    n_terms = Theta.shape[-1]

    n_iters = n_terms
    lamdas = np.zeros(n_terms)
    coefficients_evol = np.zeros((K_folds, n_iters, n_terms))
    mse_train         = np.zeros((K_folds, n_iters))
    mse_test          = np.zeros((K_folds, n_iters))

    number_non_zero_terms = np.zeros((K_folds, n_iters))


    k=0
    for train_index, test_index in kf.split(Theta):
        print('K-fold #', k)
        Theta_train, Theta_test = Theta[train_index], Theta[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        STLS_coefs = None
        for i in range(n_iters):
            STLS_coefs, lamdas[i] = STLS_1by1(X = Theta_train/Theta_Norm, y = Y_train[:,0], xi = STLS_coefs)

            coefficients_evol[k, i,:] = STLS_coefs/Theta_Norm

            mse_train[k, i] = np.mean((Y_train[:,0] - np.dot(Theta_train/Theta_Norm, STLS_coefs))**2)
            mse_test[k, i] = np.mean((Y_test[:,0] - np.dot(Theta_test/Theta_Norm, STLS_coefs))**2)

        number_non_zero_terms[k,:] = [np.sum(np.abs(coefficients_evol[k, i,:])>0) for i in range(n_iters)]

        k+=1

    regression_results = {}
    regression_results['coeff_evol'] = coefficients_evol
    regression_results['num_non_zero_terms'] = number_non_zero_terms
    regression_results['mse_test'] = mse_test
    regression_results['mse_train'] = mse_train
    regression_results['error_norm'] = error_Norm
    regression_results['K'] = K_folds
    regression_results['lamdas'] = lamdas
        
    return regression_results


def plot_pareto_front_analysis(regression_results, pareto_pos = None):

    K = regression_results['K']
    
    plt.figure(figsize=(7,3.25))
    plt.subplot(1,2,1)
    if pareto_pos != None:
    	plt.axvline(x=pareto_pos, c = 'k', ls = '--', lw=1)
	    # plt.axvline(x=2, c = 'k', ls = '--', lw=1)
    plt.scatter(regression_results['num_non_zero_terms'], regression_results['mse_train']/regression_results['error_norm'], 
                label = r'$E[\epsilon_{train}^2]$', facecolors='None', edgecolors = 'C1', s = 60)

    plt.scatter(regression_results['num_non_zero_terms'], regression_results['mse_test']/regression_results['error_norm'], 
                label = r'$E[\epsilon_{test}^2]$', facecolors='C3',s = 15)

    plt.xlabel(r'$\#$ nonzero terms')
    plt.ylabel(r'$E[\epsilon^2]$')
    plt.yscale('log')
    plt.ylim(5e-4,10)
#     plt.ylim(5e-2,5)
    plt.xlim(0,30)
    plt.xticks(range(0,32,2))
    plt.legend()


    plt.subplot(1,2,2)
    for k in range(K):
        plt.semilogx(regression_results['lamdas'], regression_results['num_non_zero_terms'][k,:])
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$\#$ nonzero terms')
    # plt.ylim(0,15)
    # plt.xlim(1e-6,1e-1)

    plt.yscale('log')
    plt.ylim(1,)

    plt.tight_layout()

def discovered_Mterm_models(regression_results, description, M = 1):
    unique_Mterm_models = np.unique(regression_results['coeff_evol'][(regression_results['num_non_zero_terms'][:,:] == M)], axis = 0)
    idx_non_zero_coefs_Mterm_models = (np.sum(unique_Mterm_models**2, axis=0)>0)

    plt.figure(figsize=(5,3.5))
    max_abs_coef = np.max(np.abs(unique_Mterm_models))
    plt.imshow(unique_Mterm_models[:,idx_non_zero_coefs_Mterm_models].T, origin='lower', 
               cmap = 'RdBu_r', aspect ='auto', vmin = -1.05*max_abs_coef, vmax = 1.05*max_abs_coef)

    # plt.yticks(range(sum(idx_non_zero_coefs_Mterm_models)), np.array(description)[idx_non_zero_coefs_Mterm_models], rotation='0')
    description_2 = [r'$'+descr +'$' for descr in description]
    plt.yticks(range(sum(idx_non_zero_coefs_Mterm_models)), np.array(description_2)[idx_non_zero_coefs_Mterm_models], rotation='0')
    plt.xlabel(str(M)+ ' term models')
    plt.ylabel('terms')
    num_unique_models = unique_Mterm_models.shape[0]
    plt.xlim(-0.5, num_unique_models-0.5)
    plt.xticks(range(num_unique_models), range(1, num_unique_models+1))
    plt.colorbar()
    plt.tight_layout()


def Mterm_models_in_each_Kfold(regression_results, description, lamda_thresh):
    lamda_opt = regression_results['lamdas'][regression_results['lamdas']>=lamda_thresh][0]
    opt_lambda_iter = np.argwhere(regression_results['lamdas'] == lamda_opt)[0,0]

    plt.figure(figsize=(15,4))
    nonzero_coefficients = (np.sum(regression_results['coeff_evol'][:, opt_lambda_iter, :]**2, axis=0) > 0)
    plt.imshow(regression_results['coeff_evol'][:, opt_lambda_iter, nonzero_coefficients], origin='lower', vmin = -2, vmax = 2, cmap = 'RdBu_r')
    description_2 = [r'$'+descr +'$' for descr in description]
    plt.xticks(range(sum(nonzero_coefficients)), np.array(description_2)[nonzero_coefficients], rotation='vertical')
    plt.ylabel('Kth-fold')
    plt.xlabel('terms')
    plt.colorbar()
    plt.tight_layout()


def LS_estimation_for_given_sparsity_pattern(Theta, Y, description, sparsity_pattern):
    # Final unbaised regression on full data
    final_Xi = np.dot(np.linalg.pinv(Theta[:,sparsity_pattern]), Y[:,0])
    print('Nonzero terms:', np.array(description)[sparsity_pattern])
    print('Mean Coefficient values (on full data):', final_Xi)
    
    return final_Xi
