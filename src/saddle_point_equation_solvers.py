import numpy as np
from scipy import optimize

def find_fixed_points(f, x_min, x_max, n_samples=1000):
    """
    Find fixed points of a function within a given interval.
    
    This function samples the given function at n_samples points within the interval
    [x_min, x_max] and looks for sign changes to detect zero crossings. It then uses
    the Brent method to find the precise location of each root. It also checks the
    endpoints of the interval.
    
    Parameters
    ----------
    f : callable
        The function for which to find fixed points (where f(x) = 0)
    x_min : float
        Lower bound of the interval to search
    x_max : float
        Upper bound of the interval to search
    n_samples : int, optional
        Number of sample points to use. Default is 1000.
        
    Returns
    -------
    numpy.ndarray
        Array of unique fixed points sorted in ascending order
    """
    xs = np.linspace(x_min, x_max, n_samples)
    fs = f(xs)

    roots = []
    # look for sign changes
    for i in range(len(xs)-1):
        if fs[i] == 0:
            roots.append(xs[i])
        elif fs[i]*fs[i+1] < 0:
            root = optimize.root_scalar(
                f, bracket=[xs[i], xs[i+1]],
                method='brentq'
            ).root
            roots.append(root)
    # Check endpoints
    if abs(f(x_min)) < 1e-10:
        roots.append(x_min)
    if abs(f(x_max)) < 1e-10:
        roots.append(x_max)
    return np.unique(np.round(roots, 8))

def find_stable_fixed_points(r, max_theta, saddle_point_equation, phi, lower_bound=-1, upper_bound=1, num_thetas=1e4):
    """
    Find local maxima of the effective free entropy potential using saddle point equation for a range of theta values.
    
    This function scans through theta values and finds stable fixed points of the saddle point equation.
    For each theta, it identifies local maxima and the global maximum of the effective free entropy.
    Assumes that if there are three fixed points, the local maxima are the first and third fixed points.
    This is used to detect phase transitions where the global maximum switches between two local maxima.
    
    Parameters
    ----------
    r : float
        Interaction strength parameter
    max_theta : float
        Maximum value of theta to scan
    saddle_point_equation : callable
        Function that takes (x, theta, r) and returns the saddle point equation
    phi : callable
        Function that takes (x, theta, r) and returns the effective free entropy
    lower_bound : float, optional
        Lower bound of the interval to scan. Default is -1.
    upper_bound : float, optional
        Upper bound of the interval to scan. Default is 1.
        
    Returns
    -------
    max_phi_arr : numpy.ndarray
        Array of maximum effective free entropy values for each theta
    global_maxima : numpy.ndarray
        Array of x values that maximize effective free entropy for each theta  
    local_maxima_arr : numpy.ndarray
        Array of local maxima pairs [max1, max2] for each theta
    thetas : numpy.ndarray
        Array of theta values scanned
    """
    thetas = np.linspace(0.01, max_theta, int(num_thetas))
    max_phi_arr = np.zeros(len(thetas))
    global_maxima = np.zeros(len(thetas))
    local_maxima_arr = np.zeros((len(thetas), 2))

    for i, theta in enumerate(thetas):
        fixed_points = find_fixed_points(lambda x: saddle_point_equation(x, theta, r), lower_bound, upper_bound, 1000)
        if fixed_points.size == 0:
            raise ValueError("No fixed points found for theta: ", theta)
        if len(fixed_points) > 2:
            local_maxima_arr[i] = [fixed_points[0], fixed_points[2]]
        elif len(fixed_points) == 2:
            print(f"Only two fixed points found for r: {r}, theta: {theta}")
            local_maxima_arr[i] = [fixed_points[0], fixed_points[1]]
        else:
            local_maxima_arr[i] = [fixed_points[0], fixed_points[0]]
        phi_values = phi(fixed_points, theta, r)
        max_phi_arr[i] = np.max(phi_values)
        maximizer_index = np.argmax(phi_values)
        global_maximum = fixed_points[maximizer_index]
        global_maxima[i] = global_maximum

    return max_phi_arr, global_maxima, local_maxima_arr, thetas

def find_critical_thetas(local_maxima_arr, global_maxima, thetas):
    """
    Find critical theta values where phase transitions occur in the system.
    
    This function analyzes the local maxima and global maximum arrays to identify three critical theta values:
    1. theta_lower: The theta value where the system first exhibits two distinct local maxima
    2. theta_transition: The theta value where the global maximum switches from one local maximum to the other
    3. theta_upper: The theta value where the system returns to having a single maximum
    
    Parameters:
    -----------
    local_maxima_arr : numpy.ndarray
        Array of shape (n, 2) containing the values of two local maxima for each theta value
    global_maxima : numpy.ndarray
        Array of shape (n,) containing the global maximum value for each theta
    thetas : numpy.ndarray
        Array of theta values corresponding to the maxima arrays
        
    Returns:
    --------
    tuple
        (theta_lower, theta_transition, theta_upper)
        If any transition is not detected, the corresponding value will be np.nan
    """
    threshold = 1e-6
    
    # Find indices where local maxima differ
    diff_indices = np.where(np.abs(local_maxima_arr[:,0] - local_maxima_arr[:,1]) > threshold)[0]
    
    # If no differences detected, return NaN values
    if len(diff_indices) == 0:
        return np.nan, np.nan, np.nan
    
    # Find transition points
    theta_lower = thetas[diff_indices[0]]
    theta_upper = thetas[diff_indices[-1]]
    
    # Find where global maximum switches
    post_transition_indices = np.where(thetas > theta_lower)[0]
    
    if len(post_transition_indices) == 0:
        return theta_lower, np.nan, theta_upper
    
    # Determine which local maximum starts as the global maximum
    first_post_idx = post_transition_indices[0]
    initial_global_is_second = np.abs(global_maxima[first_post_idx] - local_maxima_arr[first_post_idx, 1]) < threshold
    target_col = 0 if initial_global_is_second else 1
    
    # Find the transition point where global maximum switches
    for i in post_transition_indices:
        if np.abs(global_maxima[i] - local_maxima_arr[i, target_col]) < threshold:
            return theta_lower, thetas[i], theta_upper
    
    return theta_lower, np.nan, theta_upper

def find_spinodal_indices(local_maxima_arr):
    """
    Find the indices where the local maxima start to differ and end to differ.
    
    Parameters:
    -----------
    local_maxima_arr : numpy.ndarray
        Array containing local maxima values for different theta values.
        Expected shape is (n, 2) where n is the number of theta values,
        and the columns represent two different local maxima.
    
    Returns:
    --------
    first_diff_idx : int or None
        Index where local maxima start to differ. None if they never differ.
    last_diff_idx : int or None
        Index where local maxima end differing. None if they never differ.
    """

    threshold = 1e-6
    
    # Find indices where local minima differ
    diff_indices = np.where(np.abs(local_maxima_arr[:,0] - local_maxima_arr[:,1]) > threshold)[0]
    if len(diff_indices) == 0:
        return None, None
    first_diff_idx = diff_indices[0]-1
    last_diff_idx = diff_indices[-1]
    return first_diff_idx, last_diff_idx