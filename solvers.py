import numpy as np

def bisection_linesearch(x, d, f, gradf, c0=1e-4, c1=0.9, t0=1.,max_iter=1000):
    alpha = t0
    a = 0
    b = 1e6
    iter = 0
    while iter<max_iter:
        dir_deriv = np.dot(gradf(x), d)
        fval = f(x + alpha * d)
        w1rhs = f(x) + c0 * alpha * dir_deriv
        g = gradf(x + alpha * d)
        w2lhs = np.dot(g, d)
        w2rhs = c1 * dir_deriv
        
        if fval > w1rhs:
            b = alpha
            alpha = (b + a) / 2.0
        elif w2lhs < w2rhs:
            a = alpha
            alpha = np.minimum(2.0 * a, (b + a) / 2.0)
        else:
            break
        iter += 1
        alpha=max(alpha,1e-6)
    # print(f"Iter : {iter}, Step size: {alpha}, Function value: {fval}, Gradient norm: {np.linalg.norm(g)}")
    return alpha, fval, g

def initialize_histories(f, x, gradf):
    function_history = [f(x)]
    gradient_norms = [np.linalg.norm(gradf(x))]
    cumulative_times = [0]
    return function_history, gradient_norms, cumulative_times

def update_histories(function_history, gradient_norms, cumulative_times, fval, g_new):
    function_history.append(fval)
    gradient_norms.append(np.linalg.norm(g_new))
    cumulative_times.append(cumulative_times[-1] + 1)
    return function_history, gradient_norms, cumulative_times

def steepest_descent(x0, f, gradf, c0=1e-4, c1=0.9, t0=1.,grad_tol=1e-6):
    x = x0
    function_history, gradient_norms, cumulative_times = initialize_histories(f, x, gradf)
    
    while gradient_norms[-1] > grad_tol:
        d = -gradf(x)
        alpha, fval, g = bisection_linesearch(x, d, f, gradf, c0, c1, t0)
        # print("SD : grad norm",np.linalg.norm(g))
        x = x + alpha * d
        function_history, gradient_norms, cumulative_times = update_histories(function_history, gradient_norms, cumulative_times, fval, g)
    
    # print("SD : grad norm",np.linalg.norm(gradient_norms[-1]))
    x_sol = x
    return x_sol, function_history, cumulative_times, gradient_norms

def DFP(x0, f, gradf, c0=1e-4, c1=0.9, t0=1.,grad_tol=1e-6):
    x = x0
    n = len(x0)
    B = np.eye(n)
    function_history, gradient_norms, cumulative_times = initialize_histories(f, x, gradf)
    
    while gradient_norms[-1] > grad_tol:
        g = gradf(x)
        d = -np.dot(B, g)
        # t0 = min(1.0, 1.0 / np.linalg.norm(g))
        alpha, fval, g_new = bisection_linesearch(x, d, f, gradf, c0, c1, t0)
        # print("DFP : grad norm",np.linalg.norm(g_new))
        s = alpha * d
        x = x + s
        y = g_new - g
        By = np.dot(B, y)
        B = B - np.outer(By, By) / np.dot(y, By) + np.outer(s, s) / np.dot(y, s)
        function_history, gradient_norms, cumulative_times = update_histories(function_history, gradient_norms, cumulative_times, fval, g_new)
    
    # print("DFP : grad norm",np.linalg.norm(gradient_norms[-1]))

    x_sol = x
    return x_sol, function_history, cumulative_times, gradient_norms

def BFGS(x0, f, gradf, c0=1e-4, c1=0.9, t0=1., grad_tol=1e-6):
    x = x0
    n = len(x0)
    B = np.eye(n)
    function_history, gradient_norms, cumulative_times = initialize_histories(f, x, gradf)
    
    while gradient_norms[-1] > grad_tol:
        g = gradf(x)
        d = -np.dot(B, g)
        alpha, fval, g_new = bisection_linesearch(x, d, f, gradf, c0, c1, t0)
        # print("BFGS : grad norm",np.linalg.norm(g_new))
        s = alpha * d
        x = x + s
        y = g_new - g
        Bs = np.dot(B, s)
        sy = np.dot(s, y)
        M = np.eye(n) - np.outer(s, y) / sy
        B = np.dot(np.dot(M, B), M.T) + np.outer(s, s) / sy
        function_history, gradient_norms, cumulative_times = update_histories(function_history, gradient_norms, cumulative_times, fval, g_new)
    
    # print("BFGS : grad norm",np.linalg.norm(gradient_norms[-1]))
    x_sol = x
    return x_sol, function_history, cumulative_times, gradient_norms

def newton(x0, f, gradf, hessf, c0=1e-4, c1=0.9, t0=1., max_iter=1000, grad_tol=1e-6):
    x = x0
    function_history, gradient_norms, cumulative_times = initialize_histories(f, x, gradf)
    iter_count = 0
    
    while gradient_norms[-1] > grad_tol and iter_count < max_iter:
        g = gradf(x)
        Hk = hessf(x)
        if np.linalg.det(Hk) == 0:
            d = -g
        else:
            H_inv = np.linalg.inv(Hk)
            d = -np.dot(H_inv, g)
        alpha, fval, g_new = bisection_linesearch(x, d, f, gradf, c0, c1, t0)

        # print("Newton : grad norm",np.linalg.norm(g_new))
        x = x + alpha * d
        function_history, gradient_norms, cumulative_times = update_histories(function_history, gradient_norms, cumulative_times, fval, g_new)
        iter_count += 1

    # print("Newton : grad norm",np.linalg.norm(gradient_norms[-1]))
    x_sol = x
    return x_sol, function_history, cumulative_times, gradient_norms
