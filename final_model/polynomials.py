# Solving polynomials

### Solve a 2d matrix of 2nd order polynomials

def solve_square(a_matrix, b_matrix, c_matrix):
    from numpy import ones_like, nan

    upper, lower = nan * ones_like(a_matrix), nan * ones_like(a_matrix)

    D_matrix = (b_matrix**2 - 4 * a_matrix * c_matrix)
    real = (D_matrix > 0) * (a_matrix != 0)

    upper[real] = (-b_matrix[real] + D_matrix[real]**0.5) / (2 * a_matrix[real])
    lower[real] = (-b_matrix[real] - D_matrix[real]**0.5) / (2 * a_matrix[real])

    return upper, lower

### Solve a 2d martic of 3rd order polynomials (only used to solve the quartic equations, so only the real root is returned)

def solve_cubic(a_matrix, b_matrix, c_matrix, d_matrix):
    from numpy import zeros_like, nan, ones_like

    solution = nan * ones_like(a_matrix)

    offset_matrix = b_matrix / (3 * a_matrix)

    p_matrix = c_matrix / a_matrix - b_matrix**2 / (3 * a_matrix**2)
    q_matrix = 2 * b_matrix**3 / (27 * a_matrix**3) - b_matrix * c_matrix / (3 * a_matrix**2) + d_matrix / a_matrix

    D_matrix = q_matrix**2 / 4 + p_matrix**3 / 27

    one_real = D_matrix > 0
    two_real = D_matrix == 0
    three_real = D_matrix < 0

    # One real solution
    u1_matrix = -q_matrix[one_real] / 2 + D_matrix[one_real]**0.5
    u2_matrix = -q_matrix[one_real] / 2 - D_matrix[one_real]**0.5

    u1_pos, u1_neg = u1_matrix > 0, u1_matrix < 0
    u2_pos, u2_neg = u2_matrix > 0, u2_matrix < 0

    #--- Ensure cuberoot returns a real rather than complex number
    cr_u1_matrix = zeros_like(u1_matrix)
    cr_u2_matrix = zeros_like(u2_matrix)

    cr_u1_matrix[u1_pos] = u1_matrix[u1_pos]**(1/3)
    cr_u1_matrix[u1_neg] = -(-u1_matrix[u1_neg])**(1/3)

    cr_u2_matrix[u2_pos] = u2_matrix[u2_pos]**(1/3)
    cr_u2_matrix[u2_neg] = -(-u2_matrix[u2_neg])**(1/3)

    solution[one_real] = cr_u1_matrix + cr_u2_matrix

    # Two real solutions, only need one
    solution[two_real] = 3 * q_matrix[two_real] / p_matrix[two_real]

    # Three real solutions, only need one

    cubed_matrix = -q_matrix[three_real] / 2 + (q_matrix[three_real]**2 / 4 + p_matrix[three_real]**3 / 27)
    C_matrix = zeros_like(cubed_matrix)

    C_matrix[cubed_matrix > 0] = cubed_matrix[cubed_matrix > 0]**(1/3)
    C_matrix[cubed_matrix < 0] = -(-cubed_matrix[cubed_matrix < 0])**(1/3)

    solution[three_real] = C_matrix - p_matrix[three_real] / (3 * C_matrix)

    solution += -offset_matrix

    return solution

### Solve a matrix of 4th order polynomials (return all roots (up to 4), if none return 0)

def solve_simple_quartic(a_array, b_array, c_array):
    from numpy import nan

    d_array = b_array**2 - 4 * a_array * c_array
    d_array[d_array < 0] = nan

    inner_plus = (-b_array + d_array**0.5) / (2 * a_array)
    inner_minus = ((-b_array - d_array**0.5) / (2 * a_array))

    inner_plus[inner_plus < 0] = nan
    inner_minus[inner_minus < 0] = nan

    return inner_plus**0.5, inner_minus**0.5, -inner_minus**0.5, -inner_plus**0.5

def solve_quartic(a_matrix, b_matrix, c_matrix, d_matrix, e_matrix):
    from numpy import ones_like, nan

    sol1, sol2, sol3, sol4 = nan * ones_like(a_matrix), nan * ones_like(a_matrix), nan * ones_like(a_matrix), nan * ones_like(a_matrix)
    offset_matrix = b_matrix / (4 * a_matrix)

    # Special cases

    not_cubic = a_matrix != 0
    a_matrix[a_matrix == 0] = nan

    simple = (b_matrix == 0) * (d_matrix == 0)
    not_simple = simple != 1

    simple_solutions = solve_simple_quartic(a_matrix[simple], c_matrix[simple], e_matrix[simple])
    sol1[simple] = simple_solutions[0]
    sol2[simple] = simple_solutions[1]
    sol3[simple] = simple_solutions[2]
    sol4[simple] = simple_solutions[3]
    
    # Redefine

    p_matrix = (8 * a_matrix * c_matrix - 3 * b_matrix**2) / (8 * a_matrix**2)
    q_matrix = (b_matrix**3 - 4 * a_matrix * b_matrix * c_matrix + 8 * a_matrix**2 * d_matrix) / (8 * a_matrix**3)
    r_matrix = (-3 * b_matrix**4 + 16 * a_matrix * (b_matrix**2) * c_matrix - 64 * (a_matrix**2) * b_matrix * d_matrix + 256 * (a_matrix**3) * e_matrix) / (256 * a_matrix**4)

    m_matrix = solve_cubic(ones_like(a_matrix), p_matrix, p_matrix**2 / 4 - r_matrix, -q_matrix**2 / 8)
    m_matrix[m_matrix <= 0] = nan

    D_plus = -(2 * p_matrix + 2 * m_matrix + q_matrix * 2**0.5 / m_matrix**0.5)
    D_minus = -(2 * p_matrix + 2 * m_matrix - q_matrix * 2**0.5 / m_matrix**0.5)

    D_plus[D_plus < 0] = nan
    D_minus[D_minus < 0] = nan

    sol1[not_simple] = 0.5 * ((2 * m_matrix[not_simple])**0.5 + D_plus[not_simple]**0.5) - offset_matrix[not_simple]
    sol2[not_simple] = 0.5 * ((2 * m_matrix[not_simple])**0.5 - D_plus[not_simple]**0.5) - offset_matrix[not_simple]
    sol3[not_simple] = -0.5 * ((2 * m_matrix[not_simple])**0.5 - D_minus[not_simple]**0.5) - offset_matrix[not_simple]
    sol4[not_simple] = -0.5 * ((2 * m_matrix[not_simple])**0.5 + D_minus[not_simple]**0.5) - offset_matrix[not_simple]

    #solutions = stack([sol1, sol2, sol3, sol4], axis = 2)
    #solutions = flip(sort(solutions, axis = 2), axis = 2)

    return sol1, sol2, sol3, sol4

# Shape coefficients

### Return ellipsoid coefficient matrices

def ellipsoid_coefficients(x_matrix, z_matrix, a, b, alpha: float=0, beta: float=0, offset: float=0):
    from numpy import sin, cos, deg2rad, ones_like

    sina, cosa = sin(deg2rad(alpha)), cos(deg2rad(alpha))
    sinb, cosb = sin(deg2rad(beta)), cos(deg2rad(beta))

    cx_matrix = x_matrix * cosa * cosb + z_matrix * cosa * sinb
    cy_matrix = x_matrix * sina * cosb + z_matrix * sina * sinb
    cz_matrix = -x_matrix * sinb + z_matrix * cosb

    A = (b**2 * sina**2 + a**2 * cosa**2) * ones_like(x_matrix)
    B = 2 * (a**2 * cosa * cy_matrix - b**2 * sina * cx_matrix + offset * b**2 * sina)
    C = b**2 * cx_matrix**2 + a**2 * (cy_matrix**2 + cz_matrix**2) - a**2 * b**2 + b**2 * offset**2 - 2 * offset * b**2 * cx_matrix

    return A, B, C

### Return torus coefficient matrices

def torus_coefficients(x_matrix, z_matrix, a, b, R, alpha = 0, beta = 0):
    from numpy import sin, cos, deg2rad, ones_like

    sina, cosa = sin(deg2rad(alpha)), cos(deg2rad(alpha))
    sinb, cosb = sin(deg2rad(beta)), cos(deg2rad(beta))

    cx_matrix = x_matrix * cosa * cosb + z_matrix * cosa * sinb
    cy_matrix = x_matrix * sina * cosb + z_matrix * sina * sinb
    cz_matrix = -x_matrix * sinb + z_matrix * cosb

    K1 = cz_matrix**2 / b**2 + R**2 / b**2 - 1
    K2 = (4 * R**2 * cz_matrix**2) /b**4
    K3 = 2 * K1 / b**2 - 4 * R**2 / b**4

    A = (sina**4 / a**4 + cosa**4 / b**4 + 2 * sina**2 * cosa**2 / (a**2 * b**2)) * ones_like(x_matrix)
    B = -4 * sina**3 * cx_matrix / a**4 + 4 * cosa**3 * cy_matrix / b**4 + 4 * (sina**2 * cosa * cy_matrix - sina * cosa**2 * cx_matrix) / (a**2 * b**2)
    C = 6 * sina**2 * cx_matrix**2 / a**4 + 6 * cosa**2 * cy_matrix**2 / b**4 + 2 * (sina**2 * cy_matrix**2 + cosa**2 * cx_matrix**2 - 4 * sina * cosa * cx_matrix * cy_matrix) / (a**2 * b**2) + 2 * K1 * sina**2 / a**2 + K3 * cosa**2
    D = -4 * sina * cx_matrix**3 / a**4 + 4 * cosa * cy_matrix**3 / b**4 + 4 * (cosa * cx_matrix**2 * cy_matrix - sina * cx_matrix * cy_matrix**2) / (a**2 * b**2) - 4 * K1 * sina * cx_matrix / a**2 + 2 * K3 * cosa * cy_matrix
    E = cx_matrix**4 / a**4 + cy_matrix**4 / b**4 + 2 * cx_matrix**2 * cy_matrix**2 / (a**2 * b**2) + 2 * K1 * cx_matrix**2 / a**2 + K3 * cy_matrix**2 + K1**2 - K2

    return A, B, C, D, E

# Interceptions with shapes

def ellipsoid_intercepts(x_matrix, z_matrix, a, b, alpha: float=0, beta: float=0):
    from numpy import stack

    A, B, C = ellipsoid_coefficients(x_matrix, z_matrix, a, b, alpha = alpha, beta = beta)
    sol1, sol2 = solve_square(A, B, C)

    return stack([sol1, sol2], axis=2)

def void_intercepts(x_matrix, z_matrix, a, b, offset, alpha: float=0, beta: float=0):
    from numpy import stack, stack, isnan, isfinite

    Al, Bl, Cl = ellipsoid_coefficients(x_matrix, z_matrix, a, b, alpha=alpha, beta=beta, offset=offset)
    Ar, Br, Cr = ellipsoid_coefficients(x_matrix, z_matrix, a, b, alpha=alpha, beta=beta, offset=-offset)

    upper, lower = solve_square(Al, Bl, Cl)
    ur, lr = solve_square(Ar, Br, Cr)
    
    case1 = (ur > upper) + isfinite(ur) * isnan(upper)
    case2 = (lr < lower) + isfinite(lr) * isnan(lower)

    upper[case1] = ur[case1]
    lower[case2] = lr[case2]

    return stack([upper, lower], axis=2)

def torus_intercepts(x_matrix, z_matrix, a, b, R, alpha: float=0, beta: float=0):
    from numpy import stack

    A, B, C, D, E = torus_coefficients(x_matrix, z_matrix, a, b, R, alpha=alpha, beta=beta)
    sol1, sol2, sol3, sol4 = solve_quartic(A, B, C, D, E)

    return stack([sol1, sol2, sol3, sol4], axis=2)

# Interpret intercepts and return distances

def get_overlap(body):
    from numpy import nan_to_num, zeros_like
    i, overlap = 0, zeros_like(body[:,0])
    while i < body.shape[-1]:
        upper, lower = body[:,i], body[:,i+1]
        overlap += nan_to_num(upper - lower)
        i += 2
    return overlap

def subtract_edge(body, edge):
    from numpy import isnan, ones_like, nan, stack, copy, isfinite

    y1, y2 = body[:,0], body[:,1]
    e1, e2 = edge[:,0], edge[:,1]

    is_edge, isnt_edge = isfinite(e1), isnan(e1)
    is_body, isnt_body = isfinite(y1), isnan(y1)
    
    cond1 = (y1 > e1)
    cond2 = (e1 >= y1)
    cond3 = (y2 > e1)
    cond4 = (e1 >= y2)
    cond5 = (y1 >= e2)
    cond6 = (e2 > y1)
    cond7 = (y2 >= e2)
    cond8 = (e2 > y2)

    case1 = cond3
    case2 = cond1 * cond4 * cond7
    case3 = cond1 * cond8
    case4 = cond2 * cond5 * cond8
    case5 = cond6
    case6 = cond2 * cond7

    new_y1, new_y2, new_y3, new_y4 = nan * ones_like(y1), nan * ones_like(y1), nan * ones_like(y1), nan * ones_like(y1)
    new_e1, new_e2, new_e5, new_e6 = nan * ones_like(e1), nan * ones_like(e1), nan * ones_like(e1), nan * ones_like(e1)

    new_e3, new_e4 = copy(e1), copy(e2)
    case0 = isnt_edge * is_body
    if case0.any():
        new_y3[case0], new_y4[case0] = y1[case0], y2[case0]
        new_e1[case0], new_e2[case0], new_e3[case0], new_e4[case0], new_e5[case0], new_e6[case0] = y1[case0], y2[case0], y1[case0], y2[case0], y1[case0], y2[case0]
    #new_y3[isnan(e1)], new_y4[isnan(e1)] = y1[isnan(e1)], y2[isnan(e1)]
    #new_e3[isnan(y1)], new_e4[isnan(y1)] = e1[isnan(y1)], e2[isnan(y1)]

    if case1.any():
        new_y1[case1], new_y2[case1] = y1[case1], y2[case1]
        new_e1[case1], new_e2[case1] = y1[case1], y2[case1]
    if case2.any():
        new_y1[case2], new_y2[case2] = y1[case2], e1[case2]
        new_e1[case2], new_e2[case2] = y1[case2], e1[case2]
    if case3.any():
        new_y1[case3], new_y2[case3], new_y3[case3], new_y4[case3] = y1[case3], e1[case3], e2[case3], y2[case3]
        new_e1[case3], new_e2[case3], new_e5[case3], new_e6[case3] = y1[case3], e1[case3], e2[case3], y2[case3]
    if case4.any():
        new_y3[case4], new_y4[case4] = e2[case4], y2[case4]
        new_e5[case4], new_e6[case4] = e2[case4], y2[case4]
    if case5.any():
        new_y3[case5], new_y4[case5] = y1[case5], y2[case5]
        new_e5[case5], new_e6[case5] = y1[case5], y2[case5]
    if case6.any():
        pass # No changes necessary

    return stack([new_y1, new_y2, new_y3, new_y4], axis=1), stack([new_e1, new_e2, new_e3, new_e4, new_e5, new_e6], axis=1)

def simplify_edges(edges):
    from numpy import stack, nan, copy, nan_to_num, inf, sort, isnan

    simple_edges = []
    N = edges.shape[-1]

    idx = 0
    while idx < N:
        current = edges[:,idx]
        if idx == 0 or idx == N-1:
            simple_edges.append(current)
            idx += 1
        else:
            next = edges[:,idx+1]
            is_overlap = (next >= current)

            new_current, new_next = copy(current), copy(next)

            if is_overlap.any():
                new_current[is_overlap] = nan
                new_next[is_overlap] = nan

            simple_edges.append(new_current)
            simple_edges.append(new_next)

            idx += 2

    simple_edges = stack(simple_edges, axis=1)

    # Sorting, i.e. sending NaNs to the back

    simple_edges = nan_to_num(simple_edges, nan=-inf)
    simple_edges = sort(simple_edges)[:,::-1]
    simple_edges[simple_edges == -inf] = nan

    # Removing redundant layers

    result = []
    for idx in range(N):
        layer = simple_edges[:,idx]
        if isnan(layer).all() == False:
            result.append(layer)
    result = stack(result, axis=1)

    return result

def subtract_all_edges(old_body, edges):
    from numpy import stack

    final_body = []
    final_edges = []

    N = edges.shape[-1]

    idx = 0
    while idx < N:
        new_body, new_edges = subtract_edge(old_body, edges[:,idx:idx+2])

        y1, y2, y3, y4 = new_body[:,0], new_body[:,1], new_body[:,2], new_body[:,3]
        e1, e2, e3, e4, e5, e6 = new_edges[:,0], new_edges[:,1], new_edges[:,2], new_edges[:,3], new_edges[:,4], new_edges[:,5]
        
        for y in [y1, y2]:
            final_body.append(y)
        for e in [e1, e2, e3, e4, e5, e6]:
            final_edges.append(e)

        old_body = stack([y3, y4], axis=1)
        idx += 2

    for y in [y3, y4]:
        final_body.append(y)   

    final_edges = stack(final_edges, axis=1)
    final_body = stack(final_body, axis=1)

    final_edges = simplify_edges(final_edges)

    return final_body, final_edges

def produce_overlaps(torus, voids, ellipsoids):
    from numpy import isfinite, logical_or, stack, zeros

    all_bodies = [torus, voids] + ellipsoids

    for i, body in enumerate(all_bodies):
        for j in range(body.shape[2]):
            layer = body[:,:,j]

            if i == 0 and j == 0:
                mask = isfinite(layer)
                Nx, Ny = layer.shape
            else:
                mask = logical_or(mask, isfinite(layer))

    masked_bodies = []
    for body in all_bodies:
        layers = []
        for j in range(body.shape[2]):
            layer = body[:,:,j]
            layers.append(layer[mask])
        masked_bodies.append(stack(layers, axis=1))
        
    masked_overlaps = []
    for idx, masked_body in enumerate(masked_bodies):
        if idx == 0:
            trimmed_body = masked_body
            old_edges = masked_body
        else:
            trimmed_body, new_edges = subtract_all_edges(masked_body, old_edges)
            old_edges = new_edges

        masked_overlaps.append(get_overlap(trimmed_body))

    overlaps = []
    for mo in masked_overlaps:
        overlap = zeros(shape=(Nx, Ny))
        overlap[mask] = mo
        overlaps.append(overlap)
    
    return overlaps