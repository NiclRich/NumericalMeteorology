"""
    diagnostic_rmse(u_exact::AbstractMatrix, u_approx::AbstractMatrix) -> Vector{Float64}

Compute the Root Mean Square Error (RMSE) between the exact solution `u_exact` 
and the approximate solution `u_approx` at each time step.

# Arguments
- `u_exact`: Matrix of the exact solution (space × time).
- `u_approx`: Matrix of the numerical solution (same size as `u_exact`).

# Returns
- A vector of RMSE values, one per time step.
"""
function diagnostic_rmse(u_exact::AbstractMatrix, u_approx::AbstractMatrix)
    @assert size(u_exact) == size(u_approx) "Different size of matrices!"
    M, N = size(u_exact)
    rmse = [sqrt(sum((u_approx[:, n] .- u_exact[:, n]).^2) / M) for n in 1:N]
    return rmse
end


"""
    diagnostic_rm(u_approx::AbstractMatrix) -> Vector{Float64}

Compute the ratio of total mass at each time step to the initial mass.

# Arguments
- `u_approx`: Matrix of the numerical solution (space × time).

# Returns
- A vector of mass ratios for each time step.
"""
function diagnostic_rm(u_approx::AbstractMatrix)
    M, N = size(u_approx)
    initial_mass = sum(u_approx[:, 1])
    rm = [sum(u_approx[:, n]) / initial_mass for n in 1:N]
    return rm
end


"""
    diagnostic_rmp(u_approx::AbstractMatrix) -> Vector{Float64}

Compute the positive mass at each time step (sum of non-negative values).

# Arguments
- `u_approx`: Matrix of the numerical solution (space × time).

# Returns
- A vector of positive mass values for each time step.
"""
function diagnostic_rmp(u_approx::AbstractMatrix)
    M, N = size(u_approx)
    initial_positive_mass = sum(max.(u_approx[:, 1], 0.0))
    rmp = [sum(max.(u_approx[:, n], 0.0)) for n in 1:N]
    return rmp
end

"""
    diagnostic_mdr(u_approx::AbstractMatrix) -> Vector{Float64}

Compute the squared L2 norm (energy) of the solution at each time step.

# Arguments
- `u_approx`: Matrix of the numerical solution (space × time).

# Returns
- A vector of energy values for each time step.
"""
function diagnostic_mdr(u_approx::AbstractMatrix)
    M, N = size(u_approx)
    initial_energy = sum(u_approx[:, 1] .^ 2)
    mdr = [sum(u_approx[:, n] .^ 2) for n in 1:N]
    return mdr
end

"""
    diagnostic_rmax(u_approx::AbstractMatrix) -> Vector{Float64}

Compute the ratio of the maximum value at each time step to the initial range.

# Arguments
- `u_approx`: Matrix of the numerical solution (space × time).

# Returns
- A vector of maximum value ratios for each time step.

"""
function diagnostic_rmax(u_approx::AbstractMatrix)
    M, N = size(u_approx)
    initial_range = maximum(u_approx[:, 1]) - minimum(u_approx[:, 1])
    rmax = [maximum(u_approx[:, n]) / initial_range for n in 1:N]
    return rmax
end


"""
    diagnostic_rmin(u_approx::AbstractMatrix) -> Vector{Float64}

Compute the ratio of the minimum value at each time step to the initial range.

# Arguments
- `u_approx`: Matrix of the numerical solution (space × time).

# Returns
- A vector of minimum value ratios for each time step.

"""
function diagnostic_rmin(u_approx::AbstractMatrix)
    M, N = size(u_approx)
    initial_range = maximum(u_approx[:, 1]) - minimum(u_approx[:, 1])
    rmax = [minimum(u_approx[:, n]) / initial_range for n in 1:N]
    return rmax
end