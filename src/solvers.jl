"""
    module Solvers

A collection of one‐dimensional linear advection solvers implementing various
finite‐difference and finite‐volume schemes. Each solver advances an initial
scalar field in time according to

u_t + c * u_x = 0

on a uniform grid with optional periodic boundary conditions.
Provided Solvers

    donor_cell_advection    First‐order upwind (“donor‐cell”) scheme

    leap_frog_RA_advection  Leap‐frog in time with Robert–Asselin filter

    lax_wendroff_advection  Second‐order Lax–Wendroff scheme

    van_leer_advection      Second‐order flux‐limited van Leer scheme

    MOL_RK4_advection       Method‐of‐Lines with fourth‐order spatial differences and RK4 in time

# Usage

using LinearAdvectionSolvers

# Define a problem container, e.g.:
prob = LinearAdvectionProblem(...)

# Choose one of the solvers:
U1 = donor_cell_advection(prob)
U2 = lax_wendroff_advection(prob)
U3 = MOL_RK4_advection(prob)

# Notes

 - All solvers expect prob.grid_size > 2 (and > 5 for fourth‐order MOL_RK4).

 - Periodic boundary conditions are applied when prob.bc == :periodic.

 - Reference implementations assume a uniform mesh and constant advection speed.
"""
module Solvers

include("ConfigLoader.jl")
#include("LinearAdvection.jl")

using SparseArrays
using LinearAlgebra

export donor_cell_advection, leap_frog_RA_advection, lax_wendroff_advection, van_leer_advection, MOL_RK4_advection

# Definition of the Linear Advection Problem
"""
    LinearAdvectionProblem

A data structure representing a 1D linear advection problem to be solved numerically.

This struct encapsulates all necessary parameters for defining the spatial domain,
temporal resolution, initial conditions, boundary conditions, and advection speed.
It is intended for use with numerical solvers of the linear advection equation:

``math
u_t + c * u_x = 0
``
Fields

    name::String
    Descriptive name for the problem instance.

    dx::AbstractFloat
    Grid spacing in space (used for numerical schemes).

    dx_max::AbstractFloat
    Maximum allowable spatial step size.

    domain::Tuple{AbstractFloat, AbstractFloat}
    Tuple specifying the physical domain (x_min, x_max).

    grid_cell_boundary::Vector{AbstractFloat}
    Vector of grid cell boundaries; typically of length grid_size + 1.

    grid_size::Integer
    Number of spatial cells (one less than the number of boundaries).

    dt::AbstractFloat
    Time step size.

    dt_max::AbstractFloat
    Maximum allowable time step.

    final_time::AbstractFloat
    Final simulation time.

    time_steps::Integer
    Number of time steps to reach final_time.

    initial::Function
    Initial condition function, typically u₀(x).

    bc::Union{Symbol, NamedTuple}
    Boundary conditions. Accepts a Symbol currently supports only :periodic
    or a NamedTuple (e.g., (left = 0.0, right = 1.0)).

    advection_speed::AbstractFloat
    Constant advection speed c in the equation.

    cfl::AbstractFloat
    Courant–Friedrichs–Lewy number used to ensure stability:
    cfl = |c| * dt / dx.

# Example

prob = LinearAdvectionProblem(
    name = "Gaussian pulse",
    dx = 0.01,
    dx_max = 0.01,
    domain = (0.0, 1.0),
    grid_cell_boundary = collect(0.0:0.01:1.0),
    grid_size = 100,
    dt = 0.001,
    dt_max = 0.001,
    final_time = 1.0,
    time_steps = 1000,
    initial = x -> exp(-100 * (x - 0.5)^2),
    bc = :periodic,
    advection_speed = 1.0,
    cfl = 0.1
)

"""
struct LinearAdvectionProblem
    name::String                                # name of the problem
    dx::AbstractFloat                           # domain specifification
    dx_max::AbstractFloat
    domain::Tuple{AbstractFloat, AbstractFloat}
    grid_cell_boundary::Vector{AbstractFloat} # stores the boundaries
    grid_size::Integer # number of cells, one less than the size of the discretized version
    dt::AbstractFloat                           # time specifification
    dt_max::AbstractFloat
    final_time::AbstractFloat 
    time_steps::Integer
    initial::Function                           # equation specifification
    bc::Union{Symbol,NamedTuple}
    advection_speed::AbstractFloat
    cfl::AbstractFloat
end

# Helper functions for creating the LinearAdvectionProblem
function make_grid(xmin, xmax, dx_max)

    grid_size = ceil(Integer, (xmax - xmin) / dx_max)
    x = collect(range(xmin, xmax, length=grid_size + 1))
    dx = (xmax - xmin) / grid_size

    return x, dx, grid_size

end

function calc_timestep(final_time, dt_max)
    time_steps = ceil(Integer, final_time / dt_max)
    dt = final_time / time_steps

    return time_steps, dt
end

# 
function LinearAdvectionProblem(problem_conf::Dict)
    xmin, xmax = problem_conf["domain"]

    # convert automatically to floating point numbers
    xmin = convert(AbstractFloat, xmin)
    xmax = convert(AbstractFloat, xmax)
    @assert xmin < xmax "Wrong domain specifification"
    domain = (xmin, xmax)
    dx_max = problem_conf["dx_max"]
    dx_max = convert(AbstractFloat, dx_max)

    # create a grid
    grid_cell_boundary, dx, grid_size = make_grid(xmin, xmax, dx_max)

    # convert time values
    dt_max = problem_conf["dt_max"]
    dt_max = convert(AbstractFloat, dt_max)
    final_time = problem_conf["final_time"]
    final_time = convert(AbstractFloat, final_time)

    @assert final_time > 0.0 "Final time is in the past"
    @assert dt_max > 0.0 "Time step dt_max is negative"

    time_steps, dt = calc_timestep(final_time, dt_max)

    boundary_condition = :periodic # fixed for the moment, can be extended later

    advection_speed = problem_conf["advection_speed"]
    advection_speed = convert(AbstractFloat, advection_speed)

    cfl = advection_speed * dt / dx

    if abs(cfl) >= 1
        @warn "CFL number $cfl has absolute value greater than 1"
    end

    return LinearAdvectionProblem(problem_conf["name"], dx, dx_max, domain, 
    grid_cell_boundary, grid_size, dt, dt_max, final_time, time_steps,
    problem_conf["initial"], boundary_condition, advection_speed, cfl)

end


# flux computation
"""
    compute_flux_LW(u::AbstractVector{T}, advection_speed::T, dt::T, dx::T) where T

Compute the Lax–Wendroff numerical flux for a one‐dimensional linear advection equation.

# Arguments
- `u::AbstractVector{T}`  
  The vector of conserved quantities at the current time step.
- `advection_speed::T`  
  The constant advection speed `c`.
- `dt::T`  
  The time step size `\\Delta t`.
- `dx::T`  
  The spatial grid spacing `\\Delta x`.

# Returns
- `flux::Vector{T}`  
  A vector of numerical fluxes computed at each cell interface using the Lax–Wendroff scheme:
  ```math
    F_{i+\tfrac12} = \tfrac12\\,c\\,(u_i + u_{i+1})
    - \tfrac12\\,c\\,\\mathrm{CFL}\\,(u_{i+1} - u_i),
    \\quad\\mathrm{CFL} = \frac{c\\,\\Delta t}{\\Delta x}.
  ```

# Notes
- Implements periodic boundary conditions via `circshift(u, -1)`.
- Assumes uniform grid and constant advection speed.
- Stability requires ``\\mathrm{CFL} = c\\,\\Delta t / \\Delta x \\le 1``.

# Example
```julia
u0 = sin.(2π .* range(0, 1, length=100))
c   = 1.0
dt  = 0.01
dx  = 1/100
flux = compute_flux_LW(u0, c, dt, dx)

"""
function compute_flux_LW(u::AbstractVector{T}, advection_speed::T, dt::T, dx::T) where T
    uR = circshift(u, -1)
    cfl = advection_speed * dt / dx
    return (0.5 * advection_speed) .* (u .+ uR) .- (0.5 * advection_speed * cfl) .*(uR .- u)
end

"""
    compute_flux_upwind(u::AbstractVector{T}, advection_speed::T) where T

Compute the numerical flux for a linear advection equation using a first‑order upwind scheme.

# Arguments
- `u::AbstractVector{T}`  
  Vector of conserved quantities at discrete grid points.
- `advection_speed::T`  
  Scalar advection speed. Determines the direction of upwinding:
  - If `advection_speed ≥ 0`, flux is taken from the current cell.
  - If `advection_speed < 0`, flux is taken from the downstream cell (circularly shifted).

# Returns
- `flux::AbstractVector{T}`  
  A vector of the same size as `u`, giving the upwind flux at each grid point.

# Notes
- Implements periodic boundary conditions via `circshift(u, -1)` for negative speeds.
- Suitable for uniform grids and constant advection speed.

# Examples
```julia
julia> u = [1.0, 2.0, 3.0];

julia> compute_flux_upwind(u, 0.5)
3-element Vector{Float64}:
 0.5
 1.0
 1.5

julia> compute_flux_upwind(u, -0.5)
3-element Vector{Float64}:
 -1.0
 -1.5
 -0.5

"""
function compute_flux_upwind(u::AbstractVector{T}, advection_speed::T) where T
    if advection_speed >= 0
        return advection_speed * u
    else
        return advection_speed * circshift(u, -1)
    end
end

# function slope ratio
"""
    compute_slope_ratio(u::AbstractVector{T}, advection_speed::T) where {T<:AbstractFloat}

Compute a slope ratio vector `r` for flux limiting in a one-dimensional advection scheme.

# Description
For each element `i` of the input vector `u`, this function computes
a ratio of successive differences (slopes) in the upwind direction,
scaled by the sign of the denominator to preserve monotonicity and
guard against division by zero. This ratio is used in
flux‐limiter algorithms

# Arguments
- `u::AbstractVector{T}`  
  Input field values at discrete grid points.
- `advection_speed::T`  
  Scalar advection velocity. Its sign determines the upwind direction:
  - `> 0`: use left‐hand difference  
  - `< 0`: use right‐hand difference  
  - `== 0`: yields a zero ratio vector

# Returns
- `r::Vector{T}`  
  A vector of the same length as `u`, containing the computed slope
  ratios. Each entry is given by
  ```julia
  r = sign(Δu_I) * (Δ_upwind[i]) / (|Δu_I| + ε)
  ```
where I is the index as in [1] and ε is the smallest positive 
floating‐point value of type T. This prevents a zero division
error but does only affect very small values smaller than 1e-290,
(for Float64 type) otherwise it does not affect the analytical result.
# Notes

    Periodic boundary conditions are enforced via circshift.

    Division by zero is avoided by adding floatmin(T) to the denominator.

# Examples

julia> u = [0.0, 1.0, 2.0, 1.0, 0.0]
julia> compute_slope_ratio(u, 1.0)
5-element Vector{Float64}:
 0.0
 1.0
 –1.0
 –1.0
  0.0

julia> compute_slope_ratio(u, -2.5)
5-element Vector{Float64}:
 0.0
 1.0
 –1.0
 –1.0
  0.0

# References
[1] LeVeque, Randall (2002), Finite Volume Methods for Hyperbolic Problems, Cambridge University Press.
"""
function compute_slope_ratio(u::AbstractVector{T}, advection_speed::T) where {T<:AbstractFloat}

    # used by venque
    uL = circshift(u,  1)   # u_{i-1}
    uR = circshift(u, -1)   # u_{i+1}

    delta_denominator = uR - u

    if advection_speed > 0
        delta_numerator = uL - circshift(uL, 1)
    elseif advection_speed < 0
        delta_numerator = uR - u
    else
        delta_numerator = zero(u)
    end
    # make it robust against zero division
    r = @. sign(delta_denominator) * delta_numerator / (delta_denominator + floatmin(T))

    return r
end

# Limiter functions
"""
    limiter_vanleer(r::T) where {T<:AbstractFloat}

Compute the Van Leer flux/slope limiter for a given ratio of successive gradients.

# Arguments
- `r::T`: Ratio of successive solution gradients (typically Δu₁ / Δu₀). Can be any real number.

# Returns
- `ϕ::T`: The Van Leer limiter value, in the range [0, 2] used for FDM or FVM schemes.

# Notes
- Ensures total variation diminishing (TVD) behavior.
- For `r = 0`, returns 0; for `r → ∞`, returns 2.

# Examples
```julia
julia> limiter_vanleer(0.5)
0.6666666666666666

julia> limiter_vanleer(1.0)
1.0

julia> limiter_vanleer(-1.0)
0.0

"""
function limiter_vanleer(r::T) where {T<:AbstractFloat}
    return (r + abs(r)) / (1+ abs(r))
end


# functions for the discretization for the Method of Lines (MOL)
"""
    advection_rhs(u, difference_operator, advection_speed) -> Vector{T}

Compute the discretized right-hand side of the linear advection equation
for the Method of Lines.

# Arguments
- `u::Vector{T}`  
  The state vector of field values at discrete spatial points.
- `difference_operator::AbstractMatrix{T}`  
  Finite-difference matrix approximating the spatial derivative (e.g., upwind or central difference).
  This matrix incorporates already the boundary conditions like periodic bc or Neumann bc.
- `advection_speed::T`  
  Constant advection velocity. Positive values correspond to transport in the positive spatial direction.

# Returns
- `Vector{T}`  
  The time derivative ``\\dfrac{du}{dt}`` given by ``-c \\, D\\,u``, where `c` is the advection speed and `D` is the difference operator.

"""
function advection_rhs(u::Vector{T}, difference_operator::AbstractMatrix{T}, advection_speed::T) where {T<:AbstractFloat}
    return -advection_speed .* difference_operator * u
end

"""
    rk4_step(u::Vector{T}, prob::LinearAdvectionProblem) where {T<:AbstractFloat}

Perform a single time‐step update using the classical fourth‐order Runge–Kutta (RK4)
method for a linear advection problem.

# Arguments
- `u::Vector{T}`  
  Current solution values at the spatial grid points (`T<:AbstractFloat`).

- `prob::LinearAdvectionProblem`  
     Definition of the linear advection problem.
- `advection_rhs::Function`
    function for the discretized spatial derivative

# Returns
- `Vector{T}`  
  New solution vector after advancing by one time step.

# Notes
- Requires an `advection_rhs(u)` function that computes the right‐hand side (discretization of the 
spatial derivative) of the advection equation.
- The update formula is:

u_{n+1} = u_n + dt/6 * (k1 + 2k2 + 2k3 + k4)

where k1…k4 are the standard RK4 stage derivatives.

# Examples
```julia
# Define problem with dt = 0.01
prob = LinearAdvectionProblem(dt = 0.01, ...)

# Initial condition vector
u0 = sin.(range(0, 2π, length=100))

# Advance one step
u1 = rk4_step(u0, prob)

"""
function rk4_step(u::Vector{T}, prob::LinearAdvectionProblem, advection_rhs::Function) where {T<:AbstractFloat}
    k1 = advection_rhs(u)
    k2 = advection_rhs(u .+ 0.5 * prob.dt * k1)
    k3 = advection_rhs(u .+ 0.5 * prob.dt * k2)
    k4 = advection_rhs(u .+ prob.dt * k3)
    return @. u + prob.dt/6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
end

# schemes for solving the advection equation
"""
    donor_cell_advection(prob::LinearAdvectionProblem) -> Matrix{Float64}

Compute the numerical solution to a 1D linear advection problem using the donor‐cell (upwind) finite‐difference scheme.

# Arguments
- `prob::LinearAdvectionProblem`
  - A problem descriptor containing:
    - `grid_size::Int` : number of spatial cells
    - `time_steps::Int` : number of time steps
    - `grid_cell_boundary::AbstractVector{<:Real}` : cell boundary coordinates
    - `initial::Function` : initial condition function, called on cell centers
    - `advection_speed::Real` : constant advection velocity
    - `cfl::Real` : Courant–Friedrichs–Lewy number
    - `bc::Symbol` : boundary condition; either `:periodic` or non‐periodic

# Returns
- `U::Matrix{Float64}`  
  A `(grid_size × time_steps)` array where each column `U[:, k]` holds the numerical solution at time step `k`.

# Notes
- Uses an upwind differencing stencil, choosing upwind direction based on the sign of `advection_speed`.
- Supports periodic boundary conditions when `prob.bc == :periodic`.
- Efficiency is improved by assembling a sparse iteration matrix once and re‐using it.

# Example
```julia
# Define a problem on [0,1] with 100 cells and 200 time steps
prob = LinearAdvectionProblem(...)

U = donor_cell_advection(prob)
@show size(U)  # (100, 200)

"""
function donor_cell_advection(prob::LinearAdvectionProblem)
    
    # upwind scheme
    U = zeros(Float64, prob.grid_size, prob.time_steps) # storage for numerical approximation
    cell_centers = prob.grid_cell_boundary[1:end-1] + diff(prob.grid_cell_boundary) * 0.5
    U[:, 1] = prob.initial.(cell_centers)

    # positive advection speed
    if prob.advection_speed >= 0
        main_diagonal = (1 - prob.cfl) * ones(prob.grid_size)
        lower_diagonal = prob.cfl * ones(prob.grid_size-1)
    
        I = [collect(2:prob.grid_size); collect(1:prob.grid_size)]  # row indices
        J = [collect(1:prob.grid_size-1); collect(1:prob.grid_size)]  # col indices
        entries = [lower_diagonal; main_diagonal]
    
        if prob.bc == :periodic # insert periodic boundary conditions
            I = [I; prob.grid_size]
            J = [J; 1]
            entries = [entries; prob.cfl]
        end
        iteration_matrix = sparse(I, J, entries)
    else
        main_diagonal = (1 + prob.cfl) * ones(prob.grid_size)
        upper_diagonal = - prob.cfl * ones(prob.grid_size-1)
    
        I = [collect(1:prob.grid_size-1); collect(1:prob.grid_size)]  # row indices
        J = [collect(2:prob.grid_size); collect(1:prob.grid_size)]  # col indices
        entries = [upper_diagonal; main_diagonal]
    
        if prob.bc == :periodic # insert boundary conditions
            I = [I; prob.grid_size]
            J = [J; 1]
            entries = [entries; -prob.cfl]
        end
        iteration_matrix = sparse(I, J, entries)
    end
    
    
    for k in 2:prob.time_steps
        U[:,k] = iteration_matrix * U[:, k-1]
    end
    return U

end

"""
    leapfrog_ra_advection(prob::LinearAdvectionProblem, gamma::AbstractFloat)

Compute the numerical solution of a linear advection problem using the leap‐frog
method with a Robert Asselin filter.

# Arguments
- `prob::LinearAdvectionProblem`: Problem definition, with fields
  - `grid_size::Int`: Number of spatial cells.
  - `time_steps::Int`: Number of time steps to compute.
  - `grid_cell_boundary::Vector{Float64}`: Coordinates of cell boundaries.
  - `cfl::Float64`: CFL number, controls the stability
  - `initial::Function`: Function giving the initial condition at a point.
  - `bc::Symbol`: Boundary condition (`:periodic` supported).
- `gamma::AbstractFloat`: Artificial‐diffusion coefficient (Robert Asselin filter).

# Returns
- `U::Matrix{Float64}`: A `grid_size × time_steps` matrix containing the solution 
  at each cell (rows) and time level (columns).

# Notes
- Initializes the first step via a centered Euler update.
- If `prob.bc == :periodic`, periodic boundary entries are assembled in the sparse 
  operators.
- At each step `k`, the standard leap‐frog update  
  ```julia
  U[:, k+1] = M * U[:, k] + U[:, k-1]

is followed by the Robert Asselin Filter which adds a diffusion term.

U[:, k] .+= gamma .* (U[:, k-1] .- 2 .* U[:, k] .+ U[:, k+1])

    Returns U even if prob.time_steps < 2 (only initial column filled).

Example

prob = LinearAdvectionProblem(...)
U = leapfrog_ra_advection(prob, 0.1)

"""
function leap_frog_RA_advection(prob::LinearAdvectionProblem, gamma::AbstractFloat) # better name: leapfrog_ra_advection
    U = zeros(Float64, prob.grid_size, prob.time_steps) # storage for numerical approximation
    cell_centers = prob.grid_cell_boundary[1:end-1] + diff(prob.grid_cell_boundary) * 0.5
    U[:, 1] = prob.initial.(cell_centers)
    
    # euler step
    main_diagonal = ones(prob.grid_size)
    upper_diagonal = - prob.cfl * 0.5 * ones(prob.grid_size-1)
    lower_diagonal = prob.cfl * 0.5 * ones(prob.grid_size-1)
    
    I = [collect(2:prob.grid_size); collect(1:prob.grid_size-1); collect(1:prob.grid_size)]  # row indices
    J = [collect(1:prob.grid_size-1); collect(2:prob.grid_size); collect(1:prob.grid_size)]  # col indices
    entries = [lower_diagonal; upper_diagonal; main_diagonal]
    
    if prob.bc == :periodic # insert periodic boundary conditions
        I = [I; prob.grid_size; 1]
        J = [J; 1; prob.grid_size]
        entries = [entries; 0.5 * prob.cfl;- 0.5 * prob.cfl]
    end
    euler_step = sparse(I, J, entries)
    
    upper_diagonal = -prob.cfl * ones(prob.grid_size-1)
    lower_diagonal = prob.cfl * ones(prob.grid_size-1)
    entries = [lower_diagonal; upper_diagonal]
    I = [collect(2:prob.grid_size); collect(1:prob.grid_size-1)]  # row indices
    J = [collect(1:prob.grid_size-1); collect(2:prob.grid_size)]  # col indices
    
    
    if prob.bc == :periodic # insert periodic boundary conditions
        I = [I; prob.grid_size; 1]
        J = [J; 1; prob.grid_size]
        entries = [entries; prob.cfl; prob.cfl]
    end
    
    iteration_matrix = sparse(I, J, entries)
    
    if prob.time_steps >= 2
        U[:, 2] = euler_step * U[:, 1]
    end
    
    if prob.time_steps > 2
        for k in 2:prob.time_steps-1
            U[:, k+1] = iteration_matrix * U[:, k] + U[:, k-1]
            U[:, k] = U[:, k] + gamma * (U[:, k-1] - 2 * U[:, k] + U[:, k+1])
        end
    end

    return U
end

"""
    lax_wendroff_advection(prob::LinearAdvectionProblem) -> Array{Float64,2}

Compute the numerical solution of a one-dimensional linear advection problem
using the Lax–Wendroff scheme.

# Arguments
- `prob::LinearAdvectionProblem`
  - `prob.grid_size::Int`: Number of spatial cells.
  - `prob.time_steps::Int`: Number of temporal steps to advance.
  - `prob.grid_cell_boundary::AbstractVector{<:Real}`: Coordinates of cell boundaries.
  - `prob.initial::Function`: Initial condition function, evaluated at cell centers.
  - `prob.advection_speed::Real`: Constant advection speed ``c``.
  - `prob.dt::Real`: Time‐step size.
  - `prob.dx::Real`: Spatial cell width.

# Returns
- `U::Array{Float64,2}`: A matrix of size `(grid_size, time_steps)` where each
  column `U[:, k]` contains the approximate solution at time level `k`.

# Notes
- Implements the standard Lax–Wendroff two‐step method:
  1. Compute the intermediate flux via `compute_flux_LW`.
  2. Update solution using a centered finite‐difference approximation.
- Periodic boundary conditions are enforced via `circshift`.

# Example
``` julia
# Define a problem with 100 cells, 200 time steps,
# domain [0,1], constant speed c=1.0, and a sine wave initial condition.
prob = LinearAdvectionProblem(..)

U = lax_wendroff_advection(prob)
# U[:, end] holds the solution after 200 time steps
```
"""
function lax_wendroff_advection(prob::LinearAdvectionProblem)
    U = zeros(Float64, prob.grid_size, prob.time_steps) # storage for numerical approximation
    cell_centers = prob.grid_cell_boundary[1:end-1] + diff(prob.grid_cell_boundary) * 0.5
    U[:, 1] = prob.initial.(cell_centers)
    
    for k in 1:prob.time_steps-1
        flux = compute_flux_LW(U[:, k], prob.advection_speed, prob.dt, prob.dx)
        U[:,k+1] = U[:, k] .- (prob.dt/prob.dx) .* (flux .- circshift(flux, 1))
    end

    return U
end


"""
    van_leer_advection(prob::LinearAdvectionProblem) -> Matrix{Float64}

Compute the time evolution of a linear advection problem using the
van Leer flux limiter scheme.

# Arguments
- `prob::LinearAdvectionProblem`
  - `grid_size::Int` — number of spatial cells.
  - `time_steps::Int` — number of time steps to simulate.
  - `grid_cell_boundary::Vector{Float64}` — coordinates of cell boundaries.
  - `initial::Function` — initial condition function of position.
  - `advection_speed::Float64` — constant advection velocity.
  - `dt::Float64` — time‐step size.
  - `dx::Float64` — spatial cell width.

# Returns
- `U::Matrix{Float64}` — a `(grid_size × time_steps)` array, where each column
  `U[:, k]` is the approximate solution at time step `k`.

# Notes
- Uses Lax–Wendroff flux (high-order, `compute_flux_LW`) and upwind flux
  (low-order, `compute_flux_upwind`), combined via the van Leer limiter
  (`limiter_vanleer`).
- Enforces periodic boundary conditions via `circshift`.
- Cell‐center positions are computed as midpoints of `grid_cell_boundary`.

# Example
```julia
prob = LinearAdvectionProblem(...)
)
U = van_leer_advection(prob)

"""
function van_leer_advection(prob::LinearAdvectionProblem)
    U = zeros(Float64, prob.grid_size, prob.time_steps) # storage for numerical approximation
    cell_centers = prob.grid_cell_boundary[1:end-1] + diff(prob.grid_cell_boundary) * 0.5
    U[:, 1] = prob.initial.(cell_centers)
    
    for k in 1:prob.time_steps-1
        FH = compute_flux_LW(U[:, k], prob.advection_speed, prob.dt, prob.dx)
        FL = compute_flux_upwind(U[:, k], prob.advection_speed)
        slope_ratio = compute_slope_ratio(U[:, k], prob.advection_speed)
        flux = @. FL + limiter_vanleer(slope_ratio) * (FH - FL)    
        U[:,k+1] = U[:, k] .- (prob.dt/prob.dx) .* (flux .- circshift(flux, 1))
    end

    return U
end

"""
    MOL_RK4_advection(prob::LinearAdvectionProblem)

Perform advection of a scalar field using the Method‐of‐Lines with fourth‐order central differences 
in space and a classical fourth‐order Runge–Kutta (RK4) scheme in time.

# Arguments
- `prob::LinearAdvectionProblem`  
  A problem container with fields  
  – `grid_size::Int`    Number of spatial grid points (must be > 5).  
  – `grid_cell_boundary::Vector{T}` Boundaries of each grid cell (length `grid_size+1`).  
  – `dx::T`        Grid spacing.  
  – `time_steps::Int`  Number of time steps to compute.  
  – `advection_speed::T` Constant speed at which the field is advected.  
  – `initial::Function` Initial condition function; called on cell centers.  
  – `bc::Symbol`    Boundary condition flag (`:periodic` or others).

# Returns
- `U::Matrix{Float64}`  
  A `(grid_size × time_steps)` array where column `k` holds the numerical solution at time step `k`.

# Notes
- Builds a sparse fourth‐order finite‐difference operator for ∂/∂x:
  - Uses a 5‐point stencil (requires `grid_size > 5`).
  - Adds periodic wrap‐around entries if `prob.bc == :periodic`.
- Advances in time by repeatedly calling `rk4_step` on each column of `U`.

# Example
```julia
julia> prob = LinearAdvectionProblem(
           grid_size = 100,
           grid_cell_boundary = LinRange(0.0, 1.0, 101),
           dx = 0.01,
           time_steps = 200,
           advection_speed = 1.0,
           initial = x -> exp(-100*(x-0.5)^2),
           bc = :periodic
       );

julia> U = MOL_RK4_advection(prob)
100×200 Array{Float64,2}

"""
function MOL_RK4_advection(prob::LinearAdvectionProblem)
    @assert prob.grid_size > 5

    N = prob.grid_size
    # first upper diagonal; second upper diagonal; first lower diagonal; second lower diagonal
    I = [collect(1:N-1); collect(1:N-2); collect(2:N); collect(3:N)]
    J = [collect(2:N); collect(3:N); collect(1:N-1); collect(1:N-2)]
    entries = [8 * ones(N-1); -1 * ones(N-2); -8 * ones(N-1); 1 * ones(N-2)]
    
    if prob.bc == :periodic
        I = [I; N-1; N; N; 1; 1; 2]
        J = [J; 1; 1; 2; N-1; N; N]
        entries = [entries; -1; 8; -1; 1; -8; 1]
    end
    
    entries = (1 / (12 * prob.dx)) .* entries
    
    difference_operator_order4 = sparse(I,J, entries)
    advection_spatial(u) = advection_rhs(u, difference_operator_order4, prob.advection_speed)

    U = zeros(Float64, prob.grid_size, prob.time_steps) # storage for numerical approximation
    cell_centers = prob.grid_cell_boundary[1:end-1] + diff(prob.grid_cell_boundary) * 0.5
    U[:, 1] = prob.initial.(cell_centers)

    for k in 2:prob.time_steps
        U[:, k] = rk4_step(U[:, k-1], prob, advection_spatial)
    end
    return U

end

end # module