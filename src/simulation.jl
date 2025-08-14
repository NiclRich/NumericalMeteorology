include("diagnostics.jl")

"""
    simulate_advection(prob::Solvers.LinearAdvectionProblem, gamma::T) where {T<:AbstractFloat}

Simulates the linear advection equation using multiple numerical methods and returns diagnostics and solutions.

# Arguments
- `prob::Solvers.LinearAdvectionProblem`: 
    A problem definition that contains the initial condition, advection speed, grid setup, time step size, and number of time steps.
- `gamma::T`: Default = 0.1
    The Robert–Asselin filter parameter used in the Leap-Frog scheme. Must be a subtype of `AbstractFloat`.

# Returns
- `diagnostics::Dict{String, Float64}`: 
    A dictionary containing diagnostic metrics for each method and variable, including:
    - `"u_<scheme>_rmse"`: Root mean square error compared to the exact solution.
    - `"u_<scheme>_rm"`: Root mean.
    - `"u_<scheme>_rmp"`: Root mean of positive values.
    - `"u_<scheme>_mdr"`: Mean deviation ratio.
    - `"u_<scheme>_rmax"`: Relative max.
    - `"u_<scheme>_rmin"`: Relative min.
- `solutions::Dict{String, Matrix{Float64}}`:
    A dictionary mapping scheme names (`"donor_cell"`, `"leap_frog"`, `"lax_wendroff"`, `"van_leer"`, `"rk4"`) to their numerical solution arrays.

# Description
The function:
- Computes the exact solution of the linear advection equation.
- Solves the problem using five numerical schemes:
    - Donor Cell
    - Leap-Frog with Robert–Asselin filter
    - Lax–Wendroff
    - Van Leer
    - Method of Lines with RK4
- Measures performance (computation time and memory) for each scheme.
- Computes diagnostic metrics comparing each scheme to the exact or self-referential statistics.
- Logs the status of each step using `@info`.

# Example
```julia
prob = Solvers.LinearAdvectionProblem("test", ... )
gamma = 0.1
diagnostics, solutions = simulate_advection(prob, gamma)

# See Also

Solvers.donor_cell_advection, Solvers.leap_frog_RA_advection, Solvers.lax_wendroff_advection, Solvers.van_leer_advection, Solvers.MOL_RK4_advection

"""
function simulate_advection(prob::Solvers.LinearAdvectionProblem, gamma::T=0.1) where {T<:AbstractFloat}

       #  gamma constant for Robert Asselin Filter
    diagnostics = Dict() # initialise empty dictionary 

    @info "$(now()) - Start Simulation of Problem $(prob.name)"
    @info "$(now()) - Compute exact solution"
    # compute exact solution and cell centers and time
    cell_centers = prob.grid_cell_boundary[1:end-1] + diff(prob.grid_cell_boundary) * 0.5
    t = collect(range(0; step=prob.dt, length=prob.time_steps))
    exact_sol(x,t) = prob.initial(x - prob.advection_speed * t)    # exact solution as function   
    u_exact = exact_sol.(cell_centers, t')                         # exact solution as array

    # compute numerical schemes
    @info "$(now()) - Compute numerical approximation"
    u_donor_cell, comp_time, mem_allocated =  @timed Solvers.donor_cell_advection(prob)
    @info "$(now()) - Donor cell method done!\n  computation time: $(comp_time * 1000) ms\n  allocated memory: $(mem_allocated / 1000) kB"
    u_leap_frog, comp_time, mem_allocated = @timed Solvers.leap_frog_RA_advection(prob, gamma)
    @info "$(now()) - Leap frog method with Robert Asselin Filter done!\n  computation time: $(comp_time * 1000) ms\n  allocated memory: $(mem_allocated / 1000) kB"
    u_laxwendroff, comp_time, mem_allocated = @timed Solvers.lax_wendroff_advection(prob)
    @info "$(now()) - Lax Wendroff method done!\n  computation time: $(comp_time * 1000) ms\n  allocated memory: $(mem_allocated / 1000) kB"
    u_vanleer, comp_time, mem_allocated = @timed Solvers.van_leer_advection(prob)
    @info "$(now()) - Van Leer method done!\n  computation time: $(comp_time + 1000) ms\n  allocated memory: $(mem_allocated / 1000) kB"
    u_rk4, comp_time, mem_allocated = @timed Solvers.MOL_RK4_advection(prob)
    @info "$(now()) - Method of Lines done!\n  computation time: $(comp_time * 1000) ms\n  allocated memory: $(mem_allocated / 1000) kB"

    numerical_approximations = zip(["donor_cell", "leap_frog", "lax_wendroff", "van_leer", "rk4"], [u_donor_cell, u_leap_frog, u_laxwendroff, u_vanleer, u_rk4])
    for (scheme, approx) in numerical_approximations
        for (diag_name, diag) in zip(["rmse", "rm", "rmp", "mdr", "rmax", "rmin"], [diagnostic_rmse, diagnostic_rm, diagnostic_rmp, diagnostic_mdr, diagnostic_rmax, diagnostic_rmin])
            key = "u_" * scheme * "_" * diag_name
            if diag_name == "rmse"
                val = diag(u_exact, approx)
            else
                val = diag(approx)
            end
            diagnostics[key] = val
        end
    end
    @info "$(now()) - Computation of diagnostics done!"

    return diagnostics, Dict(numerical_approximations)
end