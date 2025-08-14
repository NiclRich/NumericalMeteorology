using SparseArrays
using Plots
using Dates
import TOML


"""
    leap_frog_swe(dx::T, dt::T, time_steps::Integer, grid_points::Integer,
                  h0::Function, gamma::T; bc::Symbol = :periodic,
                  g::T = 9.81, H::T = 4e3) where {T<:AbstractFloat}

Solve the 1D linearized shallow water equations using the leap-frog time-stepping
scheme with an initial Euler step and an optional Robert–Asselin (RA) time filter.

# Arguments
- `dx::T`: Spatial grid spacing.
- `dt::T`: Time step size.
- `time_steps::Integer`: Number of time steps in the simulation.
- `grid_points::Integer`: Number of spatial grid cells.
- `h0::Function`: Initial condition function for the height field `h(x)`. It will be
  evaluated at cell centers.
- `gamma::T`: Robert–Asselin filter parameter (`0 ≤ gamma ≤ 0.5` typically).
- `bc::Symbol = :periodic`: Boundary condition type. Currently supports:
    - `:periodic` — periodic boundaries in space.
    - Other values are not yet implemented.
- `g::T = 9.81`: Acceleration due to gravity.
- `H::T = 4e3`: Mean fluid depth.

# Returns
A tuple `(u, h)`:
- `u` — 2D array of size `(grid_points, time_steps)` containing velocity at each grid cell
  and time step.
- `h` — 2D array of size `(grid_points, time_steps)` containing free-surface height.

# Method
1. Discretizes the 1D shallow water equations using a finite difference
   spatial derivative (first-order upwind stencil on an Arakawa C-grid layout).
2. Uses an initial forward Euler step to start the leap-frog method.
3. Advances the solution with the leap-frog scheme in time.
4. Applies the Robert–Asselin filter at each step to suppress the computational mode.

# Notes
- The RA filter modifies intermediate time levels to control leap-frog's inherent
  time-splitting instability.
- The `bc` parameter can be extended to include reflective or open boundary conditions.
- Ensure the Courant–Friedrichs–Lewy (CFL) stability condition is satisfied:
  `c * dt / dx ≤ 1`, where `c = sqrt(g * H)` is the wave speed.

# Example
```julia
h0(x) = exp(-(x - 50)^2 / 100)  # Gaussian bump
u, h = leap_frog_swe(1.0, 0.05, 200, 101, h0, 0.1)

"""
function leap_frog_swe(dx::T, dt::T, time_steps::Integer, grid_points::Integer, h0::Function, gamma::T, bc::Symbol = :periodic, g::T=9.81, H::T=4e3) where{T<:AbstractFloat} 
    u = zeros(T, grid_points, time_steps) # storage for numerical approximation
    h = zeros(T, grid_points, time_steps)

    # compute initial conditions for h
    cell_boundaries = collect(0:grid_points) .* dx
    cell_centers = cell_boundaries[1:end-1] + 0.5 * diff(cell_boundaries)
    h[:, 1] = h0.(cell_centers)   
    
    # euler step
    # according to Equation 7.3 in the book of Döös
    upper_diagonal = - ones(grid_points-1)
    lower_diagonal = ones(grid_points-1)
    
    I = [collect(2:grid_points); collect(1:grid_points-1)]  # row indices
    J = [collect(1:grid_points-1); collect(2:grid_points)]  # col indicest
    entries = [lower_diagonal; upper_diagonal]
    
    if bc == :periodic # insert periodic boundary conditions
        I = [I; grid_points; 1]
        J = [J; 1; grid_points]
        entries = [entries; 1.0; -1.0]
    end
    difference_operator = sparse(I, J, entries)

    # perform euler step
    u[:, 2] = (- 0.5 * g * dt / dx) * difference_operator * h[:, 1] + u[:, 1]
    h[:, 2] = (- 0.5 * H * dt / dx) * difference_operator * u[:, 1] + h[:, 1]

    # leap frog
    for k in 2:time_steps-1
        u[:, k+1] = (- g * dt / dx) * difference_operator * h[:, k] + u[:, k-1]
        h[:, k+1] = (- H * dt / dx) * difference_operator * u[:, k] + h[:, k-1]

        # apply the RA filter
        u[:, k] = u[:, k] + gamma * (u[:, k-1] - 2 * u[:, k] + u[:, k+1])
        h[:, k] = h[:, k] + gamma * (h[:, k-1] - 2 * h[:, k] + h[:, k+1])
    end
    return u, h
end


"""
    leap_frog_swe_staggered(dx::T, dt::T, time_steps::Integer, grid_points::Integer,
                            h0::Function, gamma::T; bc::Symbol = :periodic,
                            g::T = 9.81, H::T = 4e3) where {T<:AbstractFloat}

Solve the 1D linearized shallow water equations using a staggered C-grid layout
and leap-frog time-stepping, with an initial Euler step and optional
Robert–Asselin (RA) time filter.

# Arguments
- `dx::T`: Spatial grid spacing (distance between adjacent velocity points).
- `dt::T`: Time step size.
- `time_steps::Integer`: Number of time steps in the simulation.
- `grid_points::Integer`: Number of spatial grid cells (also the number of `u` points).
- `h0::Function`: Initial condition function for the height field `h(x)`. It is evaluated
  at cell centers.
- `gamma::T`: Robert–Asselin filter parameter (`0 ≤ gamma ≤ 0.5` is typical).
- `bc::Symbol = :periodic`: Boundary condition type.
    - `:periodic` — periodic boundaries in space.
    - Other values are not implemented yet.
- `g::T = 9.81`: Gravitational acceleration.
- `H::T = 4e3`: Mean fluid depth.

# Returns
A tuple `(u, h)`:
- `u` — 2D array of size `(grid_points, time_steps)` containing velocity values
  located at cell boundaries.
- `h` — 2D array of size `(grid_points, time_steps)` containing free-surface height
  located at cell centers.

# Method
- **Spatial grid**: Uses a staggered Arakawa C-grid:
    - `u` (velocity) is defined at cell boundaries.
    - `h` (height) is defined at cell centers.
- **Difference operators**: First-order finite difference matrices are built separately
  for `u` and `h`, enforcing the chosen boundary conditions.
- **Time stepping**:
    1. A forward Euler step is performed to obtain the second time level.
    2. Leap-frog time-stepping is used for all subsequent updates.
    3. The Robert–Asselin filter is applied at each time level to suppress the leap-frog
       computational mode.
- **Boundary conditions**: Periodic boundaries wrap the last grid point to the first.

# Notes
- The scheme is conditionally stable; ensure the CFL condition is satisfied:
  `sqrt(g * H) * dt / dx ≤ 1`.
- The staggered grid improves accuracy of wave propagation compared to a co-located grid.

# Example
```julia
h0(x) = exp(-(x - 50)^2 / 100)  # Gaussian bump
u, h = leap_frog_swe_staggered(1.0, 0.05, 200, 101, h0, 0.1)

"""
function leap_frog_swe_staggered(dx::T, dt::T, time_steps::Integer, grid_points::Integer, h0::Function, gamma::T, bc::Symbol = :periodic, g::T=9.81, H::T=4e3) where{T<:AbstractFloat} 
    u = zeros(T, grid_points, time_steps) # storage for numerical approximation
    h = zeros(T, grid_points, time_steps)

    # compute initial conditions for h
    cell_boundaries = collect(0:grid_points) .* dx # location of u
    cell_centers = cell_boundaries[1:end-1] + 0.5 * diff(cell_boundaries) # location of h
    h[:, 1] = h0.(cell_centers)
  
    difference_operator_u = spdiagm( 0 => ones(grid_points), 1 => - ones(grid_points-1))

    # apply boundary conditions
    if bc == :periodic
        difference_operator_u[grid_points, 1] = -1
    end
    
    difference_operator_h = - transpose(difference_operator_u)
    
    # perform euler step
    u[:, 2] = (- g * dt / dx) * difference_operator_h * h[:, 1] + u[:, 1]
    h[:, 2] = (- H * dt / dx) * difference_operator_u * u[:, 1] + h[:, 1]

    # leap frog
    for k in 2:time_steps-1
        u[:, k+1] = (- 2 * g * dt / dx) * difference_operator_h * h[:, k] + u[:, k-1]
        h[:, k+1] = (- 2 * H * dt / dx) * difference_operator_u * u[:, k] + h[:, k-1]

        # apply the RA filter
        u[:, k] = u[:, k] + gamma * (u[:, k-1] - 2 * u[:, k] + u[:, k+1])
        h[:, k] = h[:, k] + gamma * (h[:, k-1] - 2 * h[:, k] + h[:, k+1])
    end
    return u, h
end

"""
    swe_arakawa(Nx::Integer, Ny::Integer, dx::T, dy::T, dt::T, time_steps::Integer,
                h0::Function, f::T; gamma::T = T(0.1), g::T = T(9.81), H::T = T(4000.0)) where {T<:AbstractFloat}

Solve the 2D linearized shallow water equations (SWE) on an Arakawa C-grid
using the leap-frog time-stepping method with a Robert–Asselin (RA) filter.

# Arguments
- `Nx::Integer`: Number of grid points in the x-direction.
- `Ny::Integer`: Number of grid points in the y-direction.
- `dx::T`: Grid spacing in the x-direction.
- `dy::T`: Grid spacing in the y-direction.
- `dt::T`: Time step size.
- `time_steps::Integer`: Number of time steps to simulate.
- `h0::Function`: Initial condition function for the free surface height `h(x, y)`.
  This function must accept two arguments `(x, y)` and return the height at that location.
- `f::T`: Constant Coriolis parameter.
- `gamma::T = 0.1`: Robert–Asselin filter coefficient to control leap-frog’s
  computational mode instability.
- `g::T = 9.81`: Gravitational acceleration.
- `H::T = 4000.0`: Mean fluid depth.

# Returns
A tuple `(u, v, h)` where:
- `u`: 3D array of shape `(Nx, Ny, time_steps)` representing the x-component of velocity.
- `v`: 3D array of shape `(Nx, Ny, time_steps)` representing the y-component of velocity.
- `h`: 3D array of shape `(Nx, Ny, time_steps)` representing the free-surface height.

# Method
1. **Grid layout**: The computation uses an Arakawa C-grid, where velocity components
   `u` and `v` are staggered relative to `h`.
2. **Initialization**: The initial condition `h0(x, y)` is evaluated at cell centers.
3. **Euler step**: The first time step is computed with an explicit Euler method.
4. **Leap-frog integration**: Subsequent steps use the leap-frog scheme for second-order
   time accuracy.
5. **RA filter**: The Robert–Asselin filter is applied to suppress leap-frog’s
   computational mode.

# Notes
- Periodic boundary conditions are applied in both x and y directions via index wrapping (`circshift`).
- Stability requires the CFL condition:
  ```math
  max(c_x, c_y) = max(√(gH) * dt / dx, √(gH) * dt / dy) ≤ 1

    The Coriolis term is treated with an Arakawa C-grid averaging stencil for improved stability.

Example

h0(x, y) = exp(-((x - 50)^2 + (y - 50)^2) / 100) # Gaussian bump
u, v, h = swe_arakawa(101, 101, 1.0, 1.0, 0.05, 200, h0, 1e-4)

"""
function swe_arakawa(Nx::Integer, Ny::Integer, dx::T, dy::T, dt::T, time_steps::Integer, h0::Function, f::T, gamma::T=T(0.1), g::T=T(9.81), H::T=T(4000.0)) where {T<:AbstractFloat}
    u = zeros(Nx, Ny, time_steps)
    v = zeros(Nx, Ny, time_steps)
    h = zeros(Nx, Ny, time_steps)

    x = collect(range(start=T(0.0), step=dx, length=Nx))
    y = collect(range(start=T(0.0), step=dy, length=Ny))

    i = collect(1:Nx)
    j = collect(1:Ny)
    ip1 = circshift(i, -1)
    im1 = circshift(i, 1)
    jp1 = circshift(j, -1)
    jm1 = circshift(j, 1)

    # apply initial conditions for the height
    h[i, j, 1] = @. h0(x, y')

    k = 1 # euler step
    # compute rhs for the method of lines
    dudt = -g / dx * (h[ip1, j, k] - h[i, j, k]) + 0.25 * f * (v[i, j, k] + v[ip1, j, k] + v[ip1, jm1, k] + v[i, jm1, k])
    dvdt = -g / dy * (h[i, jp1, k] - h[i, j, k]) + 0.25 * f * (u[i, j, k] + u[i, jp1, k] + u[im1, jp1, k] + u[im1, j, k])
    dhdt = - H * ((u[i, j, k] - u[im1, j, k]) / dx + (v[i, j, k] - v[i, jm1, k]) / dy)

    # perform euler step for time integration
    u[i, j, k+1] = u[i, j, k] + dt * dudt
    v[i, j, k+1] = v[i, j, k] + dt * dvdt
    h[i, j, k+1] = h[i, j, k] + dt * dhdt

    # leap frog
    for k = 2:time_steps-1
        dudt = -g / dx * (h[ip1, j, k] - h[i, j, k]) + 0.25 * f * (v[i, j, k] + v[ip1, j, k] + v[ip1, jm1, k] + v[i, jm1, k])
        dvdt = -g / dy * (h[i, jp1, k] - h[i, j, k]) + 0.25 * f * (u[i, j, k] + u[i, jp1, k] + u[im1, jp1, k] + u[im1, j, k])
        dhdt = - H * ((u[i, j, k] - u[im1, j, k]) / dx + (v[i, j, k] - v[i, jm1, k]) / dy)

        u[i, j, k+1] = u[i, j, k-1] + 2 * dt * dudt
        v[i, j, k+1] = v[i, j, k-1] + 2 * dt * dvdt
        h[i, j, k+1] = h[i, j, k-1] + 2 * dt * dhdt

        u[i, j, k] = u[i, j, k] + gamma * (u[i, j, k-1] - 2 * u[i, j, k] + u[i, j, k+1])
        v[i, j, k] = v[i, j, k] + gamma * (v[i, j, k-1] - 2 * v[i, j, k] + v[i, j, k+1])
        h[i, j, k] = h[i, j, k] + gamma * (h[i, j, k-1] - 2 * h[i, j, k] + h[i, j, k+1])
    end

    return u, v, h
end

"""
create_quiver_gif(u, v, h; k=8, m=10, dx=1.0, dy=1.0, filename="anim.gif",
                   cmap=:viridis, quiver_scale=1.0, fps=10)

Create a GIF visualizing the vector field (u,v) as a quiver and the scalar field h as a colormapped
background. The arrays u,v,h are expected to be (Nx, Ny, Nt).

Arguments:
- u, v, h : Float arrays of shape (Nx, Ny, Nt)
Keyword arguments:
- k         : integer ≤ 9, choose k x k arrows (default 8). Must be >= 1.
- m         : integer ≥ 1, output a frame every m time steps (default 10)
- dx, dy    : physical spacing in x and y (default 1.0)
- filename  : output GIF filename (default "anim.gif")
- cmap      : colormap symbol for the background (default :viridis)
- quiver_scale : scalar multiplied into arrow components (default 1.0)
- fps       : frames per second for the GIF (default 10)

Returns nothing; writes filename to disk.
"""
function create_quiver_gif(u::AbstractArray, v::AbstractArray, h::AbstractArray;
                           k::Integer=8, m::Integer=10,
                           dx::Real=5e5, dy::Real=5e5,
                           filename::String="anim.gif",
                           img_path::String="./",
                           cmap=:RdBu, quiver_scale::Real=1.0,
                           fps::Integer=10)

    # basic validation
    Nx, Ny, Nt = size(u)
    @assert size(v) == (Nx, Ny, Nt) "v must have same shape as u"
    @assert size(h) == (Nx, Ny, Nt) "h must have same shape as u"
    @assert 1 ≤ k ≤ min(Nx, Ny) "k must be between 1 and min(Nx,Ny)"
    @assert k < 10 "please choose k < 10 as requested"
    @assert m ≥ 1 "m must be ≥ 1"

    # physical coordinates
    x = collect(0:dx:(Nx-1)*dx)
    y = collect(0:dy:(Ny-1)*dy)

    # choose evenly spaced sample indices (k per axis)
    ix = round.(Int, range(1, stop=Nx, length=k))
    jy = round.(Int, range(1, stop=Ny, length=k))

    # sample coordinates (match the order of vec(...) for a k×k matrix)
    xs = x[ix]
    ys = y[jy]
    # vec(u[ix, jy]) stacks columns -> for j in 1:k, for i in 1:k
    xq = repeat(xs, outer=k)   # x varies fastest within each column set
    yq = repeat(ys, inner=k)   # each y repeated k times for its column

    # global color limits so color is consistent across frames
    hmin = minimum(h)
    hmax = maximum(h)

    anim = @animate for k in 1:m:Nt
        # background: heatmap of h at time t
        # note: transpose h[:,:,k] so axes align intuitively (x horizontal, y vertical)
        plt = heatmap(x, y, h[:, :, k]',
                      color = cmap,
                      clims = (hmin, hmax),
                      aspect_ratio = :equal,
                      xlabel = "x", ylabel = "y",
                      title = "t = $(k)",
                      framestyle = :box,
                      colorbar = true)

        # subsampled vector components (must be flattened in the same column-major order)
        u_sub = vec(u[ix, jy, k]) .* quiver_scale
        v_sub = vec(v[ix, jy, k]) .* quiver_scale

        # overlay quiver (arrow color chosen to be black for contrast)
        quiver!(plt, xq, yq, quiver = (u_sub, v_sub),
                linewidth = 1.2, arrow = :arrow, linecolor = :black, legend = false)

        plt
    end

    # write gif
    path = "$(img_path)/$(filename)"
    gif(anim, path; fps = fps)
    return nothing
end

"""
    plot_height_heatmap(height, times, filename; dx::Real=5e5, dy::Real=5e5, cmap::Symbol=:RdBu)

Plot and save static 2D heatmaps of the free-surface height field from the shallow water equations.

# Arguments
- `height`: 3D array `(Nx, Ny, Nt)` containing scalar field values (e.g., surface height) over space and time.
- `times`: Vector of integer time indices to plot (1-based indexing).
- `filename`: Base filename (string) used when saving plots; the time index is appended automatically.
- `img_path`: String with the directory for saving images
- `dx::Real=5e5`: Grid spacing in the x-direction (in meters by default).
- `dy::Real=5e5`: Grid spacing in the y-direction.
- `cmap::Symbol=:RdBu`: Colormap symbol from `Plots.jl` (e.g., `:RdBu`, `:viridis`).

# Behavior
- The function computes the global minimum and maximum values of `height` over all time steps
  to fix the color scale across frames.
- Physical coordinates for the x and y axes are computed from grid spacing `dx` and `dy`.
- For each requested time index in `times`, a heatmap of the height field is generated with:
    - fixed color scale (`clims`) for consistency,
    - equal aspect ratio,
    - labeled axes and colorbar,
    - a title indicating the time index.
- Each heatmap is saved as a PNG file with a name of the form:

<filename>_<time>.png


# Notes
- The input array `height` must be indexed as `height[x, y, t]`.
- The color scale is fixed between `minimum(height)` and `maximum(height)` to allow direct
visual comparison between frames.
- This function does not return the plots; it writes them to disk.

# Example
```julia
using Plots
h = randn(50, 50, 10) # mock height field
plot_height_heatmap(h, [1, 5, 10], "height_plot", dx=1.0, dy=1.0, cmap=:RdBu_r)

"""
function plot_height_heatmap(height, indices, filename::String, img_path::String, dx::Real=5e5, dy::Real=5e5, cmap::Symbol=:RdBu)
    hmin = minimum(height)
    hmax = maximum(height)

    Nx, Ny, Nt = size(height)

    # coordinates of the grid
    x = collect(0:dx:(Nx-1)*dx)
    y = collect(0:dy:(Ny-1)*dy)
    
    for k in indices
        plt = heatmap(x, y, height[:, :, k]',
                          color = cmap,
                          clims = (hmin, hmax),
                          aspect_ratio = :equal,
                          xlabel = "x", ylabel = "y",
                          title = "h in 2D SWE at t = $(k)",
                          framestyle = :box,
                          colorbar = true)
        filename_adjusted = "$(img_path)/$(filename)-$(k).png"
        savefig(plt, filename_adjusted)
    end
end


"""
    create_gif_1dswe(cell_centers::Vector{T}, y::Matrix{T};
                     filename::String, img_path::String, 
                     time_steps::Integer, m::Integer, fps::Integer) where {T<:AbstractFloat}

Create and save an animated GIF showing the time evolution of the numerical solution to the
1D Shallow Water Equation (SWE).

# Arguments
- `cell_centers::Vector{T}`: Spatial coordinates of cell centers along the domain, where `T` is a subtype of `AbstractFloat`.
- `y::Matrix{T}`: Simulation output matrix of shape `(nx, nt)`, where `nx` is the number of spatial points and `nt` is the number of time steps.
- `filename::String`: Path (including filename) where the GIF should be saved.
- `img_path::String`: Directory path where output images or animations are stored (not directly used in this function, but kept for API consistency).
- `time_steps::Integer`: Total number of time steps in the simulation.
- `m::Integer`: Frame skipping factor — only every `m`-th time step is included in the animation.
- `fps::Integer`: Frames per second of the output GIF.

# Behavior
- Iterates through the simulation data, plotting the spatial profile `y[:, k]` for each selected time step `k`.
- Uses the `Plots.@animate` macro to generate frames and saves them as a GIF via `gif()`.
- Automatically adjusts the y-axis limits to the global min/max of `y`.

# Output
Saves an animated GIF file at `filename` and logs an informational message.

# Example
```julia
create_gif_1dswe(
    cell_centers, 
    h, 
    filename = "swe_animation.gif", 
    img_path = "results", 
    time_steps = size(h, 2), 
    m = 5, 
    fps = 10
)

"""
function create_gif_1dswe(cell_centers::Vector{T}, y::Matrix{T};  filename::String, img_path::String, time_steps::Integer, m::Integer, fps::Integer) where {T<:AbstractFloat}

anim = @animate for k in 1:m:time_steps
        # background: heatmap of h at time t
        # note: transpose h[:,:,t] so axes align intuitively (x horizontal, y vertical)
        plt = plot(cell_centers, y[:, k],
                    ylim = (minimum(y), maximum(y)),
            aspect_ratio = :auto,
            xlabel = "x",
            ylabel = "h(x,t)",
            title = "Numerical Solution of the SWE in d=1 at t = $(k)")

        plt
    end
    path = "$(img_path)/$(filename)"
    gif(anim, path; fps = fps)
    @info "$(now()) - Saved animation of 1D Shallow Water Equation to $(filename)"
end


"""
    create_plots_1dswe(cell_centers::Vector{T}, y::Matrix{T}, indices::Vector{<:Integer};
                       filename::String, img_path::String, quantity::String, ylabel::String) where {T<:AbstractFloat}

Generate and save 1D plots for selected time steps from a simulation of the shallow water equations (SWE).

# Arguments
- `cell_centers::Vector{T}`: Spatial coordinates of the cell centers. Must be a vector of floating-point values.
- `y::Matrix{T}`: Simulation data matrix where each column corresponds to a time step and each row to a spatial location.
                  Must have the same floating-point element type `T` as `cell_centers`.
- `indices::Vector{<:Integer}`: Indices of the time steps to plot.

# Keyword Arguments
- `filename::String`: Base filename for saved plot images (without extension or time-step index).
- `img_path::String`: Directory path where plot images will be saved.
- `quantity::String`: Name of the plotted quantity (used in plot titles).
- `ylabel::String`: Label for the y-axis of the plots.

# Behavior
For each time step index in `indices`, this function:
1. Extracts the corresponding column from `y`.
2. Creates a plot of the data versus `cell_centers`.
3. Sets consistent y-limits across all plots based on the global min/max of `y`.
4. Saves the plot as a PNG file named `"<filename>_<index>.png"` in `img_path`.

# Example
```julia
cell_centers = collect(0.0:0.1:10.0)
y = [sin.(cell_centers) + 0.1*t for t in 1:5] |> hcat
indices = [1, 3, 5]
create_plots_1dswe(cell_centers, y, indices;
                   filename="wave",
                   img_path="plots",
                   quantity="Surface height",
                   ylabel="Height (m)")

# Notes

- The plots are saved directly to disk and are not returned.

- Ensure that img_path exists before calling the function, as the function does not create directories.
"""
function create_plots_1dswe(cell_centers::Vector{T}, y::Matrix{T}, indices::Vector{<:Integer}; filename::String, img_path::String, quantity::String, ylabel::String) where {T<:AbstractFloat}
    plots = []
    ymin = minimum(y)
    ymax = maximum(y)
    for idx in indices
        save_path = "$(img_path)/$(filename)-$(idx).png"
        title = "$(quantity) at time step $(idx)"
        plt = plot(cell_centers, y[:, idx], ylim=(ymin, ymax), title=title, legend=false)
        savefig(plt, save_path)
    end
end


"""
    simulate_swe1d(; config_path::String, img_path::String)

Simulates the one-dimensional Shallow Water Equations (SWE) using both 
unstaggered (A-grid) and staggered (C-grid) schemes with a leap-frog time
integration, and generates visualizations of the results.

# Arguments
- `config_path::String`: Path to the TOML configuration file containing 
  simulation parameters for the `swe1d` section:
    - `dt` (`Float64`): Time step size.
    - `dx` (`Float64`): Spatial step size.
    - `gamma` (`Float64`): Numerical damping coefficient.
    - `grid_points` (`Int`): Number of grid points in the spatial domain.
    - `time_steps` (`Int`): Number of time steps to simulate.
    - `indices_plotting` (`Vector{Int}`): Time step indices for which plots 
      should be generated.

- `img_path::String`: Path to the directory where output images and animations 
  will be saved.

# Behavior
1. Reads simulation parameters from the configuration file.
2. Initializes the fluid height with a rectangular pulse in the center of 
   the domain.
3. Runs the SWE solver using:
    - `leap_frog_swe` for the unstaggered grid (A-grid)
    - `leap_frog_swe_staggered` for the staggered grid (C-grid)
4. Generates:
    - An animated GIF of the water height evolution for the staggered grid.
    - Line plots of velocity and height at specified time steps for both 
      grid schemes.

# Output
- Saves:
    - `swe_1d.gif`: Animated GIF of the staggered-grid water height evolution.
    - Multiple PNG plots for velocity and height, named according to grid 
      type and variable.
- Logs progress and timing information using `@info`.

# Notes
- The function enforces that `indices_plotting` values are within 
  `[1, time_steps]`.
- Uses SI units for physical constants: gravity = `9.81 m/s²`, mean fluid depth = `4×10³ m`.
- The height initial condition is 1 within the range `450 km ≤ x ≤ 550 km`, 
  and 0 elsewhere.

# Example
```julia
simulate_swe1d(
    config_path = "config.toml",
    img_path = "output"
)

"""
function simulate_swe1d(;config_path::String, img_path::String)
    # declare constants
    h0(x)= (450.0 <= x / 1000.0 <= 550.0) ? 1 : 0   # initial function for the height
    STANDARD_GRAVITY = 9.81                         # standard gravity (m/s^2)
    FLUID_DEPTH = 4e3                               # mean fluid depth (m)
    
    filename_gif_swe1d = "swe_1d.gif"
    filename_gif_swe2d = "swe_2d.gif"
    filename_staggered = "swe1d-staggered-grid"
    filename_unstaggered = "swe1d-unstaggered-grid"
    sampling_freq = 20
    fps = 15

    # load config
    config = TOML.parsefile(config_path)
    config_swe1d = config["swe1d"]
    
    dt = config_swe1d["dt"]
    dx = config_swe1d["dx"]
    gamma = config_swe1d["gamma"]
    grid_points = config_swe1d["grid_points"]
    time_steps = config_swe1d["time_steps"]
    indices = config_swe1d["indices_plotting"]
    
    # force consistency for indices for plotting
    indices = [i for i in indices if i <= time_steps && i >= 1]
    
    cell_boundaries = dx * collect(0:grid_points)
    cell_centers = cell_boundaries[1:end-1] + 0.5 * diff(cell_boundaries)
    
    @info "$(now()) - Start computation of SWE in 1D"
    # index a for the A grid (unstaggered) and c for the C grid (staggered)
    u_a, h_a = leap_frog_swe(dx, dt, time_steps, grid_points, h0, GAMMA,:periodic, STANDARD_GRAVITY, FLUID_DEPTH)
    @info "$(now()) - Computation of SWE in 1D with unstaggered grid done"
    u_c, h_c = leap_frog_swe_staggered(dx, dt, time_steps, grid_points, h0, GAMMA, :periodic, STANDARD_GRAVITY, FLUID_DEPTH)
    @info "$(now()) - Computation of SWE in 1D with staggered grid done"
    
    #create_gif_1dswe(cell_centers::Vector{T}, y::Matrix{T},  filename::String, img_path::String, time_steps::Integer, m::Integer, fps::Integer)
    create_gif_1dswe(cell_centers, h_c,
        filename = filename_gif_swe1d,
        img_path = img_path,
        time_steps = time_steps,
        m = sampling_freq,
        fps = fps)
    
    create_plots_1dswe(cell_centers, u_a, indices,
        filename = "velocity-$(filename_unstaggered)",
        img_path = img_path,
        quantity = "Velocity",
        ylabel = "u(x,t)")
    
    create_plots_1dswe(cell_centers, h_a, indices,
        filename = "height-$(filename_unstaggered)",
        img_path = img_path,
        quantity = "Height",
        ylabel = "h(x,t)")
    
    create_plots_1dswe(cell_centers, u_c, indices,
        filename = "velocity-$(filename_staggered)",
        img_path = img_path,
        quantity = "Velocity",
        ylabel = "u(x,t)")
    
    create_plots_1dswe(cell_centers, h_c, indices,
        filename = "height-$(filename_staggered)",
        img_path = img_path,
        quantity = "Height",
        ylabel = "h(x,t)")
    
    @info "$(now()) - Saved plots for 1D SWE"

end


"""
    simulate_swe2d(; config_path::String, img_path::String)

Simulates the 2D shallow water equations (SWE) using the Arakawa C-grid scheme
for two scenarios: with and without Coriolis forcing. Generates and saves
visualizations of the results, including animated quiver plots and heatmaps of
the height field.

# Arguments
- `config_path::String`: Path to the TOML configuration file containing simulation parameters.
- `img_path::String`: Directory path where output plots and animations will be saved.

# Configuration file requirements
The TOML file must contain a `swe2d` section with the following keys:
- `dt` (`Float64`): Time step size.
- `dx`, `dy` (`Float64`): Grid spacing in x and y directions.
- `Nx`, `Ny` (`Int`): Number of grid points in x and y directions.
- `gamma` (`Float64`): Parameter for the Robert Asselin Filter.
- `time_steps` (`Int`): Total number of simulation steps.
- `indices_plotting` (`Vector{Int}`): Time step indices for which plots will be generated.
- `coriolis_1` (`Float64`): Coriolis parameter for the "no Coriolis" case (typically `0.0`).
- `coriolis_2` (`Float64`): Coriolis parameter for the "with Coriolis" case.

# Behavior
1. Initializes the height field `h0(x, y)` with a square disturbance.
2. Runs two simulations of the SWE:
   - Without Coriolis (`coriolis_1`).
   - With Coriolis (`coriolis_2`).
3. Creates:
   - An animated quiver plot (`swe2d_coriolis.gif`) of the velocity and height fields with Coriolis.
   - Heatmap plots of the height field for selected time steps, for both cases.

# Output
No return value. Saves the generated plots and animations in `img_path`.

# Example
```julia
simulate_swe2d(
    config_path="config/swe_config.toml",
    img_path="results/plots"
)

"""
function simulate_swe2d(;config_path::String, img_path::String)
    
    # declare constants
    h0(x,y) = (45 * dx <= x <= 55 * dx &&  45 * dy <= y <= 55 * dy ) ? 1 : 0   # initial function for the height   
    filename_gif_swe1d = "swe-1d.gif"
    filename_gif_swe2d = "swe_2d.gif"
    filename_no_coriolis = "swe2d-no-coriolis"
    filename_coriolis = "swe2d-coriolis"
    sampling_freq = 20
    fps = 15
    
    # load configuration
    config = TOML.parsefile(config_path)
    config_swe2d = config["swe2d"]
    
    dt = config_swe2d["dt"]
    dx = config_swe2d["dx"]
    dy = config_swe2d["dy"]
    Nx = config_swe2d["Nx"]
    Ny = config_swe2d["Ny"]
    gamma = config_swe2d["gamma"]
    time_steps = config_swe2d["time_steps"]
    indices = config_swe2d["indices_plotting"]
    coriolis_1 = config_swe2d["coriolis_1"]  # no coriolis
    coriolis_2 = config_swe2d["coriolis_2"]  # coriolis f = 1e-4
    
    # force consistency for indices for plotting
    indices = [i for i in indices if i <= time_steps && i >= 1]
    
    # computation
    @info "$(now()) - Start Simulation of Shallow Water Equation in 2D"
    u_no_coriolis, v_no_coriolis, h_no_coriolis = swe_arakawa(Nx, Ny, dx, dy, dt, time_steps, h0, coriolis_1, gamma)
    @info "$(now()) - Computation of SWE with Coriolis with f = $(coriolis_1) done"
    u_coriolis, v_coriolis, h_coriolis = swe_arakawa(Nx, Ny, dx, dy,  dt, time_steps, h0, coriolis_2, gamma)
    @info "$(now()) - Computation of SWE with Coriolis with f = $(coriolis_2) done"
    
    @info "$(now()) - Start creating Plots"
    create_quiver_gif(u_coriolis, v_coriolis, h_coriolis;
                       dx=dx, dy=dx,
                       filename="swe2d_coriolis.gif",
                       img_path=img_path)
    
    plot_height_heatmap(h_coriolis, indices, filename_coriolis, img_path, dx, dy)
    plot_height_heatmap(h_no_coriolis, indices, filename_no_coriolis, img_path, dx, dy)
    @info "$(now()) - Plots for the 2D SWE created and these can be found in $(img_path)"
end