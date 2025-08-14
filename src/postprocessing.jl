using Plots
using PrettyTables
using Dates

include("solvers.jl")
using .Solvers

"""
    generate_plot_simulation(; idx::Vector{<:Integer}, title::String, cell_centers::Vector{T}, u_approx::AbstractMatrix{T}) where {T<:AbstractFloat}

Generate a plot of the solution `u(t, x)` at selected time steps using the provided spatial grid and solution matrix.

# Arguments
- `idx::Vector{<:Integer}`: Vector of column indices in `u_approx` corresponding to the desired time steps to plot.
- `title::String`: Title of the plot.
- `cell_centers::Vector{T}`: Vector of cell center coordinates (spatial grid), where `T` is a subtype of `AbstractFloat`.
- `u_approx::AbstractMatrix{T}`: Matrix of approximate solutions. Each column `u_approx[:, i]` represents the solution at time step `i`.

# Returns
- A `Plots.Plot` object representing the solution curves over the spatial domain for the selected time steps.

# Example
```julia
using Plots

x = range(0, 1; length=100)
u = [sin.(π*x) .* exp(-π^2 * t) for t in 0:0.1:1.0] |> hcat
plt = generate_plot_simulation(idx=[1, 5, 10], title="Heat Equation", cell_centers=collect(x), u_approx=u)
display(plt)

"""
function generate_plot_simulation(;idx::Vector{<:Integer}, title::String, cell_centers::Vector{T}, u_approx::AbstractMatrix{T}) where {T<:AbstractFloat}
    plt = plot(
        xlabel = "x",
        ylabel = "u(t,x",
        title=title,
        legend=:topright)
    for i in idx
        plot!(cell_centers, u_approx[:, i], label = "n = $i")
    end

    return plt
end

function titlecase(s::AbstractString)
    join(uppercasefirst.(split(s)), " ")
end


"""
    plot_diagnostic(; diag::Dict, diagnostic_name::String, title::String, label_yaxis::String, prob::Solvers.LinearAdvectionProblem)

Plot diagnostic quantities over time for multiple numerical methods.

This function searches the keys in `diag` for those that end with `diagnostic_name`, extracts the method name from the key, and plots the corresponding values against time steps. It supports formatting of method labels (e.g., converting `"rk4"` to `"RK4"` and replacing underscores with spaces).

# Arguments
- `diag::Dict`: A dictionary where keys are strings of the form `"u_<method>_<diagnostic>"`, and values are vectors of diagnostic data over time.
- `diagnostic_name::String`: The suffix used to select relevant keys in `diag`. For example, `"rmse"` would select keys like `"u_rk4_rmse"`.
- `title::String`: The title of the plot.
- `label_yaxis::String`: Label for the y-axis (e.g., `"RMSE"` or `"Relative Error"`).
- `prob::Solvers.LinearAdvectionProblem`: A problem object that provides the number of time steps (`prob.time_steps`).

# Returns
- A plot object (`Plots.Plot`) with one line per method that has the specified diagnostic, plotted against time steps.

# Example
```julia
plt = plot_diagnostic(
    diag = diag_data,
    diagnostic_name = "rmse",
    title = "RMSE over Time",
    label_yaxis = "Root Mean Square Error",
    prob = problem_instance
)
display(plt)

"""
function plot_diagnostic(;diag::Dict, diagnostic_name::String, title::String, label_yaxis::String, prob::Solvers.LinearAdvectionProblem)
    x = 1:prob.time_steps
    plt = plot(
        xlabel = "Time Step",
        ylabel = label_yaxis,
        title=title,
        legend=:outertopright)

    for key in keys(diag)
        pos = findlast(==('_'), key)
        if key[pos+1:end] == diagnostic_name
            label = replace(key[3:pos-1], "_" => " ")
            if label == "rk4"
                    label = "RK4"
            else
                label = titlecase(label)
            end
            plot!(x, diag[key], label=label)
        end
    end

    return plt
end

"""
    generate_table(; diag::Dict, indices::Vector{<:Integer}, path_table::String,
                    title::String, latex_label::String, prob::Solvers.LinearAdvectionProblem)

Generate a diagnostics table for various numerical schemes applied to a linear advection problem.

This function collects diagnostic metrics from a dictionary for multiple numerical schemes
(`Donor Cell`, `Leap Frog`, `Lax-Wendroff`, `van Leer`, and `RK4`) and constructs a formatted table.
The resulting table is printed to the terminal and also saved as a LaTeX file.

# Arguments
- `diag::Dict`: A dictionary containing diagnostic arrays, keyed by metric names like `"u_donor_cell_rmse"`,
  `"u_leap_frog_rm"`, etc. Each value must be indexable with `indices`.
- `indices::Vector{<:Integer}`: Indices to extract from each diagnostic array. Usually corresponds to time steps but can also be a subset of it.
- `path_table::String`: Path to the directory where the LaTeX file should be saved.
- `title::String`: Title of the table to be used both in terminal and LaTeX output.
- `latex_label::String`: The LaTeX label for referencing the table (e.g., `"tab:diagnostics"`).
- `prob::Solvers.LinearAdvectionProblem`: A problem object whose `name` field is used to name the output file.

# Behavior
- Extracts diagnostic values for each numerical method.
- Builds a table with the following columns:
  `"Method"`, `"time step"`, `"RMSE"`, `"RM"`, `"RMP"`, `"MDR"`, `"RMAX"`, `"RMIN"`.
- Outputs the table to the terminal using `PrettyTables.jl`.
- Writes a LaTeX version of the table to the file
  `path_table / (prob.name * "-diagnostics-table.tex")`.

# Example
```julia
generate_table(
    diag = diagnostics_dict,
    indices = 1:10,
    path_table = "results/tables",
    title = "Diagnostics for Linear Advection",
    latex_label = "tab:linear_adv_diag",
    prob = my_problem_instance
)

"""
function generate_table(;diag::Dict, indices::Vector{<:Integer}, path_table::String, title::String, latex_label::String, prob::Solvers.LinearAdvectionProblem)
    data_dc = hcat(["Donor Cell"; repeat([""], length(indices)-1)],
                indices,
                diag["u_donor_cell_rmse"][indices],
                diag["u_donor_cell_rm"][indices],
                diag["u_donor_cell_rmp"][indices],
                diag["u_donor_cell_mdr"][indices],
                diag["u_donor_cell_rmax"][indices], 
                diag["u_donor_cell_rmin"][indices])    
    data_lf = hcat( ["Leap Frog"; repeat([""], length(indices)-1)],
                indices, 
                diag["u_leap_frog_rmse"][indices],
                diag["u_leap_frog_rm"][indices],
                diag["u_leap_frog_rmp"][indices],
                diag["u_leap_frog_mdr"][indices],
                diag["u_leap_frog_rmax"][indices], 
                diag["u_leap_frog_rmin"][indices])
    data_lw = hcat(["Lax Wendroff"; repeat([""], length(indices)-1)],
                indices,
                diag["u_lax_wendroff_rmse"][indices],
                diag["u_lax_wendroff_rm"][indices],
                diag["u_lax_wendroff_rmp"][indices],
                diag["u_lax_wendroff_mdr"][indices],
                diag["u_lax_wendroff_rmax"][indices], 
                diag["u_lax_wendroff_rmin"][indices])    
    data_vl = hcat(["van Leer"; repeat([""], length(indices)-1)],
                indices,
                diag["u_van_leer_rmse"][indices],
                diag["u_van_leer_rm"][indices],
                diag["u_van_leer_rmp"][indices],
                diag["u_van_leer_mdr"][indices],
                diag["u_van_leer_rmax"][indices], 
                diag["u_van_leer_rmin"][indices])    
    data_rk4 = hcat(["RK4"; repeat([""], length(indices)-1)],
                indices,
                diag["u_rk4_rmse"][indices],
                diag["u_rk4_rm"][indices],
                diag["u_rk4_rmp"][indices],
                diag["u_rk4_mdr"][indices],
                diag["u_rk4_rmax"][indices], 
                diag["u_rk4_rmin"][indices])
    
    table_data = vcat(data_dc, data_lf, data_lw, data_vl, data_rk4)
    #title = "Diagnostics for the linear advection problem"
    column_labels =  ["Method", "time step", "RMSE", "RM", "RMP", "MDR", "RMAX", "RMIN"]
    #merge_column_label_cells = :auto
    cuts    = 1 .+ (0:5) * length(indices)     # [1, n+1, 2n+1, 3n+1, 4n+1, 5n+1]
    file_name = path_table * "/" * prob.name * "-diagnostics-table.tex"

    # write to latex file
    open(file_name, "w") do io
        pretty_table(io, table_data;
            header  = column_labels,
            title   = title,
            backend = Val(:latex),
            label   = latex_label
        )
    end
    @info "Table with diagnostics saved as $(file_name)"
# print to stdout
    pretty_table(
        table_data,
        header = column_labels,
        title = title,
        hlines = [0;cuts], # repeat([length(indices)+1], 5)],
        crop = :none
        )
end


"""
    post_process(
        diagnostics::Dict,
        numerical_approximations::Dict,
        indices::Vector{T},
        path_images::String,
        path_table::String,
        label_table::String,
        prob::Solvers.LinearAdvectionProblem
    ) where {T<:Integer}

Generate and save diagnostic plots and simulation visualizations for a linear advection problem using various numerical schemes, and export a diagnostics table in LaTeX format.

# Arguments
- `diagnostics::Dict`: Dictionary containing diagnostic results keyed by diagnostic names and scheme identifiers.
- `numerical_approximations::Dict`: Dictionary mapping scheme names (e.g., `"donor_cell"`, `"leap_frog"`) to numerical solution arrays.
- `indices::Vector{T}`: Time step indices to be used for plotting snapshots of the simulation.
- `path_images::String`: Directory path where plot images will be saved.
- `path_table::String`: Path to the output LaTeX file containing the diagnostics table.
- `label_table::String`: LaTeX label for referencing the diagnostics table (used in label{}).
- `prob::Solvers.LinearAdvectionProblem`: The problem definition, including grid and time step information.

# Behavior
- Filters out indices that exceed the number of time steps in `prob`.
- Plots solution snapshots at the specified time steps for five numerical schemes:
  - Donor Cell
  - Leap Frog
  - Lax-Wendroff
  - Van Leer
  - Runge-Kutta 4 (RK4)
- Plots six diagnostic metrics over time:
  - RMSE, RM, RMP, MDR, RMAX, RMIN
- Saves all plots as PNG files in `path_images`.
- Generates and saves a LaTeX diagnostics table in `path_table`.

# Side Effects
- Writes image files to disk.
- Writes a LaTeX table file to disk.
- Logs progress messages using `@info`.

# Example
```julia
post_process(diagnostics, approximations, 1:10, "images", "output/table.tex", "tab:diagnostics", problem)

"""
function post_process(;diagnostics::Dict, numerical_approximations::Dict, indices::Vector{T}, path_images::String, path_table::String, label_table::String, prob::Solvers.LinearAdvectionProblem) where {T<:Integer}
    idx = [i for i in indices if i <= prob.time_steps]
    cell_centers = prob.grid_cell_boundary[1:end-1] + diff(prob.grid_cell_boundary) * 0.5
    @info "Start post processing"
    
    # generate plots for the simulation
    plt_donor_cell = generate_plot_simulation(idx=idx,     # indices to plot
        title = "Simulation with the Donor Cell Scheme",
        cell_centers = cell_centers,
        u_approx =  numerical_approximations["donor_cell"])
    filename = path_images * "/" * prob.name *  "-simulation-donorcell.png"
    savefig(plt_donor_cell, filename)
    @info "$(now()) - Simulation image saved to $(filename)"

    plt_leap_frog = generate_plot_simulation(idx=idx,
        title = "Simulation with the Leap Frog Scheme",
        cell_centers = cell_centers,
        u_approx = numerical_approximations["leap_frog"])
    filename = path_images * "/" * prob.name *  "-simulation-leapfrog.png"
    savefig(plt_leap_frog, filename)
    @info "$(now()) - Simulation image saved to $(filename)"

    plt_laxwendroff = generate_plot_simulation(idx=idx,
        title = "Simulation with the Lax Wendroff Scheme",
        cell_centers = cell_centers,
        u_approx = numerical_approximations["lax_wendroff"])
    filename =  path_images * "/" * prob.name *  "-simulation-laxwendroff.png"
    savefig(plt_laxwendroff, filename)
    @info "$(now()) - Simulation image saved to $(filename)"

    plt_van_leer = generate_plot_simulation(idx=idx,
        title = "Simulation with the Van Leer Scheme",
        cell_centers = cell_centers,
        u_approx = numerical_approximations["van_leer"])
    filename = path_images * "/" * prob.name *  "-simulation-vanleer.png"
    savefig(plt_van_leer, filename)
    @info "$(now()) - Simulation image saved to $(filename)"

    plt_rk4 = generate_plot_simulation(idx=idx,
        title = "Simulation with the RK4 Method",
        cell_centers = cell_centers,
        u_approx = numerical_approximations["rk4"])
    filename = path_images * "/" * prob.name *  "-simulation-rk4.png"
    savefig(plt_rk4, filename)
    @info "$(now()) - Simulation image saved to $(filename)"

    # plot diagnostics
    plt_rmse = plot_diagnostic(diag=diagnostics,
        diagnostic_name="rmse",
        title="RMSE for Different Schemes",
        label_yaxis="RMSE",
        prob=prob)
    filename = path_images * "/" * prob.name * "-RMSE.png"
    savefig(plt_rmse, filename)
    @info "$(now()) - Plot RMSE saved to $(filename)"

    plt_rm = plot_diagnostic(diag=diagnostics,
        diagnostic_name="rm",
        title="RM for Different Schemes",
        label_yaxis="RM",
        prob=prob)
    filename = path_images * "/" * prob.name * "-RM.png"
    savefig(plt_rm, filename)
    @info "$(now()) - Plot RM saved to $(filename)"

    plt_rmp = plot_diagnostic(diag=diagnostics,
        diagnostic_name="rmp",
        title="RMP for Different Schemes",
        label_yaxis="RMP",
        prob=prob)
    filename = path_images * "/" * prob.name * "-RMP.png"
    savefig(plt_rmp, filename)
    @info "$(now()) - Plot RMP saved to $(filename)"

    plt_mdr = plot_diagnostic(diag=diagnostics,
        diagnostic_name="mdr",
        title="MDR for Different Schemes",
        label_yaxis="MDR",
        prob=prob)
    filename = path_images * "/" * prob.name * "-MDR.png"
    display(plt_mdr)
    savefig(plt_mdr, filename)
    @info "$(now()) - Plot MDR saved to $(filename)"

    plt_rmax = plot_diagnostic(diag=diagnostics,
        diagnostic_name="rmax",
        title="RMAX for Different Schemes",
        label_yaxis="RMAX",
        prob=prob)
    filename = path_images * "/" * prob.name * "-RMAX.png"
    savefig(plt_rmax, filename)
    @info "$(now()) - Plot RMAX saved to $(filename)"

    plt_rmin = plot_diagnostic(diag=diagnostics,
        diagnostic_name="rmin",
        title="RMIN for Different Schemes",
        label_yaxis="RMIN",
        prob=prob)
    filename = path_images * "/" * prob.name * "-RMIN.png"
    savefig(plt_rmin, filename)
    @info "$(now()) - Plot RMIN saved to $(filename)"

    # generate the table with the diagnostics
    generate_table(diag=diagnostics,
    indices=idx,
    path_table=path_table, 
    title="Diagnostics of the linear advection problem",
    latex_label=label_table,
    prob=prob)
end

