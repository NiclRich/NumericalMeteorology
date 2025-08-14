using Plots
using PrettyTables
using Dates
import TOML

include("./src/solvers.jl")
include("./src/postprocessing.jl")
include("./src/ConfigLoader.jl")
include("./src/diagnostics.jl")
include("./src/simulation.jl")
include("./src/shallowwater.jl")
include("./src/tex.jl")

using .Solvers

PATH_CONFIG_ADVECTION = "./config_advection.toml"
PATH_CONFIG_SWE = "./config_swe.toml"
PATH_TABLES = "./tex/tables"
PATH_IMAGES = "./images"
GAMMA = 0.1
INDICES_ADVECTION = [1, 5, 10, 90, 180]

@info "$(now()) - Start Simulation of the Advection Schemes"
configurations = parse_config(PATH_CONFIG_ADVECTION)

for (k, conf) in enumerate(configurations)
    prob = Solvers.LinearAdvectionProblem(conf)
    diag, u_approx = simulate_advection(prob, GAMMA)
    post_process(diagnostics=diag,
        numerical_approximations=u_approx,
        indices=INDICES_ADVECTION, 
        path_images=PATH_IMAGES,
        path_table=PATH_TABLES, label_table="tab:$k",
        prob=prob)
end

@info "$(now()) - Start Simulations of Shallow Water Equation"
simulate_swe1d(config_path="./config_swe.toml", img_path=PATH_IMAGES)
simulate_swe2d(config_path="./config_swe.toml", img_path=PATH_IMAGES)
@info "$(now()) - Simulation of the Shallow Water Equation completed!"


# compile latex document automatically
@info "$(now()) - Start Compilation of the LaTeX Document"
run(`bash -c "mv ./images/*png ./tex/images/"`)
pdfpath = compile_latex_silent(workdir="./tex")
@info "$(now()) - LaTeX Compilation finished!\nDocument can be found under: $(pdfpath)"
    