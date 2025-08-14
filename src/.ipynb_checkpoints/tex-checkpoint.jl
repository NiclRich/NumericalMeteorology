function compile_latex_silent(; texfile::AbstractString = "main.tex",
                               workdir::AbstractString = pwd(),
                               clean_intermediates::Bool = false)
    latexmk = Sys.which("latexmk")
    latexmk === nothing && error("latexmk not found on PATH. Please install TeX Live/MiKTeX and latexmk.")

    # Ensure we run in the given directory (so latexmk finds aux files)
    pdfpath = nothing
    cd(workdir) do
        # Build PDF quietly; suppress all stdout/stderr to the Julia console
        cmd_build = `$latexmk -pdf -interaction=nonstopmode -halt-on-error -silent $texfile`
        ok = success(pipeline(cmd_build, stdout=devnull, stderr=devnull))
        ok || error("LaTeX build failed (see log files in $(workdir)).")

        # Optionally clean intermediates while keeping the PDF
        if clean_intermediates
            cmd_clean = `$latexmk -c`
            success(pipeline(cmd_clean, stdout=devnull, stderr=devnull))
        end

        pdfpath = abspath(replace(texfile, r"\.tex$" => ".pdf"))
    end
    return pdfpath
end

compile_latex_silent()
