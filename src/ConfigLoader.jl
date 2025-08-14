import TOML

function parse_config(path)
    config = TOML.parsefile(path)

    # add name to the configuration
    for (key, val) in config
        val["name"] = key
        # convert initial function from string to function
        init_str = val["initial"]
        expr = Meta.parse(init_str)
        initial_func = eval(expr)
        val["initial"] = initial_func
    end

    return collect(values(config))
end

    