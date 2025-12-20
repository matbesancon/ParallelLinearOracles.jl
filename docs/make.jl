using ParallelLinearOracles
using Documenter

DocMeta.setdocmeta!(ParallelLinearOracles, :DocTestSetup, :(using ParallelLinearOracles); recursive=true)

makedocs(;
    modules=[ParallelLinearOracles],
    authors="matbesancon <mathieu.besancon@gmail.com> and contributors",
    sitename="ParallelLinearOracles.jl",
    format=Documenter.HTML(;
        canonical="https://matbesancon.github.io/ParallelLinearOracles.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/matbesancon/ParallelLinearOracles.jl",
    devbranch="main",
)
