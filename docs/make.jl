using CATN
using Documenter

DocMeta.setdocmeta!(CATN, :DocTestSetup, :(using CATN); recursive=true)

makedocs(;
    modules=[CATN],
    authors="Xuanzhao Gao <xgao@flatironinstitute.org> and contributors",
    sitename="CATN.jl",
    format=Documenter.HTML(;
        canonical="https://xuanzhaogao.github.io/CATN.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/xuanzhaogao/CATN.jl",
    devbranch="main",
)
