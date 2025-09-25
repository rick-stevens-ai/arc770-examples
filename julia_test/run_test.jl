push!(LOAD_PATH, "./src")
include("src/OneAPITest.jl")
using .OneAPITest
OneAPITest.run_test()
