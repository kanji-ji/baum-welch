using StatsBase

x = countmap(split("a"))

addcounts!(x,split("ab cd ef", " "))

