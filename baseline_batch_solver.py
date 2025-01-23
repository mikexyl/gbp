from graphslam.graph import Graph

g=Graph.from_g2o("data/initial.g2o")
g.plot(vertex_markersize=1)
g.calc_chi2()
ret=g.optimize(max_iter=100)
print(ret)
g.plot(vertex_markersize=1) 
