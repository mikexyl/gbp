from graphslam.graph import Graph

g=Graph.from_g2o("data/input_INTEL_g2o.g2o")
g.plot(vertex_markersize=1)
g.calc_chi2()
g.optimize(max_iter=100)
g.plot(vertex_markersize=1) 
