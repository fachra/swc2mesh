from swc2mesh import Swc2mesh

mesh = Swc2mesh('example.swc')
mesh.build('example.ply', compartment='cell', simplification=True)
