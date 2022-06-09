from swc2mesh import Swc2mesh

mesh = Swc2mesh('04b_spindle3aFI.swc')
mesh.build('04b_spindle3aFI.ply', compartment='cell', simplification=True)
