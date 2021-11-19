from .segement import Sphere, Frustum
import numpy as np
from copy import deepcopy as dcp
import pymeshlab as mlab
from trimesh import Trimesh


class Swc2mesh():
    """Convert neuronal SWC files to watertight surface meshs.

    Details about SWC format can be found [here](http://neuromorpho.org/myfaq.jsp#QS3).
    """
    types = ('undefined',
             'soma',
             'axon',
             'basal_dendrite',
             'apical_dendrite',
             'custom',
             'unspecified_neurites',
             'glia_processes',
             'neuron')
    colors = ('black',  # 'undefined'
              'red',    # 'soma'
              'gray',   # 'axon'
              'green',  # 'basal_dendrite'
              'magenta',# 'apical_dendrite'
              'yellow', # 'custom'
              'pink',   # 'unspecified_neurites'
              'blue')   # 'glia_processes'

    def __init__(self, file=None, soma_shape='sphere', 
                 to_origin=True, use_scale = False,
                 **kwargs) -> None:
        self.file = file
        if soma_shape in ['sphere']:
            self.soma_shape = soma_shape
        else:
            raise NotImplementedError(f'{soma_shape} not implemented.')
        self.to_origin = to_origin
        self.use_scale = use_scale
        self.scale = np.ones(3)
        self.geoms = {k: [] for k in self.types}
        self.meshes = {k: [] for k in self.types}
        if self.file:
            self.read_swc()
        else:
            self.nodes = []
        self.kwargs = kwargs
        pass # get swc and mesh measurement

    def read_swc(self, file=None) -> list:
        if (not file) and (not self.file):
            raise RuntimeError('Please provide SWC file.')
        if file:
            self.file = file    
        swc = self._parse_swc()
        swc = self._process_soma(swc)
        self.nodes = self._create_nodes(swc)
        return self.nodes

    def _parse_swc(self):
        file = self.file
        # check file name
        if not file.lower().endswith('.swc'):
            Warning(f'{file} may not be in the SWC format.')
        swc = {'soma': [], 'neurites': []}
        # read swc file
        first_line = True
        with open(file, 'r') as f:
            for iline in f:
                line = iline.strip().lower().split(' ')
                if self.use_scale and 'scale' in line:
                    if line[0] == '#':
                        self.scale[0] = float(line[2])
                        self.scale[1] = float(line[3])
                        self.scale[2] = float(line[4])
                    else:
                        self.scale[0] = float(line[1])
                        self.scale[1] = float(line[2])
                        self.scale[2] = float(line[3])
                if len(line) != 7: continue

                if line[0].isnumeric():
                    # check the parent compartment of the first point
                    if first_line and int(line[6]) != -1:
                        raise ValueError('Parent of the first node must be -1.')
                    else:
                        first_line = False
                    # get data
                    id, type = int(line[0]) - 1, int(line[1])
                    position = np.array([float(line[2]), float(line[3]), float(line[4])])
                    position *= self.scale
                    radius, parent_id = float(line[5]), int(line[6]) - 1
                    # check data
                    if parent_id < 0: parent_id = -1
                    if parent_id == -1 and type != 1:
                        type = 1
                        Warning('No soma. Convert the first point to soma.')
                    if parent_id >= id:
                        raise ValueError("Parent id must be less than children id.")
                    if id < 0: raise ValueError('Negative compartment ID.')
                    if radius <= 0: raise ValueError('Negative radius.')
                    if type < 0 or type > 7:
                        raise TypeError('Undefined geomal compartment type.')
                    data = {
                        'id': id,
                        'type': type,
                        'position': position,
                        'radius': radius,
                        'parent_id': parent_id,
                        'children_id': []
                    }
                    if type == 1:
                        swc['soma'].append(data)
                    else:
                        swc['neurites'].append(data)
                else:
                    continue
        return swc

    def _process_soma(self, swc):
        soma_swc = swc['soma']
        if self.soma_shape == 'sphere':
            radius = 0
            position = np.zeros(3)
            # get average radius and position
            for isoma in soma_swc:
                radius = np.max([radius, isoma['radius']])
                position += isoma['position']
            position = position / len(soma_swc)
            # define soma as a sphere
            for isoma in soma_swc:
                isoma['radius'] = radius
                isoma['position'] = position
                isoma['parent_id'] = -1
            # NOTE: maybe unnecessary
            swc['soma'] = soma_swc
        else:
            raise NotImplementedError(f'soma type {self.soma_shape} not implemented.')
        return swc

    def _create_nodes(self, swc):
        n_compartment = len(swc['soma']) + len(swc['neurites'])
        nodes = [0] * n_compartment
        for icmpt in swc['soma'] + swc['neurites']:
            if nodes[icmpt['id']] == 0:
                nodes[icmpt['id']] = dcp(icmpt)
                if self.to_origin:
                    # move soma center to origin
                    nodes[icmpt['id']]['position'] -= swc['soma'][0]['position']
            else:
                raise ValueError('Invalid swc file. \
                    Every node can only be defined once.')
        # add children
        for ind in range(n_compartment):
            parent_id = nodes[ind]['parent_id']
            if parent_id != -1:
                if nodes[parent_id]['type'] != 1:
                    nodes[parent_id]['children_id'].append(nodes[ind]['id'])
                else:
                    # parent is soma
                    nodes[0]['children_id'].append(nodes[ind]['id'])
        return nodes

    def generate(self,
                 savename=None,
                 compartment='neuron',
                 post_cleaning=False,
                 simplify=True,
                 **kwargs):
        # TODO
        self._create_geoms_list(compartment)
        for geoms_icmpt in self.geoms[compartment]:
            mesh_icmpt = self._build_mesh(geoms_icmpt, savename)
            self.meshes[compartment].append(mesh_icmpt)

    def _create_geoms_list(self, compartment) -> None:
        if isinstance(compartment, str):
            compartment = self.types.index(compartment)
        if not isinstance(compartment, int) or compartment > 8:
            raise ValueError(f'Compartment {compartment} is illegal.')
        
        nodes = self.nodes
        geoms = []
        if compartment in [1, 8]: # build soma or neuron
            # add soma to geoms
            if self.soma_shape == 'sphere':
                geoms.append(Sphere(nodes[0]))
            else:
                raise NotImplementedError(
                    f'soma type {self.soma_shape} not implemented.')
            if compartment == 1:
                self._check_all_intersect(geoms)
                self.geoms['soma'].append(geoms)
                return None

            # build the whole neuron, start with soma's children
            parent_id = 0   # index in the nodes
            parent_geoms_index = 0   # index in the geoms
            for child_id in nodes[parent_id]['children_id']:
                start = dcp(nodes[parent_id])
                start['radius'] = nodes[child_id]['radius']
                end = nodes[child_id]
                # add new geom
                child_geoms_index = len(geoms)
                geoms.append(Frustum(start, end))
                self._parent_child_intersect(geoms, parent_geoms_index, child_geoms_index)
                # create subsequent frustums of child_id (deep-first)
                self._add_frustums(geoms, nodes, child_id, child_geoms_index)
            self._check_all_intersect(geoms)
            self.geoms['neuron'].append(geoms)
        else:
            # TODO: geoms list for neurites
            raise NotImplementedError
        return None

    def _add_frustums(self, geoms, nodes, parent_id, parent_geoms_index):
        nchildren = len(nodes[parent_id]['children_id'])
        # without bifurcation, looping to add segements
        while nchildren == 1:
            # add current frustum
            child_id = nodes[parent_id]['children_id'][0]
            child_geoms_index = len(geoms)
            geoms.append(Frustum(nodes[parent_id], nodes[child_id]))
            self._parent_child_intersect(geoms, parent_geoms_index, child_geoms_index)
            # create subsequent frustums
            parent_id = child_id
            parent_geoms_index = child_geoms_index
            nchildren = len(nodes[parent_id]['children_id'])
        if nchildren == 0:
            return 0
        elif nchildren > 1:
            # with bifurcation, add segements recursively (deep-first)
            for child_id in nodes[parent_id]['children_id']:
                # add current frustum
                child_geoms_index = len(geoms)
                geoms.append(Frustum(nodes[parent_id], nodes[child_id]))
                self._parent_child_intersect(geoms, parent_geoms_index, child_geoms_index)
                # create subsequent frustums
                self._add_frustums(geoms, nodes, child_id, child_geoms_index)
    
    @staticmethod
    def _parent_child_intersect(geoms, parent_index, child_index) -> None:
        # update parent
        [_, on, outer] = geoms[child_index].intersect(geoms[parent_index])
        geoms[parent_index].update(np.logical_or(on, outer))
        # update child
        [_, on, outer] = geoms[parent_index].intersect(geoms[child_index])
        geoms[child_index].update(np.logical_or(on, outer))

    def _check_all_intersect(self, geoms):
        len_geoms = len(geoms)
        indices = ((i, j) for i in range(len_geoms -1) 
                        for j in range(i+1, len_geoms))
        for i, j in indices:
            if len(geoms[i]) == 0 or len(geoms[j]) == 0:
                continue
            if geoms[i].fast_intersect(geoms[j]):
                self._parent_child_intersect(geoms, i, j)
        return geoms

    def _build_mesh(self, geoms, savename):
        # TODO
        point_list, normal_list = [], []
        for igeoms in geoms:
            p, n = igeoms.output()
            point_list.append(p)
            # normal_list.append(n)
        points = np.concatenate(point_list, axis=1)

        # with open("simple.obj", "w") as f:
        #     for ii in range(points.shape[1]):
        #         p_str = f"v {points[0, ii]} {points[1, ii]} {points[2, ii]}\n"
        #         f.write(p_str)
        # f.close()

        # normals = np.concatenate(normal_list, axis=1)
        normals = self._estimate_normals(points)
        normals = self._fix_normals(points, normals, geoms)
        # build surface
        ms = mlab.MeshSet()
        m = mlab.Mesh(vertex_matrix = points.T, 
                        v_normals_matrix = normals.T)
        ms.add_mesh(m)
        ms.surface_reconstruction_screened_poisson(depth=10)
        # ms = self._post_cleaning(ms)
        # ms = self._simplification(ms)
        if savename:
            ms.save_current_mesh(savename)
        m = ms.current_mesh()
        tmesh = Trimesh(
                    process = False,
                    use_embree = False,
                    vertices = m.vertex_matrix(),
                    faces = m.face_matrix(),
                    face_normals = m.face_normal_matrix(),
                    vertex_normals = m.vertex_normal_matrix(),
                    # face_colors = m.face_color_matrix(),
                    # vertex_colors = m.vertex_color_matrix()
                )
        return tmesh

    def _post_cleaning(self, ms):
        # TODO
        return ms

    def _simplification(self, ms):
        # TODO
        return ms

    def _fix_normals(self, points, normals, geoms):
        start = 0
        end = 0
        for igeom in geoms:
            end += len(igeom)
            p = points[:, start:end]
            n = normals[:, start:end]
            normals[:, start:end] = igeom.fix_normals(p, n)
            start = end
        return normals

    @staticmethod
    def _estimate_normals(points):
        m = mlab.Mesh(vertex_matrix = points.T)
        ms = mlab.MeshSet()
        ms.add_mesh(m)
        ms.compute_normals_for_point_sets(k=10)
        return ms.current_mesh().vertex_normal_matrix().T
