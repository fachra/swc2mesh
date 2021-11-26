from .segement import Sphere, Frustum, Ellipsoid, Cylinder, Contour
import numpy as np
from copy import deepcopy as dcp
import pymeshlab as mlab
from os.path import splitext
from multiprocessing import Pool
from scipy.io import savemat
import time


class Swc2mesh():
    """Convert neuronal SWC files to watertight surface meshs.

    Details about SWC format can be found [here](http://neuromorpho.org/myfaq.jsp#QS3).
    """
    # compartment types
    types = (
        'undefined',
        'soma',
        'axon',
        'basal_dendrite',
        'apical_dendrite',
        'custom',
        'unspecified_neurites',
        'glia_processes'
    )
    # compartment colors
    colors = (
        'black',  # 'undefined'
        'red',    # 'soma'
        'gray',   # 'axon'
        'green',  # 'basal_dendrite'
        'magenta',# 'apical_dendrite'
        'yellow', # 'custom'
        'pink',   # 'unspecified_neurites'
        'blue'   # 'glia_processes'
    )
    soma_types = (
        'sphere',
        'ellipsoid',
        'cylinder',
        'contour'
    )

    def __init__(self,
                 file=None,
                 soma_shape='sphere',
                 to_origin=True,
                 use_scale=False
                ) -> None:
        # TODO:
        # 1. simplification
        # 2. post-cleaning
        # 3. adding colors
        # 4. swc and mesh measurement
        self.file = file
        self.soma_shape = soma_shape
        self.to_origin = to_origin
        self.use_scale = use_scale
        self.scale = np.ones(3)
        self.meshes = dict()
        if self.file:
            self.read_swc()
        pass # get swc and mesh measurement

    def read_swc(self, file=None):
        # check SWC file
        if (file is None) and (self.file is None):
            raise RuntimeError('Please provide SWC file.')
        if file:
            self.file = file
        if not self.file.lower().endswith('.swc'):
            Warning(f'{self.file} may not be in the SWC format.')
        
        self.swc = self._parse_swc()
        self.nodes = self._create_nodes()

    def generate(self,
                 savename=None,
                 compartment='neuron',
                 density=1.0,
                 cleaning=False,
                 simplification=True,
                ) -> None:
        """legal compartment list = ['undefined', ..., 'glia_processes',
            'neuron', 'all', 'soma+...']"""
        if compartment == 'all':
            # create meshes for all compartments and the neuron
            for cmpt in self.types + ('neuron',):
                self.generate(savename, cmpt, density,
                    cleaning, simplification)
        else:
            self.density = density
            geoms = self._create_geoms_list(compartment)
            self.meshes[compartment] = []
            for ind, igeom in enumerate(geoms):
                if len(geoms) == 1:
                    name = self._create_name(savename, compartment)
                else:
                    name = self._create_name(savename, compartment, ind)
                imesh = self._build_mesh(
                    igeom, name, cleaning, simplification)
                self.meshes[compartment].append(imesh)

    def _create_geoms_list(self, compartment):
        cmpt_id = self._cmpt_number(compartment)
        geoms = []
        if cmpt_id == 1:
            # only soma
            geoms.append(self._create_soma())
        elif cmpt_id == 8:
            # neuron
            geoms.append(self._create_soma())
            self._add_neurites(geoms)
        elif cmpt_id in [0] + list(range(2,8)):
            # only neurites
            self._add_neurites(geoms, type=cmpt_id)
        elif cmpt_id >= 10: 
            # soma+neurite
            geoms.append(self._create_soma())
            self._add_neurites(geoms, type=cmpt_id%10)
        return geoms

    def _cmpt_number(self, compartment):
        if compartment in self.types:
            cmpt_id = self.types.index(compartment)
        elif compartment == 'neuron':
            cmpt_id = 8
        elif '+' in compartment:
            compartment = compartment.split('+')
            if compartment[0] == 'soma' \
                and compartment[1] in self.types \
                and compartment[1] != 'soma':
                cmpt_id = 10 + self.types.index(compartment)
            else:
                raise ValueError(
                    f'Compartment {compartment} is illegal.')    
        else:
            raise ValueError(
                f'Compartment {compartment} is illegal.')
        return cmpt_id

    def _build_mesh(self, geom, savename, cleaning, simplification):
        # TODO
        # 1. merge theoretical normals and estimated normals
        point_list, normal_list = [], []
        for igeom in geom:
            p, n = igeom.output()
            point_list.append(p)
            normal_list.append(n)
        points = np.concatenate(point_list, axis=1)
        normals = np.concatenate(normal_list, axis=1)
        # normals_esti = self._estimate_normals(points)
        # normals_esti = self._fix_normals(points, normals_esti, geom)
        # build surface
        ms = mlab.MeshSet()
        m = mlab.Mesh(
                vertex_matrix = points.T, 
                v_normals_matrix = normals.T
            )
        ms.add_mesh(m)
        print('build mesh')
        s = time.time()
        ms.surface_reconstruction_screened_poisson(depth=18)
        print(time.time() - s)
        # ms = self._post_cleaning(ms)
        # ms = self._simplification(ms)
        if savename:
            ms.save_current_mesh(savename)
        m = ms.current_mesh()
        return m

    def _create_soma(self):
        if self.soma_shape == 'sphere':
            soma = Sphere(self.nodes[0], self.density)
        elif self.soma_shape == 'ellipsoid':
            soma = Ellipsoid(self.nodes[:3], self.density)
        elif self.soma_shape == 'cylinder':
            soma = Cylinder(self.nodes[:3], self.density)
        elif self.soma_shape == 'contour':
            soma = Contour(
                    self.nodes[:len(self.swc['soma'])], self.density)
        return [soma]

    def _add_neurites(self, geoms, type=None) -> None:
        if type in self.types:
            type = self.types.index(type)
        nodes = self.nodes
        d = self.density
        if geoms:
            # soma is in geoms
            geom = geoms[0]
            parent_id = 0
            parent_geom_index = 0
            for child_id in nodes[parent_id]['children_id']:
                if type and nodes[child_id]['type'] != type:
                    continue
                start = dcp(nodes[parent_id])
                start['radius'] = nodes[child_id]['radius']
                end = nodes[child_id]
                # add new geom
                child_geom_index = len(geom)
                geom.append(Frustum(start, end, d))
                self._parent_child_intersect(geom, parent_geom_index, child_geom_index)
                # create subsequent frustums of child_id (deep-first)
                self._add_frustums(geom, child_id, child_geom_index, type)
            self._check_all_intersect(geom)
        else:
            # soma is not in geoms, add all neurites
            soma_id = 0
            for child_id in nodes[soma_id]['children_id']:
                if type and nodes[child_id]['type'] != type:
                    continue
                igeom = []
                # create the first neurite segement that is outside soma
                start = dcp(nodes[soma_id])
                start['radius'] = nodes[child_id]['radius']
                end = nodes[child_id]
                child_geom_index = len(igeom)
                igeom.append(Frustum(start, end, d))
                # create subsequent frustums of child_id (deep-first)
                self._add_frustums(igeom, child_id, child_geom_index, type)
                self._check_all_intersect(igeom)
                geoms.append(igeom)

    def _add_frustums(self, geom, parent_id, parent_geom_index, type=None):
        d = self.density
        nodes = self.nodes
        while len(nodes[parent_id]['children_id']) == 1:
            # without bifurcation, looping to add segements
            child_id = nodes[parent_id]['children_id'][0]
            child_geom_index = len(geom)
            geom.append(Frustum(nodes[parent_id], nodes[child_id], d))
            self._parent_child_intersect(geom, parent_geom_index, child_geom_index)
            # create next frustums
            parent_id = child_id
            parent_geom_index = child_geom_index
        if len(nodes[parent_id]['children_id']) > 1:
            # with bifurcation, add segements recursively (deep-first)
            for child_id in nodes[parent_id]['children_id']:
                if type and nodes[child_id]['type'] != type:
                    continue
                child_geom_index = len(geom)
                geom.append(Frustum(nodes[parent_id], nodes[child_id], d))
                self._parent_child_intersect(geom, parent_geom_index, child_geom_index)
                # create next frustums
                self._add_frustums(geom, child_id, child_geom_index, type)
        elif len(nodes[parent_id]['children_id']) == 0:
            # no children, stop adding frustums
            return 0
    
    def _check_all_intersect(self, geom):
        indices = self.aabb(geom)
        for i, j in indices:
            if len(geom[i]) != 0 and len(geom[j]) != 0:
                self._parent_child_intersect(geom, i, j)
        return geom

    @staticmethod
    def _parent_child_intersect(geom, p, c) -> None:
        # update parent
        [_, on, outer] = geom[c].intersect(geom[p])
        geom[p].update(np.logical_or(on, outer))
        # update child
        [_, _, outer] = geom[p].intersect(geom[c])
        geom[c].update(outer)

    def _post_cleaning(self, ms):
        # TODO
        return ms

    def _simplification(self, ms):
        # TODO
        return ms

    def _fix_normals(self, points, normals, geom):
        start = 0
        end = 0
        for igeom in geom:
            end += len(igeom)
            p = points[:, start:end]
            n = normals[:, start:end]
            normals[:, start:end] = igeom.fix_normals(p, n)
            start = end
        return normals

    @staticmethod
    def _create_name(savename, compartment, i=None):
        if not savename:
            return savename
        root, ext = splitext(savename)
        if not ext:
            ext = ".ply"
        if i:
            name = f"{root}_{compartment}_{i}" + ext
        else:
            name = f"{root}_{compartment}" + ext
        return name

    def _parse_swc(self):
        swc = {
            'soma': [],
            'neurites': []
        }
        # read swc file
        first_node = True
        with open(self.file, 'r') as f:
            for iline in f:
                line = iline.strip().lower().split()
                if self.use_scale and 'scale' in line:
                    if line[0] == '#':
                        self.scale = np.array(line[2:5], dtype=float)
                    else:
                        self.scale = np.array(line[1:4], dtype=float)
                if len(line) == 7 and line[0].isnumeric():
                    # check the parent compartment of the first point
                    if first_node and int(line[6]) != -1:
                        raise ValueError(
                            'Parent of the first node must be -1.')
                    else:
                        first_node = False
                    # get entry
                    id, type = int(line[0])-1, int(line[1])
                    position = self.scale * np.array(line[2:5], dtype=float)
                    radius, parent_id = float(line[5]), int(line[6])-1
                    # check entry
                    if parent_id < 0: parent_id = -1
                    if parent_id == -1 and type != 1:
                        type = 1
                        Warning('Soma absent. Convert the first point to soma.')
                    if parent_id >= id:
                        raise ValueError("Parent id must be less than children id.")
                    if id < 0: raise ValueError('Negative compartment ID.')
                    if radius <= 0: raise ValueError('Negative radius.')
                    if type < 0 or type > 7:
                        raise TypeError('Undefined neuronal compartment type.')
                    # record entry
                    entry = {
                        'id':           id,
                        'type':         type,
                        'position':     position,
                        'radius':       radius,
                        'parent_id':    parent_id,
                        'children_id':  []
                    }
                    if type == 1:
                        swc['soma'].append(entry)
                    else:
                        swc['neurites'].append(entry)
        self._process_soma(swc['soma'])
        return swc

    def _process_soma(self, soma_swc) -> None:
        # check soma shape
        if self.soma_shape not in self.soma_types:
            raise NotImplementedError(
                f'{self.soma_shape} soma is not implemented.')
        if self.soma_shape in self.soma_types[1:]:
            if len(soma_swc) <= 2:
                Warning(
                    f'Have {len(soma_swc)} soma nodes (< 3). \
                    Change "soma_shape" from {self.soma_shape} to sphere.'
                )
                self.soma_shape = 'sphere'
            elif len(soma_swc) > 3 \
                 and self.soma_shape in self.soma_types[1:3]:
                Warning(
                    f'Have {len(soma_swc)} soma nodes (> 3). \
                    Change "soma_shape" from {self.soma_shape} to contour.'
                )
                self.soma_shape = 'contour'
        # spherical soma
        if self.soma_shape == 'sphere':
            radius = 0
            position = np.zeros(3)
            # get maximum radius and averaged position
            for isoma in soma_swc:
                radius = np.max([radius, isoma['radius']])
                position += isoma['position']
            position = position / len(soma_swc)
            # define soma as a sphere
            for isoma in soma_swc:
                isoma['radius'] = radius
                isoma['position'] = position
                isoma['parent_id'] = -1

    def _create_nodes(self):
        swc = self.swc
        n_compartment = len(swc['soma']) + len(swc['neurites'])
        nodes = [0] * n_compartment
        # create node list
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

    @staticmethod
    def _estimate_normals(points):
        m = mlab.Mesh(vertex_matrix = points.T)
        ms = mlab.MeshSet()
        ms.add_mesh(m)
        ms.compute_normals_for_point_sets(k=10)
        return ms.current_mesh().vertex_normal_matrix().T
    
    @staticmethod
    def aabb(geom):
        len_geom = len(geom)
        indices = []
        aabb_pairs = []
        print('get aabb_pairs')
        s = time.time()
        for i in range(len_geom - 1):
            aabb_i = geom[i].aabb
            for j in range(i+1, len_geom):
                aabb_j = geom[j].aabb
                indices.append((i, j))
                aabb_pairs.append((aabb_i, aabb_j))
        print(time.time() - s)
        with Pool(processes=None) as p:
            flags = p.map(_aabb_collision, aabb_pairs)
        collision_indices = []
        for ind, flag in enumerate(flags):
            if flag:
                collision_indices.append(indices[ind])
        return collision_indices

def _aabb_collision(aabb_pair):
    xa, ya, za = aabb_pair[0]
    xb, yb, zb = aabb_pair[1]
    if xa['max'] <= xb['min'] or xa['min'] >= xb['max'] \
        or ya['max'] <= yb['min'] or ya['min'] >= yb['max'] \
        or za['max'] <= zb['min'] or za['min'] >= zb['max']:
        # no collision
        return False
    else:
        # collision detected
        return True

def show(m) -> None:
    from trimesh import Trimesh
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
    tmesh.show()
