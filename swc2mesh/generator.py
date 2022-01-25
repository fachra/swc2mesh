import warnings
from .segement import Sphere, Frustum, Ellipsoid, Cylinder, Contour
import numpy as np
from copy import deepcopy as dcp
import pymeshlab as mlab
from os.path import splitext
from multiprocessing import Pool
from trimesh import Trimesh
import time


class Swc2mesh():
    """Convert neuronal SWC files to watertight surface meshes.

    More details can be found [here](http://neuromorpho.org/myfaq.jsp#QS3).
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
    soma_types = (
        'sphere',
        'ellipsoid',
        'cylinder',
        'contour'
        )

    def __init__(self,
                 file = None,
                 soma_shape = 'sphere',
                 to_origin = True,
                 use_scale = False,
                 depth = None
                ) -> None:
        # TODO:
        # 1. simplification
        # 2. post-cleaning
        self.file = file
        self.soma_shape = soma_shape
        self.to_origin = to_origin
        self.use_scale = use_scale
        self.depth = depth
        self.scale = np.ones(3)
        self.meshes = dict()
        if self.file:
            self.read_swc()

    def read_swc(self, file=None):
        """Read the given SWC file 
            and save the node info in the list `nodes`.

        Args:
            file (string, optional): path to the SWC file. Defaults to None.

        Raises:
            RuntimeError: the SWC file is not provided.
        """
        # check the SWC file
        if (file is None) and (self.file is None):
            raise RuntimeError('Please provide SWC file.')

        # start reading the file
        if file:
            self.file = file
        if not self.file.lower().endswith('.swc'):
            warnings.warn(f'{self.file} may not be in the SWC format.')

        # parse the file
        print(f"Read {self.file}.")
        self.swc = self._parse_swc()
        self.nodes = self._create_nodes()

    def generate(self,
                 savename = None,
                 compartment = 'neuron',
                 density = 1.0,
                 depth = None,
                 simplification = False
                ) -> None:
        """legal compartment list = ['undefined', ..., 'glia_processes',
            'neuron', 'all', 'soma+...']"""
        # the depth of the screened poisson surface reconstruction method
        if depth is not None:
            self.depth = depth

        # generate meshes according to the parameter compartment
        if compartment == 'all':
            # create meshes for all compartments and the neuron
            for cmpt in self.types + ('neuron',):
                self.generate(savename, cmpt, density, depth, simplification)
        else:
            # set the point cloud density
            self.density = density

            print(f"Construct {compartment} segments.")
            # create the list containing the building units
            geoms = self._create_geoms_list(compartment)
            # build mesh for a certain compartment
            self.meshes[compartment] = []
            for ind, igeom in enumerate(geoms):
                # create name if savename is not None
                if len(geoms) == 1:
                    name = self._create_name(savename, compartment)
                else:
                    name = self._create_name(savename, compartment, ind)
                # mesh generation
                print(f"Generate mesh for {compartment} compartment. \
                        [{ind+1}/{len(geoms)}]")
                imesh = self._build_mesh(igeom, name, simplification)
                self.meshes[compartment].append(imesh)

    def _create_geoms_list(self, compartment):
        """Create the list `geoms` containing the building units
            for the compartment.
        """
        # get the compartment number
        cmpt_type = self._cmpt_number(compartment)
        # save building units in geoms
        geoms = []
        if isinstance(cmpt_type, list):
            # case: soma+neurites or neuron-neurites
            geoms.append(self._create_soma())
            self._add_neurites(geoms, cmpt_type=cmpt_type)
        elif cmpt_type == 1:
            # case: soma
            geoms.append(self._create_soma())
        elif cmpt_type == 8:
            # case: neuron
            geoms.append(self._create_soma())
            self._add_neurites(geoms)
        elif cmpt_type in [0, 2, 3, 4, 5, 6, 7]:
            # case: only a neurite
            self._add_neurites(geoms, cmpt_type=[cmpt_type])
        return geoms

    def _cmpt_number(self, compartment):
        """Get the compartment number or a list of compartment numbers."""
        # create a deep copy of the compartment types
        type_list = dcp(self.types)

        # get the compartment number or the number list
        if compartment in type_list:
            cmpt_id = type_list.index(compartment)

        elif compartment == 'neuron':
            cmpt_id = 8

        elif '+' in compartment:
            # mesh consists of soma and some neurites
            # e.g., 'soma+axon+basal_dendrite'
            # cmpt_id only contains the number of neurites
            cmpts = compartment.split('+')
            cmpt_id = []
            if cmpts[0] == 'soma':
                type_list.remove('soma')
                for icmpt in cmpts[1:]:
                    if icmpt in type_list:
                        cmpt_id.append(self.types.index(icmpt))
                    else:
                        raise ValueError(
                            f'Compartment "{compartment}" is illegal.')
            else:
                raise ValueError(
                    f'Compartment "{compartment}" is illegal.')

        elif '-' in compartment:
            # mesh consists of the neuron except some neurites
            # e.g., 'neuron-basal_dendrite-undefined'
            # cmpt_id only contains the number of neurites
            cmpts = compartment.split('-')
            if cmpts[0] == 'neuron':
                type_list.remove('soma')
                for icmpt in cmpts[1:]:
                    if icmpt in type_list:
                        type_list.remove(icmpt)
                    else:
                        raise ValueError(
                            f'Compartment "{compartment}" is illegal.')
            else:
                raise ValueError(
                    f'Compartment "{compartment}" is illegal.')
            cmpt_id = [self.types.index(t) for t in type_list]

        else:
            raise ValueError(
                f'Compartment "{compartment}" is illegal.')
        return cmpt_id

    def _build_mesh(self, geom, savename, simplification):
        # construct point cloud
        point_list, normal_list = [], []
        quality_list, color_list = [], []
        r_min = np.inf
        for igeom in geom:
            p, n, c, on = igeom.output()
            point_list.append(p)
            normal_list.append(n)
            # color_list.append(c)
            if isinstance(igeom, Frustum):
                r_min = min(r_min, igeom.r_min)
            # q = igeom.quality()
            # quality_list.append(q)

        if r_min <= 0.1:
            warnings.warn(f"Neuron has some extremely fine neurites (< {r_min}um). "\
                + "Manual post-clearning could be needed.")
        
        points = np.concatenate(point_list, axis=1)
        normals = np.concatenate(normal_list, axis=1)
        # colors = np.concatenate(color_list, axis=1)
        # quality = np.concatenate(quality_list, axis=1)

        # normals_esti = self._estimate_normals(points)
        # normals_esti = self._fix_normals(points, normals_esti, geom)
        # merge theoretical normals and estimated normals
        # normals = 0*normals + 0.3*normals_esti
        # normals = normals / np.linalg.norm(normals, axis=0)
        # fix normals_esti based on normals
        # cos_ang = np.einsum('ij,ij->j', normals_esti, normals)
        # normals_esti2 = dcp(normals_esti)
        # normals_esti2[:, cos_ang<0] *= -1

        # with open("ans_esti_fix.ply", "w") as f:
        #     f.write("ply\n")
        #     f.write("format ascii 1.0\n")
        #     f.write(f"element vertex {normals.shape[1]}\n")
        #     f.write(f"property float x\n")
        #     f.write(f"property float y\n")
        #     f.write(f"property float z\n")
        #     f.write(f"property float nx\n")
        #     f.write(f"property float ny\n")
        #     f.write(f"property float nz\n")
        #     f.write(f"end_header\n")
        #     for ii in range(points.shape[1]):
        #         p_str = f"{points[0, ii]} {points[1, ii]} {points[2, ii]} {normals_esti2[0, ii]} {normals_esti2[1, ii]} {normals_esti2[2, ii]}\n"
        #         f.write(p_str)
        # f.close()

        # ms = mlab.MeshSet()
        # m = mlab.Mesh(
        #         vertex_matrix = points.T, 
        #         v_normals_matrix = normals.T
        #     )
        # ms.add_mesh(m)
        # ms.smooths_normals_on_a_point_sets(k=8)
        # nor_smooth = ms.current_mesh().vertex_normal_matrix().T
        # with open("ans_ana_smooth.ply", "w") as f:
        #     f.write("ply\n")
        #     f.write("format ascii 1.0\n")
        #     f.write(f"element vertex {normals.shape[1]}\n")
        #     f.write(f"property float x\n")
        #     f.write(f"property float y\n")
        #     f.write(f"property float z\n")
        #     f.write(f"property float nx\n")
        #     f.write(f"property float ny\n")
        #     f.write(f"property float nz\n")
        #     f.write(f"end_header\n")
        #     for ii in range(points.shape[1]):
        #         p_str = f"{points[0, ii]} {points[1, ii]} {points[2, ii]} {nor_smooth[0, ii]} {nor_smooth[1, ii]} {nor_smooth[2, ii]}\n"
        #         f.write(p_str)
        # f.close()

        # build surface
        ms = mlab.MeshSet()
        m = mlab.Mesh(
                vertex_matrix = points.T, 
                v_normals_matrix = normals.T,
                # v_quality_array = quality.T,
                # v_color_matrix = colors.T
            )
        ms.add_mesh(m)
        ms.remove_duplicate_vertices()
        # ms.merge_close_vertices(threshold=mlab.Percentage(1))
        ms.smooths_normals_on_a_point_sets(k=5)
        ms.normalize_vertex_normals()
        nn = ms.current_mesh().vertex_normal_matrix().T
        pp = ms.current_mesh().vertex_matrix().T
        with open("Anstoetz_PV-IN6_pcd.ply", "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {pp.shape[1]}\n")
            f.write(f"property float x\n")
            f.write(f"property float y\n")
            f.write(f"property float z\n")
            f.write(f"property float nx\n")
            f.write(f"property float ny\n")
            f.write(f"property float nz\n")
            f.write(f"end_header\n")
            for ii in range(pp.shape[1]):
                p_str = f"{pp[0, ii]} {pp[1, ii]} {pp[2, ii]} {nn[0, ii]} {nn[1, ii]} {nn[2, ii]}\n"
                f.write(p_str)
        f.close()

        depth = self._depth(r_min)
        print("Building mesh ...")
        start = time.time()
        ms.surface_reconstruction_screened_poisson(depth=depth)
        ms = remove_small_components(ms)
        print(f"Elapsed time: {time.time() - start:.2f} s.")

        if simplification:
            print("Simplifying mesh ...")
            start = time.time()
            ms = simplify(ms, simplification)
            ms = remove_small_components(ms)
            print(f"Elapsed time: {time.time() - start:.2f} s.")

        if savename:
            ms.save_current_mesh(savename)
        m = ms.current_mesh()
        return m

    def _create_soma(self):
        """Create the building unit for soma."""
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

    def _add_neurites(self, geoms, cmpt_type=None) -> None:
        """Add the neurite building units to geoms.

        Args:
            geoms (list): 2D list containing the building units.
            cmpt_type (list, optional): list of needed compartment numbers. Defaults to None.
        """
        nodes = self.nodes
        d = self.density
        if geoms:
            # soma is in geoms, add neurites to geom
            geom = geoms[0]
            parent_id = 0
            parent_geom_index = 0
            for child_id in nodes[parent_id]['children_id']:
                if (not cmpt_type or nodes[child_id]['type'] in cmpt_type) \
                    and (nodes[child_id]['type'] != 1):
                    # child is neurite and is needed
                    # prepare start and end position
                    start = dcp(nodes[parent_id])
                    start['radius'] = nodes[child_id]['radius']
                    end = nodes[child_id]
                    # add new geom
                    child_geom_index = len(geom)
                    geom.append(Frustum(start, end, d))
                    self._parent_child_intersect(geom, parent_geom_index, child_geom_index)
                    # create subsequent frustums of child_id (deep-first)
                    self._add_frustums(geom, child_id, child_geom_index, cmpt_type)
            
            # check intersection between all building units
            self._check_all_intersect(geom)

        else:
            # geoms is empty, only add neurites to geoms
            soma_id = 0
            for child_id in nodes[soma_id]['children_id']:
                if (not cmpt_type or nodes[child_id]['type'] in cmpt_type) \
                    and (nodes[child_id]['type'] != 1):
                    # child is neurite and is needed
                    igeom = []
                    # create the first neurite segement starting at the soma center
                    start = dcp(nodes[soma_id])
                    start['radius'] = nodes[child_id]['radius']
                    end = nodes[child_id]
                    igeom.append(Frustum(start, end, d))
                    # create subsequent frustums of child_id (deep-first)
                    self._add_frustums(igeom, child_id, 0, cmpt_type)
                    self._check_all_intersect(igeom)
                    geoms.append(igeom)

    def _add_frustums(self, geom, parent_id, parent_geom_index, cmpt_type=None):
        """Add neurite building units (frustums) to `geom`.

        Args:
            geom (list): list of building units.
            parent_id (int): parent node id in the list `nodes`.
            parent_geom_index (int): parent node index in the list `geom`.
            cmpt_type (list, optional): list of needed compartment numbers. Defaults to None.
        """
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
                if (not cmpt_type or nodes[child_id]['type'] in cmpt_type) \
                    and (nodes[child_id]['type'] != 1):
                    child_geom_index = len(geom)
                    geom.append(Frustum(nodes[parent_id], nodes[child_id], d))
                    self._parent_child_intersect(geom, parent_geom_index, child_geom_index)
                    # add subsequent frustums
                    self._add_frustums(geom, child_id, child_geom_index, cmpt_type)

        elif len(nodes[parent_id]['children_id']) == 0:
            # no children, stop adding frustums
            return 0
    
    def _check_all_intersect(self, geom):
        """Remove inner points."""
        collision_index_pairs = self.aabb(geom)
        for i, j in collision_index_pairs:
            if len(geom[i]) != 0 and len(geom[j]) != 0:
                self._parent_child_intersect(geom, i, j)
        return geom

    @staticmethod
    def _parent_child_intersect(geom, p, c) -> None:
        # TODO: 去除两点特别近，法向成钝角
        # update parent
        [_, on, outer, out_near] = geom[c].intersect(geom[p])
        geom[p].update(
            np.logical_or(on, outer),
            np.logical_or(on, out_near)
            )
        # update child
        [_, _, outer, _] = geom[p].intersect(geom[c])
        geom[c].update(outer)

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

    def _depth(self, r_min):
        if self.depth is None:
            if r_min > 2:
                depth = 10
            elif r_min > 1:
                depth = 12
            elif r_min > 0.5:
                depth = 13
            elif r_min > 0.25:
                depth = 14
            elif r_min > 0.1:
                depth = 16
            elif r_min > 0.05:
                depth = 18
            else:
                depth = 20
        else:
            depth = self.depth
        return depth

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
        """Read the SWC file and save the nodes in the dict `swc`."""
        # initialization
        swc = {
            'soma': [],
            'neurites': []
        }
        # read swc file
        with open(self.file, 'r') as f:
            first_node = True
            for iline in f:
                line = iline.strip().lower().split()

                # read the scale array
                if self.use_scale and 'scale' in line:
                    if line[0] == '#':
                        self.scale = np.array(line[2:5], dtype=float)
                    else:
                        self.scale = np.array(line[1:4], dtype=float)
                
                # read the SWC nodes
                if len(line) == 7 and line[0].isnumeric():
                    # check the parent compartment of the first node
                    if first_node and int(line[6]) != -1:
                        raise ValueError(
                            'Parent of the first node must be -1.')
                    else:
                        first_node = False

                    # extract entries
                    id = int(line[0]) - 1
                    node_type = int(line[1])
                    position = self.scale * np.array(line[2:5], dtype=float)
                    radius = float(line[5])
                    parent_id = int(line[6]) - 1

                    # check entries
                    if parent_id < 0: parent_id = -1
                    if parent_id == -1 and node_type != 1:
                        node_type = 1
                        warnings.warn('Soma absent. Convert the first point to soma.')
                    if parent_id >= id:
                        raise ValueError(f"Node id {line[0]}: \
                            parent id must be less than children id.")
                    if id < 0: 
                        raise ValueError(f'Node id {line[0]}: \
                            negative compartment ID.')
                    if radius <= 0:
                        raise ValueError(f'Node id {line[0]}: \
                            negative radius.')
                    if node_type < 0 or node_type > 7:
                        raise TypeError(f'Node id {line[0]}: \
                            undefined neuronal compartment type.')

                    # record entries
                    entry = {
                        'id':           id,
                        'type':         node_type,
                        'position':     position,
                        'radius':       radius,
                        'parent_id':    parent_id,
                        'children_id':  []
                    }
                    if node_type == 1:
                        swc['soma'].append(entry)
                    else:
                        swc['neurites'].append(entry)
        # process soma according to the required soma shape
        self._process_soma(swc['soma'])
        return swc

    def _process_soma(self, soma_swc) -> None:
        """Process the list `soma_swc` 
            according to the required soma shape.
        """
        # check soma shape
        if self.soma_shape not in self.soma_types:
            raise NotImplementedError(
                f'{self.soma_shape} soma is not implemented.')

        # if soma_shape is cylinder, ellipsoid or contour
        if self.soma_shape in self.soma_types[1:]:
            if len(soma_swc) <= 2:
                warnings.warn(
                    f'Have {len(soma_swc)} soma nodes (< 3). \
                    Change "soma_shape" from {self.soma_shape} to sphere.'
                )
                self.soma_shape = 'sphere'
            elif len(soma_swc) > 3 \
                 and self.soma_shape in self.soma_types[1:3]:
                warnings.warn(
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
        """Convert the dict `swc` to a node list `nodes`."""
        # initialization
        swc = self.swc
        len_swc = len(swc['soma']) + len(swc['neurites'])
        nodes = [0] * len_swc
        
        # create node list
        for iswc in swc['soma'] + swc['neurites']:
            if nodes[iswc['id']] == 0:
                nodes[iswc['id']] = dcp(iswc)
                if self.to_origin:
                    # move soma center to origin
                    nodes[iswc['id']]['position'] -= swc['soma'][0]['position']
            else:
                raise ValueError('Invalid swc file. \
                    Every node can only be defined once.')
        
        # add children_id
        for ind in range(len_swc):
            parent_id = nodes[ind]['parent_id']
            if parent_id != -1:
                # current node is not the first node
                if nodes[parent_id]['type'] != 1:
                    # parent is not soma
                    nodes[parent_id]['children_id'].append(nodes[ind]['id'])
                elif nodes[ind]['type'] != 1:
                    # parent is soma, current node is not soma
                    nodes[0]['children_id'].append(nodes[ind]['id'])
                else:
                    # current node is soma
                    nodes[parent_id]['children_id'].append(nodes[ind]['id'])
        return nodes

    @staticmethod
    def _estimate_normals(points):
        """Estimate normal vectors of the point cloud `points`."""
        m = mlab.Mesh(vertex_matrix = points.T)
        ms = mlab.MeshSet()
        ms.add_mesh(m)
        ms.compute_normals_for_point_sets(k=10)
        return ms.current_mesh().vertex_normal_matrix().T
    
    @staticmethod
    def aabb(geom):
        """Get aabb collision index pairs."""
        len_geom = len(geom)
        indices = [(i, j) for i in range(len_geom - 1)
                        for j in range(i+1, len_geom)]
        aabbs = [igeom.aabb for igeom in geom]
        aabb_pairs = [(aabbs[i], aabbs[j]) for i, j in indices]
        with Pool() as p:
            flags = p.map(_aabb_collision, aabb_pairs)
        collision_index_pairs = []
        for ind, flag in enumerate(flags):
            if flag:
                collision_index_pairs.append(indices[ind])
        return collision_index_pairs

def _aabb_collision(aabb_pair):
    """Detect "Axis-Aligned Bounding Box" collision."""
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

def _fix_mesh(ms):
    pass

def simplify(mesh, sim):
    if isinstance(mesh, mlab.Mesh):
        ms = mlab.MeshSet()
        ms.add_mesh(mesh)
    elif isinstance(mesh, mlab.MeshSet):
        ms = dcp(mesh)
    else:
        raise TypeError("Unsupported mesh type.")

    if isinstance(sim, float) and sim < 1:
        # sim is the reduction percentage
        ms.simplification_quadric_edge_collapse_decimation(
            targetperc = sim,
            qualityweight = True,
            preservenormal = True
        )
        flag = _fix_mesh(ms)
    elif isinstance(sim, int):
        # sim is the target number of faces
        ms.simplification_quadric_edge_collapse_decimation(
            targetfacenum = sim,
            qualityweight = True,
            preservenormal = True
        )
        flag = _fix_mesh(ms)
    else:
        # sim is not False or None
        geo_measure = ms.compute_geometric_measures()
        area = geo_measure['surface_area']
        nface = int(50 * area)
        nvertex = ms.current_mesh().vertex_number()
        nface = min(nface, int(nvertex*0.8))
        ms.simplification_quadric_edge_collapse_decimation(
            targetfacenum = nface,
            qualityweight = True,
            preservenormal = True
        )
        flag = _fix_mesh(ms)

        while flag or ms.current_mesh().vertex_number() > 2*area:
            ms_temp = dcp(ms)
            ms_temp.simplification_quadric_edge_collapse_decimation(
                targetperc = 0.5,
                qualityweight = True,
                preservenormal = True
            )
            flag = _fix_mesh(ms_temp)
            if flag: ms = ms_temp
    return ms, flag

def remove_small_components(mesh):
    if isinstance(mesh, mlab.MeshSet):
        ms = mesh
    else:
        raise TypeError("Unsupported mesh type.")
    # only keep the largest component
    ms.select_small_disconnected_component(nbfaceratio=0.99)
    ms.delete_selected_faces_and_vertices()
    return ms

def post_cleaning(ms):
    ms.remove_duplicate_vertices()
    ms.remove_duplicate_faces()
    ms.remove_zero_area_faces()
    ms.remove_t_vertices()
    ms.remove_unreferenced_vertices()

def show(m) -> None:
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
