from .segments import (
    Sphere, Frustum, Ellipsoid, Cylinder, Contour
)
from copy import deepcopy as dcp
from multiprocessing import Pool
from os.path import splitext
from trimesh import Trimesh
import pymeshlab as mlab
import numpy as np
import warnings
import time


class Swc2mesh():
    """Build watertight surface meshes based on a SWC file.

    More details about SWC format can be found in neuromorpho.org.
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

    # soma shape types
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
                 use_scale=False,
                 depth=None
                 ) -> None:
        """Preparation for mesh generation.

        Args:
            file (str, optional): path to a SWC file. Defaults to None.
            soma_shape (str, optional): required soma shape. Defaults to 'sphere'.
            to_origin (bool, optional): move soma to origin. Defaults to True.
            use_scale (bool, optional): scale the cell if scale ratio is provided
            by the SWC file. Defaults to False.
            depth (int, optional): the depth of the screened poisson
            surface reconstruction method. Defaults to None.

        Available soma shape:
            'sphere', 'ellipsoid', 'cylinder', 'contour'.
        """

        self.file = file
        self.soma_shape = soma_shape
        self.to_origin = to_origin
        self.use_scale = use_scale
        self.depth = depth
        self.scale = np.ones(3)
        self.meshes = dict()
        self.swc = dict()
        self.nodes = []
        if self.file:
            self.read_swc()

        return None

    # Process SWC files

    def read_swc(self, file=None) -> None:
        """Read the given SWC file and save the node info in `self.nodes`.

        Args:
            file (string, optional): path to the SWC file. Defaults to None.

        Raises:
            RuntimeError: the SWC file is not provided.
        """

        # check the input file
        if (file is None) and (self.file is None):
            raise RuntimeError("Please provide a SWC file.")

        if file:
            self.file = file

        if not self.file.lower().endswith('.swc'):
            msg = f"'{self.file}' may not be in the SWC format."
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        # read the file
        print(f"Reading {self.file} ...")
        start = time.time()
        self.swc = self._parse_swc()
        self.nodes = self._create_nodes()
        print(f"Elapsed time: {time.time() - start:.4f} s.")

        return None

    def _parse_swc(self):
        """Parse the SWC file."""

        # initialization
        swc = {
            'soma': [],
            'neurites': []
        }

        # parse swc file
        with open(self.file, 'r') as f:
            first_node = True

            for iline in f:
                line = iline.strip().lower().split()

                # get the scale
                if self.use_scale and 'scale' in line:
                    if line[0] == '#':
                        self.scale = np.array(line[2:5], dtype=float)
                    else:
                        self.scale = np.array(line[1:4], dtype=float)

                # read nodes
                if len(line) == 7 and line[0].isnumeric():
                    # check the parent compartment of the first node
                    if first_node and int(line[6]) != -1:
                        raise ValueError(
                            "Parent of the first node must be -1.")
                    else:
                        first_node = False

                    # extract info
                    id = int(line[0]) - 1
                    node_type = int(line[1])
                    position = self.scale * np.array(line[2:5], dtype=float)
                    radius = float(line[5])
                    parent_id = int(line[6]) - 1

                    # check parameters
                    if parent_id < 0:
                        parent_id = -1

                    if parent_id == -1 and node_type != 1:
                        node_type = 1
                        msg = "Soma absent. Convert the first point to soma."
                        warnings.warn(msg, RuntimeWarning, stacklevel=2)

                    if parent_id >= id:
                        msg = " ".join((
                            f"Node id {line[0]}:",
                            "parent id must be less than children id."))
                        raise ValueError(msg)

                    if id < 0:
                        msg = f"Node id {line[0]}: negative compartment ID."
                        raise ValueError(msg)

                    if radius <= 0:
                        msg = f"Node id {line[0]}: negative radius."
                        raise ValueError(msg)

                    if node_type < 0 or node_type > 7:
                        msg = " ".join((
                            f"Node id {line[0]}:",
                            "unknown compartment type."))
                        raise TypeError(msg)

                    # record info
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

        # set soma shape based on self.soma_shape
        self._set_soma_shape(swc['soma'])

        return swc

    def _set_soma_shape(self, soma_swc) -> None:
        """Set soma shape based on `self.soma_shape`."""

        # check soma shape
        if self.soma_shape not in self.soma_types:
            msg = f"'{self.soma_shape}' soma is not implemented."
            raise NotImplementedError(msg)

        # if soma shape is cylinder, ellipsoid or contour
        if self.soma_shape in self.soma_types[1:]:
            if len(soma_swc) <= 2:
                msg = " ".join((
                    f"Get {len(soma_swc)} soma nodes (< 3). Set 'soma_shape'",
                    f"to 'sphere' instead of '{self.soma_shape}'"
                ))
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                self.soma_shape = 'sphere'

            elif len(soma_swc) > 3 and self.soma_shape in self.soma_types[1:3]:
                msg = " ".join((
                    f"Get {len(soma_swc)} soma nodes (> 3). Set 'soma_shape'",
                    f"to 'contour' instead of '{self.soma_shape}'"
                ))
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                self.soma_shape = 'contour'

        # some shape is sphere
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

        return None

    def _create_nodes(self):
        """Convert `self.swc` to a list `nodes`."""

        # initialization
        swc = self.swc
        len_swc = len(swc['soma'] + swc['neurites'])
        nodes = [None] * len_swc

        # create node list
        for iswc in swc['soma'] + swc['neurites']:
            if nodes[iswc['id']] is None:
                nodes[iswc['id']] = dcp(iswc)
                if self.to_origin:
                    # move soma center to origin
                    nodes[iswc['id']]['position'] -= swc['soma'][0]['position']
            else:
                msg = f"Node {iswc['id']} has duplicate definition."
                raise ValueError(msg)

        # add children_id
        for ind in range(len_swc):
            parent_id = nodes[ind]['parent_id']

            # skip the first node
            if parent_id == -1:
                continue

            # parent is not soma, add children_id
            if nodes[parent_id]['type'] != 1:
                nodes[parent_id]['children_id'].append(nodes[ind]['id'])

            # parent is soma, current node is not soma
            elif nodes[ind]['type'] != 1:
                nodes[0]['children_id'].append(nodes[ind]['id'])

            # current node is soma
            else:
                nodes[parent_id]['children_id'].append(nodes[ind]['id'])

        return nodes

    # Build watertight meshes

    def build(self,
              meshname=None,
              compartment='cell',
              simplification=False,
              depth=None,
              density=1.0
              ) -> None:
        """Build watertight surface meshes.

        Args:
            meshname (str, optional): filename to which the meshes are saved.
            Defaults to None.
            compartment (str, optional): the required cell compartment.
            Defaults to 'cell'.
            simplification (bool, optional): simplification parameter.
            Defaults to False.
            depth (int, optional): the depth of the screened poisson
            surface reconstruction method. Defaults to None.
            density (float, optional): point cloud density. Defaults to 1.0.

        Basic cell compartments:
            'undefined', 'soma', 'axon', 
            'basal_dendrite', 'apical_dendrite', 
            'custom', 'unspecified_neurites', 
            'glia_processes'.

        Composite cell compartments:
            'soma+"one or several basic compartment"',
            'cell-"one or several basic compartment"',
            e.g., 'soma+basal_dendrite+apical_dendrite',
            'soma+undefined+custom', 'cell-axon',
            'cell-axon-unspecified_neurites'.
            But 'cell-soma' is an invalid compartment.

        Other compartment:
            'all' creates meshes for all compartments,
            'cell' creates a mesh for the whole cell.
        """

        if not self.nodes:
            raise ValueError("No SWC file is provided.")

        # the depth of the screened poisson surface reconstruction method
        if depth is not None:
            self.depth = depth

        # build meshes
        if compartment == 'all':
            # create meshes for all compartments and the entire cell
            for cmpt in self.types + ('cell',):
                self.build(meshname, cmpt, density, depth, simplification)

        else:
            # set the point cloud density
            self.density = density

            # create the list of building units
            print(f"Create '{compartment}' segments.")
            segments = self._create_segments(compartment)

            # build meshes for compartment
            self.meshes[compartment] = []
            for ind, isegs in enumerate(segments):
                # create name if meshname is not None
                if len(segments) == 1:
                    name = self._create_name(meshname, compartment)
                else:
                    name = self._create_name(meshname, compartment, ind)

                print(
                    f"Build mesh for '{compartment}' [{ind+1}/{len(segments)}]")
                self.meshes[compartment].append(
                    self._build_mesh(isegs, name, simplification)
                )

        return None

    def _create_segments(self, compartment):
        """Create the list `segments` containing several sublists
        of compartment's the building units.
        """

        # get the compartment number
        cmpt_type = self._cmpt_number(compartment)

        # save building units in segments
        segments = []

        # case: soma+neurites or cell-neurites
        if isinstance(cmpt_type, list):
            segments.append(self._create_soma())
            self._add_neurites(segments, cmpt_type=cmpt_type)

        # case: soma
        elif cmpt_type == 1:
            segments.append(self._create_soma())

        # case: cell
        elif cmpt_type == 8:
            segments.append(self._create_soma())
            self._add_neurites(segments)

        # case: neurites
        elif cmpt_type in [0, 2, 3, 4, 5, 6, 7]:
            self._add_neurites(segments, cmpt_type=[cmpt_type])

        return segments

    def _cmpt_number(self, compartment):
        """Get the compartment number or a list of compartment numbers."""

        # create a deep copy of the compartment types
        type_list = list(self.types)

        err_msg = f"Unkonwn compartment '{compartment}'."

        # case: soma or neurites
        if compartment in type_list:
            cmpt_id = type_list.index(compartment)

        # case: cell
        elif compartment == 'cell':
            cmpt_id = 8

        # case: soma+neurites
        elif '+' in compartment:
            # mesh consists of soma and some neurites
            # e.g., compartment = 'soma+axon+basal_dendrite'
            # cmpt_id only contains the number of neurites

            cmpt_id = []
            cmpts = compartment.split('+')

            if cmpts[0] == 'soma':
                type_list.remove('soma')
                for icmpt in cmpts[1:]:
                    if icmpt in type_list:
                        cmpt_id.append(self.types.index(icmpt))
                    else:
                        raise ValueError(err_msg)
            else:
                raise ValueError(err_msg)

        # case: cell-neurites
        elif '-' in compartment:
            # mesh consists of the cell except some neurites
            # e.g., compartment = 'cell-basal_dendrite-undefined'
            # cmpt_id only contains the number of neurites

            cmpts = compartment.split('-')

            if cmpts[0] == 'cell':
                type_list.remove('soma')
                for icmpt in cmpts[1:]:
                    if icmpt in type_list:
                        type_list.remove(icmpt)
                    else:
                        raise ValueError(err_msg)
            else:
                raise ValueError(err_msg)

            cmpt_id = [self.types.index(t) for t in type_list]

        # case: unkonwn compartment
        else:
            raise ValueError(err_msg)

        return cmpt_id

    def _build_mesh(self, segs, meshname, simplification):
        """Build a single mesh based on a segment list.

        Args:
            segs (list): list of segments.
            meshname (str): filename to which the mesh is saved.
            simplification (bool, float, str): simplification parameter.

        Returns:
            mlab.MeshSet: Meshlab meshset object.
        """

        # initialization
        point_list = []
        normal_list = []
        color_list = []
        r_min = np.inf

        # construct point cloud
        for iseg in segs:
            p, n, c = iseg.output()
            point_list.append(p)
            normal_list.append(n)
            color_list.append(c)
            if isinstance(iseg, Frustum):
                r_min = min(r_min, iseg.r_min)
        points = np.concatenate(point_list, axis=1)
        normals = np.concatenate(normal_list, axis=1)
        colors = np.concatenate(color_list, axis=1)

        if r_min <= 0.1:
            msg = " ".join((
                f"Neuron has some extremely fine neurites ({r_min}um).",
                "Manual post-cleaning might be needed."
            ))
            warnings.warn(msg, UserWarning, stacklevel=2)

        # assemble and clean point cloud
        ms = mlab.MeshSet()
        m = mlab.Mesh(
            vertex_matrix=points.T,
            v_normals_matrix=normals.T,
            v_color_matrix=colors.T
        )
        ms.add_mesh(m)
        ms.remove_duplicate_vertices()
        ms.merge_close_vertices(threshold=mlab.Percentage(1))
        ms.normalize_vertex_normals()
        ms.smooths_normals_on_a_point_sets(k=5)

        # build surface
        depth = self._depth(r_min)
        print("Building mesh ...")
        start = time.time()
        ms.surface_reconstruction_screened_poisson(
            depth=depth,
            preclean=True
        )
        ms = remove_small_components(ms)
        ms = _reset_quality(ms)
        print(f"Elapsed time: {time.time() - start:.4f} s.")

        # post-cleaning
        if simplification:
            print("Simplifying mesh ...")
            start = time.time()
            ms, flag = simplify(ms, simplification)
            print(f"Elapsed time: {time.time() - start:.4f} s.")
        else:
            ms, flag = _fix_mesh(ms)

        if not flag:
            msg = "Mesh is not watertight. Require manual repair."
            warnings.warn(msg, UserWarning, stacklevel=2)

        if meshname.endswith((".ply", ".stl")):
            ms.save_current_mesh(meshname, binary=False)
        elif meshname:
            ms.save_current_mesh(meshname)

        return ms

    def _create_soma(self):
        """Create the soma segment."""

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

    def _add_neurites(self, segments, cmpt_type=None) -> None:
        """Add neurites to `segments`.

        Args:
            segments (list): 2D nested list.
            cmpt_type (list, optional): list of compartment numbers.
            Defaults to None.
        """

        # alias
        nodes = self.nodes
        d = self.density

        if segments:
            # soma is in segments
            seg = segments[0]
            soma_id = 0
            soma_seg_index = 0

            # add neurites to seg
            for child_id in nodes[soma_id]['children_id']:

                # child is a required neurite segment
                if (not cmpt_type or nodes[child_id]['type'] in cmpt_type) \
                        and (nodes[child_id]['type'] != 1):
                    # prepare start and end position
                    start = dcp(nodes[soma_id])
                    start['radius'] = nodes[child_id]['radius']
                    end = nodes[child_id]

                    # add new segment (frustum)
                    child_seg_index = len(seg)
                    seg.append(Frustum(start, end, d))
                    self._parent_child_intersect(
                        seg, soma_seg_index, child_seg_index)

                    # create subsequent segments (frustums) of child_id (deep-first)
                    self._add_frustums(
                        seg, child_id, child_seg_index, cmpt_type)

            # check intersection between all building units
            self._check_all_intersect(seg)

        else:
            # soma is not in segments, only add neurites to segments
            soma_id = 0

            # save each neurite into a sublist
            for child_id in nodes[soma_id]['children_id']:

                # child is a neurite and is required
                if (not cmpt_type or nodes[child_id]['type'] in cmpt_type) \
                        and (nodes[child_id]['type'] != 1):
                    # initialize a sublist
                    seg = []

                    # create the first neurite segment starting at the soma center
                    start = dcp(nodes[soma_id])
                    start['radius'] = nodes[child_id]['radius']
                    end = nodes[child_id]
                    seg.append(Frustum(start, end, d))

                    # create subsequent segments (frustums) of child_id (deep-first)
                    self._add_frustums(seg, child_id, 0, cmpt_type)
                    self._check_all_intersect(seg)
                    segments.append(seg)

        return None

    def _add_frustums(self, seg, parent_id, parent_seg_index, cmpt_type=None):
        """Add neurite segments (frustums) to `seg`.

        Args:
            seg (list): list of segments.
            parent_id (int): parent node id in the list `self.nodes`.
            parent_seg_index (int): parent node index in the list `seg`.
            cmpt_type (list, optional): list of required compartment numbers.
            Defaults to None.
        """

        # alias
        d = self.density
        nodes = self.nodes

        # without bifurcation, looping to add segments
        while len(nodes[parent_id]['children_id']) == 1:
            # add current segment
            child_id = nodes[parent_id]['children_id'][0]
            child_seg_index = len(seg)
            seg.append(Frustum(nodes[parent_id], nodes[child_id], d))
            self._parent_child_intersect(
                seg, parent_seg_index, child_seg_index)

            # add next segment
            parent_id = child_id
            parent_seg_index = child_seg_index

        # with bifurcation, add segments recursively (deep-first)
        if len(nodes[parent_id]['children_id']) > 1:
            for child_id in nodes[parent_id]['children_id']:
                if (not cmpt_type or nodes[child_id]['type'] in cmpt_type) \
                        and (nodes[child_id]['type'] != 1):

                    child_seg_index = len(seg)
                    seg.append(Frustum(nodes[parent_id], nodes[child_id], d))
                    self._parent_child_intersect(
                        seg, parent_seg_index, child_seg_index)

                    # add subsequent frustums
                    self._add_frustums(
                        seg, child_id, child_seg_index, cmpt_type)

        # no children, recursion ends
        elif len(nodes[parent_id]['children_id']) == 0:
            return 0

        else:
            msg = " ".join((
                "Unknown error.",
                "Please report it and send your SWC file to the author.",
                "Thanks!"
            ))
            raise ValueError(msg)

    def _check_all_intersect(self, seg):
        """Remove collision points."""

        collision_index_pairs = self.aabb(seg)
        for i, j in collision_index_pairs:
            if len(seg[i]) * len(seg[j]) != 0:
                self._parent_child_intersect(
                    seg, i, j, remove_close_points=True)

        return seg

    @staticmethod
    def _parent_child_intersect(seg, p, c, remove_close_points=False) -> None:
        """Remove collision points in the parent and child nodes.

        Args:
            seg (list): list of segments.
            p (int): parent index in `seg`.
            c (int): child index in `seg`.
            remove_close_points (bool, optional): If this is set to True,
            remove all the points that are nearer than the specified threshold.
            Defaults to False.
        """

        # update parent
        [_, p_on, p_outer, p_out_near] = seg[c].intersect(seg[p])
        seg[p].update(np.logical_or(p_on, p_outer))

        # update child
        [_, _, c_outer, c_out_near] = seg[p].intersect(seg[c])
        seg[c].update(c_outer)

        if remove_close_points:
            # get minimum radius
            if isinstance(seg[p], Frustum):
                r_min = min(seg[p].r_min, seg[c].r_min)
            else:
                r_min = seg[c].r_min

            # get points and normals
            p_points, p_normals, _ = seg[p].output(p_out_near)
            c_points, c_normals, _ = seg[c].output(c_out_near)

            # compute distance between two point clouds
            p_points = p_points.T.reshape((-1, 1, 3))
            c_points = c_points.T.reshape((1, -1, 3))
            dist = np.linalg.norm(p_points - c_points, axis=2)

            # compute angle between normals
            angle = p_normals.T @ c_normals

            # remove close points
            mask_far = (dist >= 0.1*r_min) | (angle >= 0)
            mask_far = mask_far & (dist >= 0.02*r_min)
            if not mask_far.all():
                p_mask = np.all(mask_far, axis=1)
                c_mask = np.all(mask_far, axis=0)

                # remove parent's points
                p_keep = dcp(seg[p].keep)
                p_keep[p_keep & p_out_near] = p_mask
                seg[p].update(p_keep)

                # remove child's points
                c_keep = dcp(seg[c].keep)
                c_keep[c_keep & c_out_near] = c_mask
                seg[c].update(c_keep)

        return None

    def _depth(self, r_min) -> int:
        """Get the depth of the screened poisson surface reconstruction
        method. Those choices are made based on author's experience.

        Args:
            r_min (float): the radius minimum.

        Returns:
            int: the depth.
        """

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
                depth = 15
            elif r_min > 0.05:
                depth = 17
            else:
                depth = 18

        else:
            depth = int(self.depth)

        return depth

    @staticmethod
    def _create_name(meshname, compartment, i=None):
        """Create name for the compartment.

        Args:
            meshname (str or None): the base name.
            compartment (str): compartment.
            i (int, optional): the index of the compartment. Defaults to None.

        Returns:
            str or None: name string.
        """

        if not meshname:
            return meshname

        root, ext = splitext(meshname)

        # set default extension
        if not ext:
            ext = ".ply"

        if i:
            # if compartment index is given
            name = f"{root}_{compartment}_{i}" + ext
        else:
            name = f"{root}_{compartment}" + ext

        return name

    @staticmethod
    def aabb(seg):
        """Get the aabb collision index pairs."""

        # get all indices
        indices = [(i, j) for i in range(len(seg) - 1)
                   for j in range(i+1, len(seg))]

        # get axis-aligned bounding box
        aabbs = [iseg.aabb for iseg in seg]
        aabb_pairs = [(aabbs[i], aabbs[j]) for i, j in indices]

        # detect aabb collision
        with Pool() as p:
            flags = p.map(_aabb_collision, aabb_pairs)
        collision_index_pairs = []

        # assemble the collision index pairs
        for ind, flag in enumerate(flags):
            if flag:
                collision_index_pairs.append(indices[ind])

        return collision_index_pairs

    @staticmethod
    def _estimate_normals(points):
        """Estimate normal vectors of the point cloud `points`."""

        m = mlab.Mesh(vertex_matrix=points.T)
        ms = mlab.MeshSet()
        ms.add_mesh(m)
        ms.compute_normals_for_point_sets(k=10)

        return ms.current_mesh().vertex_normal_matrix().T


def _aabb_collision(aabb_pair):
    """Detect "Axis-Aligned Bounding Box" collision."""

    xa, ya, za = aabb_pair[0]
    xb, yb, zb = aabb_pair[1]

    # no collision
    if xa['max'] <= xb['min'] or xa['min'] >= xb['max'] \
            or ya['max'] <= yb['min'] or ya['min'] >= yb['max'] \
            or za['max'] <= zb['min'] or za['min'] >= zb['max']:
        return False

    # found collision
    else:
        return True


# Post-processing


def simplify(mesh, sim):
    """Reduce the number of faces and vertices of the mesh.

    Args:
        mesh (mlab.MeshSet, mlab.Mesh): Meshlab mesh.
        sim (bool, float, str): simplification parameter.

    Raises:
        TypeError: wrong mesh type.
        ValueError: invalide simplification parameter `sim`.

    Returns:
        tuple: Meshlab mesh and watertight indicator.
    """

    # check mesh type
    if isinstance(mesh, mlab.Mesh):
        ms = mlab.MeshSet()
        ms.add_mesh(mesh)
    elif isinstance(mesh, mlab.MeshSet):
        ms = dcp_meshset(mesh)
    else:
        raise TypeError("Wrong mesh type.")

    # sim is the target reduction percentage
    if isinstance(sim, float) and sim < 1:
        ms.simplification_quadric_edge_collapse_decimation(
            targetperc=sim,
            qualityweight=True,
            preservenormal=True
        )
        ms = remove_small_components(ms)
        ms, flag = _fix_mesh(ms)

    # sim is the target number of faces
    elif (not isinstance(sim, bool)) and isinstance(sim, int):
        ms.simplification_quadric_edge_collapse_decimation(
            targetfacenum=sim,
            qualityweight=True,
            preservenormal=True
        )
        ms = remove_small_components(ms)
        ms, flag = _fix_mesh(ms)

    # sim is a multiplier of the mesh area
    # e.g., sim = '0.8 area'
    elif isinstance(sim, str) and sim.endswith(' area'):
        try:
            target = float(sim[:-5])
        except:
            msg = f"Invalid simplification parameter: {sim}."
            raise ValueError(msg)

        geo_measure = ms.compute_geometric_measures()
        area = geo_measure['surface_area']
        ms.simplification_quadric_edge_collapse_decimation(
            targetfacenum=int(target * area),
            qualityweight=True,
            preservenormal=True
        )
        ms = remove_small_components(ms)
        ms, flag = _fix_mesh(ms)

    # simplify mesh iteratively
    # unitl mesh can not longer be watertight
    else:
        flag = True
        iter = 1

        # get mesh area
        geo_measure = ms.compute_geometric_measures()
        area = geo_measure['surface_area']

        # keep simplifying until the mesh is not watertight
        # or the face number is too small
        while flag and ms.current_mesh().face_number() > area:
            ms_temp = dcp_meshset(ms)
            ms_temp.simplification_quadric_edge_collapse_decimation(
                targetperc=0.8,
                qualityweight=True,
                preservenormal=True
            )
            ms_temp = remove_small_components(ms_temp)
            ms_temp, flag = _fix_mesh(ms_temp)

            # the simplified mesh is watertight
            # go to the next iteration
            if flag:
                ms = dcp_meshset(ms_temp)
                iter = iter + 1

            # the simplified mesh is not watertight
            # return the last watertight mesh
            elif iter > 1:
                flag = True
                break

    return ms, flag


def remove_small_components(mesh):
    """If `mesh` consists of several disconnected components,
    we only keep the largest one.

    Args:
        mesh (mlab.MeshSet or mlab.Mesh): Meshlab mesh.

    Raises:
        TypeError: unknown mesh type.

    Returns:
        mlab.MeshSet or mlab.Mesh: The largest mesh.
    """

    # check mesh type
    if isinstance(mesh, mlab.MeshSet):
        ms = dcp_meshset(mesh)
    elif isinstance(mesh, mlab.Mesh):
        ms = mlab.MeshSet()
        ms.add_mesh(mesh)
    else:
        raise TypeError("Unknown mesh type.")

    # only keep the largest component
    ms.select_small_disconnected_component(nbfaceratio=0.99)
    ms.delete_selected_faces_and_vertices()

    if isinstance(mesh, mlab.MeshSet):
        res = ms
    else:
        res = ms.current_mesh()

    return res


def _fix_mesh(ms):
    """Remove bad faces and close holes.

    Args:
        ms (mlab.MeshSet): Meshlab meshset.

    Returns:
        tuple: mesh after fixing routine 
        and watertight indicator.
    """

    # cleaning
    ms.remove_duplicate_vertices()
    ms.remove_duplicate_faces()
    ms.remove_zero_area_faces()
    ms.remove_unreferenced_vertices()

    # maximum iteration: 5
    for _ in range(5):
        # delete bad faces and vertices
        ms.select_self_intersecting_faces()
        ms.delete_selected_faces_and_vertices()
        ms.remove_t_vertices()
        ms.repair_non_manifold_edges()

        # close holes
        try:
            ms.close_holes()
            ms.laplacian_smooth(stepsmoothnum=6, selected=True)
        except:
            pass

        # delete bad faces and vertices
        ms.select_self_intersecting_faces()
        ms.delete_selected_faces_and_vertices()
        ms.remove_t_vertices()
        ms.repair_non_manifold_edges()

        # close holes
        try:
            ms.close_holes()
            ms.close_holes()
            res = ms.close_holes()
            # mesh is watertight, exit
            if is_watertight(ms) and \
                    (res['closed_holes']+res['new_faces'] == 0):
                return ms, True
        except:
            pass

    return ms, False


def _reset_quality(mesh):
    """Set vertex quality values according to cell compartments.

    The vertex quality values are used in the surface simplification
    method. A vertex with a high quality value will not be simplified 
    and a portion of the mesh with low quality values will be 
    aggressively simplified.

    The soma vertex quality is set to 0.01.
    The axon vertex quality is set to 10000.
    Other vertex quality is 10.

    Args:
        mesh (mlab.MeshSet or mlab.Mesh): Meshlab mesh.

    Returns:
        mlab.MeshSet or mlab.Mesh: Meshlab mesh 
        with vertex quality array.
    """

    # get color matrix
    m = mesh.current_mesh()
    color_matrix = m.vertex_color_matrix()[:, :3]

    # get soma mask according to vertex color
    color_norm = np.linalg.norm(
        color_matrix - np.array([0.77, 0.3, 0.34]), axis=1)
    soma_mask = color_norm < 0.05

    # get axon mask according to vertex color
    color_norm = np.linalg.norm(
        color_matrix - np.array([0.7, 0.7, 0.7]), axis=1)
    axon_mask = color_norm < 0.1

    # set vertex quality values
    quality = np.ones_like(color_norm)*10
    quality[soma_mask] = 0.01
    quality[axon_mask] = 10000

    # create mesh with vertex quality array
    m_new = mlab.Mesh(
        vertex_matrix=m.vertex_matrix(),
        face_matrix=m.face_matrix(),
        v_normals_matrix=m.vertex_normal_matrix(),
        f_normals_matrix=m.face_normal_matrix(),
        v_color_matrix=m.vertex_color_matrix(),
        v_quality_array=quality
    )
    ms = mlab.MeshSet()
    ms.add_mesh(m_new)

    return ms


# tools


def dcp_meshset(meshset):
    """Make a deepcopy of mlab.MeshSet."""

    ms = mlab.MeshSet()
    ms.add_mesh(meshset.current_mesh())

    return ms


def mlab2tmesh(ms):
    """Convert Meshlab mesh to Trimesh object.

    Args:
        ms (mlab.MeshSet, mlab.Mesh): Meshlab mesh.

    Raises:
        TypeError: wrong mesh type.

    Returns:
        Trimesh: Trimesh object.
    """

    # check mesh type
    if isinstance(ms, mlab.MeshSet):
        m = ms.current_mesh()
    elif isinstance(ms, mlab.Mesh):
        m = ms
    else:
        raise TypeError("Unknown mesh type.")

    # convert mlab.Mesh to Trimesh
    tmesh = Trimesh(
        process=False,
        use_embree=False,
        vertices=m.vertex_matrix(),
        faces=m.face_matrix(),
        face_normals=m.face_normal_matrix(),
        vertex_normals=m.vertex_normal_matrix(),
        vertex_colors=m.vertex_color_matrix()
    )

    return tmesh


def show(ms):
    """Plot meshes.

    Args:
        ms (mlab.Mesh or mlab.MeshSet): Meshlab mesh.
    """

    tmesh = mlab2tmesh(ms)

    return tmesh.show()


def is_watertight(ms):
    """Check whether the mesh is watertight using trimesh routine.

    Args:
        ms (mlab.Mesh or mlab.MeshSet): mesh.
    """

    tmesh = mlab2tmesh(ms)

    return tmesh.is_watertight
