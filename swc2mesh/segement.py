import numpy as np
from numpy import linalg as LA
from copy import deepcopy as dcp
import pymeshlab as mlab


class Geom():
    """Template for any geometry subclasses.
    
        A Geom's subclass must have three attributes/properties:
            points (ndarray): coordinates of sampled points,
                size: [3 x npoint];
            normals (ndarray): out-pointing normal vectors,
                size: [3 x npoint];
            keep (ndarray): the mask of points to keep.
                Inner points are removed;

        and three methods:
            intersect:
                check intersection with another geometry;
            update:
                update the mask `keep`;
            output:
                output valid points.
    """
    # compartment colors
    colors = np.array([
        [0.4, 0.2, 0.6, 0.6],   # purple, undefined
        [0.77, 0.3, 0.34, 0.6], # red, soma
        [0.7, 0.7, 0.7, 0.6],   # gray, axon
        [0.09, 0.63, 0.52, 0.6],# green, basal_dendrite
        [0.73, 0.33, 0.83, 0.6],# magenta, apical_dendrite
        [0.97, 0.58, 0.02, 0.6],# orange, custom
        [1.0, 0.75, 0.8, 0.6],  # pink, unspecified_neurites
        [0.17, 0.17, 2/3, 0.6]  # blue, glia_processes
        ])

    def __init__(self) -> None:
        self.color = np.array([[1], [1], [1]])
        self.points = None
        self.normals = None
        self.keep = None

    def intersect(self, geom, eps=1e-14):
        """Check intersection with another geometry."""
        raise NotImplementedError

    def fix_normals(self, points, normals):
        """Flip inward normal vectors."""
        raise NotImplementedError
    
    def _create_points(self):
        """Create points and normals on geom surface."""
        raise NotImplementedError
    
    @property
    def area(self):
        raise NotImplementedError

    @property
    def volume(self):
        raise NotImplementedError

    def update(self, mask) -> None:
        """Update the mask `keep` and 'normals'.

        Args:
            mask (ndarray): mask of points.
            on (ndarray): mask of on-boundary points.
        """
        self.keep = np.logical_and(self.keep, mask.reshape(-1))

    def output(self, mask=None):
        """Output all valid points.
            Valid points do not intersect with other geometries. 

        Returns:
            tuple: valid points and their out-pointing normals.
        """
        if mask is not None:
            keep = self.keep & mask
        else:
            keep = self.keep
        
        p = self.points[:, keep]
        n = self.normals[:, keep]
        color = np.repeat(self.color, p.shape[1], axis=1)
        return p, n, color

    def __len__(self) -> int:
        return np.count_nonzero(self.keep)

    @property
    def aabb(self):
        """Axis-aligned bounding box"""
        if self.__len__() < 1:
            x = {'min': -np.inf, 'max': -np.inf}
            y = {'min': -np.inf, 'max': -np.inf}
            z = {'min': -np.inf, 'max': -np.inf}
        else:
            p, _, _ = self.output()
            x = {'min': np.min(p[0,:]), 'max': np.max(p[0,:])}
            y = {'min': np.min(p[1,:]), 'max': np.max(p[1,:])}
            z = {'min': np.min(p[2,:]), 'max': np.max(p[2,:])}
        return x, y, z


class Sphere(Geom):
    """Sphere objects representing somas.

    Attributes:
        r (float): soma radius.
        center (ndarray): soma center, 
            size: [3 x 1].
        points (ndarray): coordinates of sampled points,
            size: [3 x npoint].
        normals (ndarray): out-pointing normal vectors,
            size: [3 x npoint].
        keep (ndarray): the mask of points to keep.
            Inner points are removed.
    """
    def __init__(self, soma, density) -> None:
        """Create a sphere using soma's position and radius.

        Args:
            soma (dict): A dictionary with 
                two keys: `position` and `radius`.
        """
        super().__init__()
        self.color = self.colors[1].reshape(4, 1)
        self.r = soma['radius']
        self.center = soma['position'].reshape(3, 1)
        self.density = density
        self.points, self.normals = self._create_points()
        self.keep = np.full(self.points.shape[1], True)

    def intersect(self, geom, eps=1e-14):
        """Check intersection with another geometry.

        Args:
            geom (Geom): another geometry.
            eps (float, optional): margin of the boundary. Defaults to 1e-14.

        Returns:
            tuple: contains four masks
                `inner`: mask of inner points;
                `on`: mask of points on the boundary;
                `outer`: mask of outer points;
                `out_near`: mask of outer points close to the boundary.
        """
        # compute distance
        dist = LA.norm(geom.points - self.center, axis=0)
        dist = dist - self.r
        # check the intersection
        inner = dist < -eps
        on = (-eps <= dist) & (dist <= eps)
        outer = dist > eps
        out_near = (dist > eps) & (dist < 0.1*geom.r_min)
        return inner, on, outer, out_near

    def fix_normals(self, points, normals):
        """Flip inward normal vectors.

        Args:
            points (ndarray): coordinates of points,
                size: [3 x npoint].
            normals (ndarray): normal vectors,
                size: [3 x npoint].

        Returns:
            ndarray: fixed normal vectors.
        """
        # translate points to local coordinate system
        points -= self.center
        cos_angle = np.einsum('ij,ij->j', points, normals)
        # fix normals
        normals[:, cos_angle<0] *= -1
        return normals

    def _create_points(self):
        """Create points and normals on sphere surface.

        Returns:
            tuple: contains two ndarrays
                `points`: coordinates of sampled points,
                `normals`: out-pointing normal vectors.
        """
        npoint = int(self.density * self.area)
        npoint = np.max([128, npoint])
        normals = unitsphere(int(npoint))
        return self.r * normals + self.center, normals

    @property
    def area(self):
        """Sphere area."""
        return 4*np.pi*self.r**2

    @property
    def volume(self):
        """Sphere volume."""
        return 4*np.pi*self.r**3 / 3


class Ellipsoid(Geom):
    def __init__(self, soma, density) -> None:
        super().__init__()
        self.color = self.colors[1].reshape(4, 1)
        self.center = soma[0]['position'].reshape(3, 1)
        # axes
        self.a = soma[0]['radius']
        self.b = self.a
        self.c_axis = soma[1]['position'] - soma[2]['position']
        self.c_axis = self.c_axis.reshape(3, 1)
        self.c = LA.norm(self.c_axis) / 2
        # transformation
        self._translation = self.center
        self._rotation = self.rotation_matrix
        # point cloud
        self.density = density
        self.points, self.normals = self._create_points()
        self.keep = np.full(self.points.shape[1], True)

    def intersect(self, geom, eps=1e-14):
        """Check intersection with another geometry.

        Args:
            geom (Geom): another geometry.
            eps (float, optional): margin of the boundary. Defaults to 1e-14.

        Returns:
            tuple: contains four masks
                `inner`: mask of inner points;
                `on`: mask of points on the boundary;
                `outer`: mask of outer points;
                `out_near`: mask of outer points close to the boundary.
        """
        # transform points to local coordinate
        points = self._rotation.T @ (geom.points - self._translation)
        
        axes = np.array([[self.a, self.b, self.c]]).T
        dist = LA.norm(points / axes, axis=0) - 1
        # masks
        inner = dist < -eps
        on = (-eps <= dist) & (dist <= eps)
        outer = dist > eps
        out_near = (dist > eps) & (dist < 0.01)
        return inner, on, outer, out_near

    def fix_normals(self, points, normals):
        """Flip inward normal vectors.

        Args:
            points (ndarray): coordinates of points,
                size [3 x npoint].
            normals (ndarray): normal vectors, 
                size [3 x npoint].

        Returns:
            ndarray: fixed normal vectors.
        """
        # translate points to local coordinate system
        points = self._rotation.T @ (points - self._translation)
        normals = self._rotation.T @ normals
        # fix normals
        cos_angle = np.einsum('ij,ij->j', points, normals)
        normals[:, cos_angle<0] *= -1
        # rotate normals back to global coordinate system
        return self._rotation @ normals

    def _create_points(self):
        npoint = int(self.density * self.area)
        npoint = np.max([128, npoint])
        points, normals = ellipsoid(npoint, self.a, self.b, self.c)
        # move the local ellipsoid
        points = self._rotation @ points + self._translation
        normals = self._rotation @ normals
        return points, normals

    @property
    def rotation_matrix(self):
        """Create rotation matrix transforming z-axis to ellipsoid's c-axis.

        Returns:
            ndarray: rotation matrix,
                size: [3 x 3].
        """
        # compute rotation matrix to transform z to c-axis
        z = np.array([[0, 0, 1]]).T
        ax = self.c_axis / (self.c * 2)
        c = ax + z
        # R: matrix transforming z to c-axis
        if LA.norm(c) < 1e-12:
            # ax = -z
            R = np.eye(3)
            R[2, 2] = -1
        else:
            R = 2  * (c @ c.T).T / (c.T @ c) - np.eye(3)
        return R

    @property
    def area(self):
        """
        Ellipsoid area.
        https://en.wikipedia.org/wiki/Ellipsoid#Approximate_formula
        """
        p = 1.6075
        s = 4 * np.pi * ((self.a*self.b)**p / 3 \
            + (self.a*self.c)**p / 3 \
            + (self.b*self.c)**p / 3)**(1/p) 
        return s

    @property
    def volume(self):
        """Ellipsoid volume."""
        vol = 4*np.pi*self.a*self.b*self.c/3
        return vol


class Cylinder(Geom):
    def __init__(self, soma, density) -> None:
        super().__init__()
        self.color = self.colors[1].reshape(4, 1)
        self.center = soma[0]['position'].reshape(3, 1)
        self.r = soma[0]['radius']
        self.axis = soma[1]['position'] - soma[2]['position']
        self.axis = self.axis.reshape(3, 1)
        self.h = LA.norm(self.axis)
        # transformation
        self._translation = self.center
        self._rotation = self.rotation_matrix
        # point cloud
        self.density = density
        self.points, self.normals = self._create_points()
        self.keep = np.full(self.points.shape[1], True)

    def intersect(self, geom, eps=1e-14):
        """Check intersection with another geometry.

        Args:
            geom (Geom): another geometry.
            eps (float, optional): margin of the boundary. Defaults to 1e-14.

        Returns:
            tuple: contains four masks
                `inner`: mask of inner points;
                `on`: mask of points on the boundary;
                `outer`: mask of outer points;
                `out_near`: mask of outer points close to the boundary.
        """
        # transform points to local coordinate
        points = self._rotation.T @ (geom.points - self._translation)
        dist = LA.norm(points[:2, :], axis=0) - self.r

        # masks
        mask_in = (points[2, :] < self.h/2-eps) & (points[2, :] > -self.h/2+eps)
        mask_updown = ((points[2, :] >= self.h/2-eps)&(points[2, :] <= self.h/2+eps)) | \
            ((points[2, :] >= -self.h/2-eps)&(points[2, :] <= -self.h/2+eps))
        
        inner = mask_in & (dist < -eps)
        on = (mask_in & (dist >= -eps) & (dist <= eps)) | (mask_updown & (dist <= 0))
        outer = (dist > eps) | ~(mask_in & mask_updown)
        out_near = ((dist > eps) & (dist < 0.1*geom.r_min) & mask_in) | \
                    ((points[2, :] >= self.h/2+eps) 
                        & (points[2, :] < self.h/2+0.1*geom.r_min) 
                        & (dist <= 0)) | \
                    ((points[2, :] <= -self.h/2-eps) 
                        & (points[2, :] > -self.h/2-0.1*geom.r_min) 
                        & (dist <= 0))
        return inner, on, outer, out_near

    def fix_normals(self, points, normals):
        """Flip inward normal vectors.

        Args:
            points (ndarray): coordinates of points,
                size [3 x npoint].
            normals (ndarray): normal vectors, 
                size [3 x npoint].

        Returns:
            ndarray: fixed normal vectors.
        """
        # translate points to local coordinate system
        points = self._rotation.T @ (points - self._translation)
        normals = self._rotation.T @ normals
        # fix normals
        cos_angle = np.einsum('ij,ij->j', points, normals)
        normals[:, cos_angle<0] *= -1
        # rotate normals back to global coordinate system
        return self._rotation @ normals

    def _create_points(self):
        npoint = int(self.density * self.area)
        npoint = np.max([128, npoint])
        points, normals = cylinder(npoint, self.r, self.h)
        # move the local cylinder
        points = self._rotation @ points + self._translation
        normals = self._rotation @ normals
        return points, normals

    @property
    def rotation_matrix(self):
        """Create rotation matrix transforming z-axis to cylinder's axis.

        Returns:
            ndarray: rotation matrix,
                size: [3 x 3].
        """
        # compute rotation matrix to transform z to axis
        z = np.array([[0, 0, 1]]).T
        ax = self.axis / self.h
        c = ax + z
        # R: matrix transforming z to axis
        if LA.norm(c) < 1e-12:
            # ax = -z
            R = np.eye(3)
            R[2, 2] = -1
        else:
            R = 2  * (c @ c.T).T / (c.T @ c) - np.eye(3)
        return R

    @property
    def area(self):
        """Cylinder area."""
        s = 2*np.pi*self.r**2 + 2*np.pi*self.r*self.h 
        return s

    @property
    def volume(self):
        """Cylinder volume."""
        return np.pi*self.r**2*self.h

class Contour(Geom):
    def __init__(self, soma, density) -> None:
        super().__init__()
        self.color = self.colors[1].reshape(4, 1)
        self.center = soma[0]['position'].reshape(3, 1)
        self._translation = self.center
        self.density = density
        self.points, self.normals = self._create_points(soma)
        self._geometric_measures = self.geometric_measures
        self.keep = np.full(self.points.shape[1], True)

    def intersect(self, geom, eps=1e-14):
        """Check intersection with another geometry.

        Args:
            geom (Geom): another geometry.
            eps (float, optional): margin of the boundary. Defaults to 1e-14.

        Returns:
            tuple: contains three masks
                `inner`: mask of inner points;
                `on`: mask of points on the boundary;
                `outer`: mask of outer points;
                `out_near`: mask of outer points close to the boundary.
        """
        # transform points to local coordinate
        ms = mlab.MeshSet()
        mref = mlab.Mesh(
            vertex_matrix = self.points.T,
            v_normals_matrix = self.normals.T
            )
        m = mlab.Mesh(
            vertex_matrix = geom.points.T
            )
        ms.add_mesh(mref)
        ms.add_mesh(m)
        ms.distance_from_reference_mesh(measuremesh=1, refmesh=0)
        # distance is saved in the quality array
        dist = ms.mesh(1).vertex_quality_array()
        # masks
        inner = dist < -eps
        on = (dist >= -eps) & (dist <= eps)
        outer = dist > eps
        out_near = (dist > eps) & (dist < 0.1*geom.r_min)
        return inner, on, outer, out_near

    def fix_normals(self, points, normals):
        """Flip inward normal vectors.

        Args:
            points (ndarray): coordinates of points,
                size [3 x npoint].
            normals (ndarray): normal vectors, 
                size [3 x npoint].

        Returns:
            ndarray: fixed normal vectors.
        """
        # translate points to local coordinate system
        points = points - self._translation
        # fix normals
        cos_angle = np.einsum('ij,ij->j', points, normals)
        normals[:, cos_angle<0] *= -1
        return normals

    def _create_points(self, soma):
        p = []
        parent_id = 0
        for child_id in soma[parent_id]['children_id']:
            if child_id < len(soma):
                self._add_points(p, soma, parent_id, child_id)
        p = np.hstack(p)
        # create convex hull
        ms = mlab.MeshSet()
        m = mlab.Mesh(vertex_matrix = p.T)
        ms.add_mesh(m)
        ms.convex_hull()
        out_dict = ms.compute_geometric_measures()
        ms.poisson_disk_sampling(
            samplenum = int(self.density*out_dict['surface_area'])
            )
        points = ms.current_mesh().vertex_matrix().T
        normals = ms.current_mesh().vertex_normal_matrix().T
        # move points to center
        points = points - out_dict['barycenter'].reshape(3,1) + self.center
        # flip inward-pointing normals
        normals = self.fix_normals(points, normals)
        return points, normals

    def _add_points(self, points, soma, parent_id, child_id):
        # build cylinder
        cylin1 = soma[parent_id]
        cylin2 = soma[child_id]
        cylin0 = dcp(soma[child_id])
        cylin0['position'] = (cylin1['position'] + cylin2['position'])/2
        cylin0['radius'] = np.max([cylin1['radius'], cylin2['radius']])
        temp_cylinder = Cylinder([cylin0, cylin1, cylin2], 1)
        # add points
        p, _, _ = temp_cylinder.output()
        points.append(p)
        # new node
        parent_id = child_id
        if len(soma[parent_id]['children_id']) != 0:
            for child_id in soma[parent_id]['children_id']:
                if child_id < len(soma):
                    self._add_points(points, soma, parent_id, child_id)
        else:
            # stop recursion
            return 0

    @property
    def geometric_measures(self):
        ms = mlab.MeshSet()
        m = mlab.Mesh(
            vertex_matrix = self.points.T,
            v_normals_matrix = self.normals.T
            )
        ms.add_mesh(m)
        ms.convex_hull()
        return ms.compute_geometric_measures()

    @property
    def area(self):
        return self._geometric_measures['surface_area']

    @property
    def volume(self):
        return self._geometric_measures['mesh_volume']


class Frustum(Geom):
    """Round frustum objects representing neurite segments.

    Attributes:
        ra (float): frustum bottom radius.
        rb (float): frustum top radius.
        a (ndarray): frustum bottom center, size: [3 x 1].
        b (ndarray): frustum top center, size: [3 x 1].
        points (ndarray): coordinates of sampled points,
            size: [3 x npoint].
        normals (ndarray): out-pointing normal vectors,
            size: [3 x npoint].
        keep (ndarray): the mask of points to keep.
            Inner points are removed.
        _translation (ndarray): vector translating
            the local frustum to position `a`, size: [3 x 1].
        _rotation (ndarray): matrix rotating
            the local frustum, size: [3 x 3].
    """
    def __init__(self, start, end, density) -> None:
        """Create a round frustum.

        Args:
            start (dict): start defines frustum's bottom.
                A dictionary with two keys: `position` and `radius`.
            end (dict): end defines frustum's top.
                A dictionary with two keys: `position` and `radius`.
        """
        super().__init__()
        self.color = self.colors[end['type']].reshape(4, 1)
        self.ra = start['radius']
        self.rb = end['radius']
        self.a = start['position'].reshape(3, 1)
        self.b = end['position'].reshape(3, 1)
        self._translation = self.a
        self._rotation = self.rotation_matrix
        self.density = density
        self.points, self.normals = self._create_points()
        self.keep = np.full(self.points.shape[1], True)

    def intersect(self, geom, eps=1e-14):
        """Check intersection with another geometry.

        Args:
            geom (Geom): another geometry.
            eps (float, optional): margin of the boundary. Defaults to 1e-14.

        Returns:
            tuple: contains three masks
                `inner`: mask of inner points;
                `on`: mask of points on the boundary;
                `outer`: mask of outer points;
                `out_near`: mask of outer points close to the boundary.
        """
        # transform points to local coordinate
        points = geom.points - self._translation
        points = self._rotation.T @ points

        # get r_min
        if isinstance(geom, Frustum):
            r_min = min(geom.r_min, self.r_min)
        else:
            r_min = self.r_min

        # top
        top_mask = points[2, :] >= self.h
        dist = LA.norm(points - np.array([[0,0,self.h]]).T, axis=0)
        dist = dist - self.rb
        top_in, top_on, top_out, top_near = \
            self._create_masks(top_mask, dist, eps, r_min)
        # bottom
        bottom_mask = points[2, :] <= 0
        dist = LA.norm(points, axis=0) - self.ra
        bottom_in, bottom_on, bottom_out, bottom_near = \
            self._create_masks(bottom_mask, dist, eps, r_min)
        # lateral
        lateral_mask = (points[2, :] > 0) & (points[2, :] < self.h)
        dist = LA.norm(points[:2, :], axis=0) - self._r(points[2, :])
        lateral_in, lateral_on, lateral_out, lateral_near = \
            self._create_masks(lateral_mask, dist, eps, r_min)

        # assemble masks
        inner = top_in | lateral_in | bottom_in
        on = top_on | lateral_on | bottom_on
        outer = top_out | lateral_out | bottom_out
        out_near = top_near | lateral_near | bottom_near
        return inner, on, outer, out_near

    def fix_normals(self, points, normals):
        """Flip inward normal vectors.

        Args:
            points (ndarray): coordinates of points,
                size [3 x npoint].
            normals (ndarray): normal vectors, 
                size [3 x npoint].

        Returns:
            ndarray: fixed normal vectors.
        """
        # translate points to local coordinate system
        points = self._rotation.T @ (points - self._translation)
        points[2, :] -= self.h / 2
        normals = self._rotation.T @ normals
        # fix normals
        cos_angle = np.einsum('ij,ij->j', points, normals)
        normals[:, cos_angle<0] *= -1
        # rotate normals back to global coordinate system
        return self._rotation @ normals

    def _r(self, z):
        """Lateral circle radius of the frustum.

        Args:
            z (float or ndarray): the position of the lateral circle
                in local coordiante.

        Returns:
            float or ndarray: lateral circle radius.
        """
        return self.ra + (self.rb - self.ra) * z / self.h

    def _create_points(self):
        """Create points and normals on frustum surface.

        Returns:
            tuple: contains two ndarrays
                `points`: coordinates of sampled points,
                `normals`: out-pointing normal vectors.
        """
        # create points on local frustum
        points, normals = self._create_local_frustum()
        # move the local frustum
        points = self._rotation @ points + self._translation
        normals = self._rotation @ normals
        return points, normals

    def _create_local_frustum(self):
        """Create frustum in its local coordinate system. 
            Bottom center is origin and axis is z-axis.

        Args:
            nlayer (int):
                number of layers on the lateral surface.
            ncircle (int):
                number of points on a lateral layer (circle).
            nsphere (int):
                number of points on the bottom and top semispheres.

        Returns:
            tuple: contains two ndarrays
                `points`: coordinates of sampled points,
                `normals`: out-pointing normal vectors.
        """
        # number of lateral points
        if self.r_min < 0.1:
            npoint_lateral = int(self.density*16*(10 + 10*self.h))
        elif self.r_min < 0.5:
            npoint_lateral = int(self.density*16*(10 + 5*self.h))
        elif self.r_min < 1:
            npoint_lateral = int(self.density*16*(10 + 2*self.h))
        else:
            npoint_lateral = int(self.density*16*(10 + self.h))
        npoint_lateral = np.max([npoint_lateral, 200])
        # create lateral points and normals
        points_lateral, theta = self._localfrustum(npoint_lateral)        
        normals_lateral = self._rotate_local_normal(
            theta, self.local_lateral_normal)
        # get top sphere
        nsphere = int(self.density * self.top_area)
        nsphere = np.max([nsphere, 64])
        sphere = unitsphere(2 * nsphere)
        points_top = self.rb * sphere[:, :nsphere]
        points_top[2, :] += self.h
        normals_top = sphere[:, :nsphere]
        # get bottom sphere
        nsphere = int(self.density * self.bottom_area)
        nsphere = np.max([nsphere, 64])
        sphere = unitsphere(2 * nsphere)
        points_bottom = self.ra * sphere[:, nsphere:]
        normals_bottom = sphere[:, nsphere:]
        # get top junction
        npoint_junc_top = int(self.density * 16)
        npoint_junc_top = np.max([npoint_junc_top, 16])
        normals_junc_top, theta = unitcircle(npoint_junc_top)
        points_junc_top = self.rb * normals_junc_top
        points_junc_top[2, :] += self.h
        normals_junc_top2 = self._rotate_local_normal(theta, self.local_lateral_normal)
        normals_junc_top += normals_junc_top2
        normals_junc_top = normals_junc_top / LA.norm(normals_junc_top, axis=0)
        # get bottom junction
        npoint_junc_bottom = int(self.density * 16)
        npoint_junc_bottom = np.max([npoint_junc_bottom, 16])
        normals_junc_bottom, theta = unitcircle(npoint_junc_bottom)
        points_junc_bottom = self.ra * normals_junc_bottom
        normals_junc_bottom2 = self._rotate_local_normal(theta, self.local_lateral_normal)
        normals_junc_bottom += normals_junc_bottom2
        normals_junc_bottom = normals_junc_bottom / LA.norm(normals_junc_bottom, axis=0)

        # assemble points and normals
        points = np.hstack((points_top, points_junc_top, points_lateral,
            points_junc_bottom, points_bottom))
        normals = np.hstack((normals_top, normals_junc_top, normals_lateral,
            normals_junc_bottom, normals_bottom))
        return points, normals

    def _rotate_local_normal(self, theta, normal):
        """Rotate the local normal vector `normal`
            around the frustum axis.

        Args:
            theta (ndarray): sampled angles on a lateral circle,
                size: ntheta.
            normal (ndarray): the normal vector when theta = 0.

        Returns:
            ndarray: rotated normal vectors,
                size: [3 x ntheta].
        """
        normal = normal.reshape((1,3,1))
        normals = np.repeat(normal, len(theta), axis=0)
        # create rotation matrices
        R = np.zeros((len(theta), 3, 3))
        c, s = np.cos(theta), np.sin(theta)
        R[:, 0, 0] = c
        R[:, 0, 1] = -s
        R[:, 1, 0] = s
        R[:, 1, 1] = c
        R[:, 2, 2] = 1
        return np.squeeze(R @ normals).T

    def _localfrustum(self, n):
        """Evenly distribute points on a local frustum lateral surface.

        Args:
            n (int): number of sampled points.

        Returns:
            tuple: coordinates and angles of sampled points.
        """
        # Create fibonacci lattice
        x, y = fibonacci_lattice(n)
        theta = 2 * np.pi * x

        points = np.zeros([3, n])
        z = self.h * y
        points[0, :] = np.cos(theta) * self._r(z)
        points[1, :] = np.sin(theta) * self._r(z)
        points[2, :] = z
        return points, theta

    @property
    def axis(self):
        """Frustum axis pointing from start to end."""
        ax = self.b - self.a
        return ax

    @property
    def h(self):
        """Height of the frustum."""
        return LA.norm(self.axis)

    @property
    def slant_h(self):
        """Slant height of the frustum."""
        return LA.norm([self.h, self.ra - self.rb])

    @property
    def local_lateral_normal(self):
        """A local lateral normal vector of the frustum.
            The normal vector of point (ra, 0, 0) on the local frustum.

        Returns:
            ndarray: the normal vector, 
                size: [3 x 1].
        """
        x = np.array([self.rb-self.ra, 0, self.h])
        x = x / LA.norm(x)
        y = np.array([0, 1, 0])
        return np.cross(y, x).T

    @property
    def rotation_matrix(self):
        """Create rotation matrix transforming z-axis to frustum's axis.

        Returns:
            ndarray: rotation matrix,
                size: [3 x 3].
        """
        # compute rotation matrix to transform z to axis
        z = np.array([[0, 0, 1]]).T
        ax = self.axis / LA.norm(self.axis)
        c = ax + z
        # R: matrix transforming z to axis
        if LA.norm(c) < 1e-12:
            # ax = -z
            R = np.eye(3)
            R[2, 2] = -1
        else:
            R = 2  * (c @ c.T).T / (c.T @ c) - np.eye(3)
        return R

    @property
    def r_max(self):
        return max(self.ra, self.rb)

    @property
    def r_min(self):
        return min(self.ra, self.rb)

    @property
    def lateral_area(self):
        return np.pi * self.slant_h * (self.ra + self.rb)

    @property
    def top_area(self):
        return 2 * np.pi * self.rb**2

    @property
    def bottom_area(self):
        return 2 * np.pi * self.ra**2

    @property
    def area(self):
        return self.top_area + self.lateral_area + self.bottom_area

    @property
    def lateral_volume(self):
        return np.pi*self.h*(self.ra**2 + self.rb**2 + self.ra*self.rb)/3

    @property
    def top_volume(self):
        return 2 * np.pi * self.rb**3 /3

    @property
    def bottom_volume(self):
        return 2 * np.pi * self.ra**3 /3

    @property
    def volume(self):
        return self.top_volume + self.lateral_volume + self.bottom_volume
    
    @staticmethod
    def _create_masks(mask, dist, eps, r_min):
        """Create masks for inner, on-interface and outer points."""
        inner = mask & (dist < -eps)
        on = mask & (-eps <= dist) & (dist <= eps)
        outer = mask & (eps < dist)
        out_near = mask & (dist > eps) & (dist < 0.1*r_min)
        return inner, on, outer, out_near


def fibonacci_lattice(n):
    """http://extremelearning.com.au/evenly-distributing-points-on-a-sphere/"""
    golden_ratio = (1 + 5**0.5)/2
    indices = np.arange(n)
    x, _ = np.modf(indices / golden_ratio)
    y = (indices + 0.5) / n
    return x, y

def unitsphere(n):
    """Evenly distribute points on a unit sphere surface.

    Args:
        n (int): number of sampled points.

    Returns:
        ndarray: coordinates of sampled points,
            size: [3 x n].
    """
    # Create angles
    x, y = fibonacci_lattice(n)
    theta = 2 * np.pi * x
    phi = np.arccos(1 - 2 * y)
    # Create points
    points = np.zeros([3, n])
    points[0, :] = np.cos(theta) * np.sin(phi)
    points[1, :] = np.sin(theta) * np.sin(phi)
    points[2, :] = np.cos(phi)
    return points

def unitdisk(n):
    """Evenly distribute points on a unit disk in x-y plane.

    Args:
        n (int): number of sampled points.

    Returns:
        ndarray: coordinates of sampled points,
            size: [3 x n].
    """
    # Create angles
    x, y = fibonacci_lattice(n)
    theta = 2 * np.pi * x
    r = np.sqrt(y)
    # Create points
    points = np.zeros([3, n])
    points[0, :] = np.cos(theta) * r
    points[1, :] = np.sin(theta) * r
    return points

def unitcircle(n):
    """Evenly distribute points on a unit circle in x-y plane.

    Args:
        n (int): number of sampled points.

    Returns:
        tuple: coordinates and angles of sampled points.
    """
    # Create angles
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    # Create points
    points = np.zeros([3, n])
    points[0, :] = np.cos(theta)
    points[1, :] = np.sin(theta)
    return points, theta

def ellipsoid(n, a, b, c):
    """Evenly distribute points on an ellipsoid surface.

        Ellipsoid surface is:
            x^2/a^2 + y^2/b^2 + z^2/c^2 = 1

    Args:
        n (int): number of sampled points.
        a (float): semi a-axis length.
        b (float): semi b-axis length.
        c (float): semi c-axis length.

    Returns:
        tuple:
            coordinates of sampled points and their normals.
    """
    if a!=0 and b!=0 and c!=0:
        axes = np.abs([[a, b, c]]).T
        points = unitsphere(n) * axes
        normals = points / axes**2
    else:
        raise ValueError('Invalid ellipsoid axis length.')
    return points, normals

def cylinder(n, r, h):
    """Evenly distribute points on a cylinder surface.

        Cylinder lateral surface is defined by:
            x^2/r^2 + y^2/r^2 = 1,
            z in range(-h/2, h/2).

    Args:
        n (int): number of sampled points.
        r (float): cylinder radius.
        h (float): cylinder height.

    Returns:
        tuple: coordinates of sampled points and their normals.
    """
    # number of points on a disk
    n_disk = int(n*r/(h+r)/2)
    if r!=0 and h!=0:
        r, h = np.abs(r), np.abs(h)
        # Create angles
        x, y = fibonacci_lattice(n)
        theta = 2 * np.pi * x
        z = h * (y - 0.5)
        # lateral points and normals
        normals_lateral = np.zeros([3, n])
        normals_lateral[0, :] = np.cos(theta)
        normals_lateral[1, :] = np.sin(theta)
        points_lateral = r * normals_lateral
        points_lateral[2, :] = z
        # bottom disk
        points_bottom = r * unitdisk(n_disk) - np.array([[0,0,h/2]]).T
        normals_bottom = np.zeros((3, n_disk))
        normals_bottom[2, :] = -1
        # top disk
        points_top = r * unitdisk(n_disk) + np.array([[0,0,h/2]]).T
        normals_top = np.zeros((3, n_disk))
        normals_top[2, :] = 1
        # bottom junction
        npoint_junc = int(np.max([n_disk/10, 30]))
        normals_junc_bottom, _ = unitcircle(npoint_junc)
        points_junc_bottom = r * normals_junc_bottom
        points_junc_bottom[2, :] -= h/2
        normals_junc_bottom += np.array([[0,0,-1]]).T
        normals_junc_bottom = normals_junc_bottom / LA.norm(normals_junc_bottom, axis=0)
        # top junction
        normals_junc_top, _ = unitcircle(npoint_junc)
        points_junc_top = r * normals_junc_top
        points_junc_top[2, :] += h/2
        normals_junc_top += np.array([[0,0,1]]).T
        normals_junc_top = normals_junc_top / LA.norm(normals_junc_top, axis=0)
        # assemble points and normals
        points = np.hstack((points_top, points_junc_top, points_lateral,
            points_junc_bottom, points_bottom))
        normals = np.hstack((normals_top, normals_junc_top, normals_lateral,
            normals_junc_bottom, normals_bottom))
    else:
        raise ValueError('Invalid cylinder parameters.')
    return points, normals
