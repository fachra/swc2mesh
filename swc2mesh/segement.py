from abc import ABC, abstractmethod
import numpy as np
from numpy import linalg as LA


class Geom(ABC):
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
    def __init__(self) -> None:
        self.points = None
        self.normals = None
        self.keep = None

    def __len__(self) -> int:
        return np.count_nonzero(self.keep)

    def fast_intersect(self, geom):
        """Axis-aligned bounding box collision detection."""
        xa, ya, za = self.aabb
        xb, yb, zb = geom.aabb
        if xa['max'] <= xb['min'] or xa['min'] >= xb['max'] \
            or ya['max'] <= yb['min'] or ya['min'] >= yb['max'] \
            or za['max'] <= zb['min'] or za['min'] >= zb['max']:
            # no collision
            return False
        else:
            # collision detected
            return True
    
    def update(self, mask) -> None:
        """Update the mask `keep`.

        Args:
            mask (ndarray): mask of points.
        """
        self.keep = np.logical_and(self.keep, mask.reshape(-1))

    def output(self):
        """Output all valid points.
            Valid points do not intersect with other geometries. 

        Returns:
            tuple: valid points and their out-pointing normals.
        """
        p = self.points[:, self.keep]
        n = self.normals[:, self.keep]
        return p, n

    @property
    def aabb(self):
        """Axis-aligned bounding box"""
        p, _ = self.output()
        x = {'min': np.min(p[0,:]), 'max': np.max(p[0,:])}
        y = {'min': np.min(p[1,:]), 'max': np.max(p[1,:])}
        z = {'min': np.min(p[2,:]), 'max': np.max(p[2,:])}
        return x, y, z

    @abstractmethod
    def intersect(self):
        pass



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
    def __init__(self, soma, **kwargs) -> None:
        """Create a sphere using soma's position and radius.

        Args:
            soma (dict): A dictionary with 
                two keys: `position` and `radius`.
        """
        self.r = soma['radius']
        self.center = soma['position'].reshape(3, 1)
        if 'soma_pdensity' in kwargs:
            self.points, self.normals = \
                self._create_points(kwargs['soma_pdensity'])
        else:
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
                `outer`: mask of outer points.
        """
        # compute distance
        dist = LA.norm(geom.points - self.center, axis=0)
        dist = dist - self.r
        # check the intersection
        inner = dist < -eps
        on = (-eps <= dist) & (dist <= eps)
        outer = dist > eps
        return inner, on, outer

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

    def _create_points(self, sph_pdensity = 0.25):
        """Create points and normals on sphere surface.

        Args:
            sph_pdensity (float, optional): 
                Number of points per unit area (approximately).
                Defaults to 0.25.

        Returns:
            tuple: contains two ndarrays
                `points`: coordinates of sampled points,
                `normals`: out-pointing normal vectors.
        """
        npoint = sph_pdensity * np.max([36*4, int(self.area)])
        normals = unitsphere(int(npoint))
        return self.r * normals + self.center, normals

    @property
    def area(self):
        """Sphere area.

        Returns:
            float: area.
        """
        return 4*np.pi*self.r**2

    @property
    def volume(self):
        """Sphere volume.

        Returns:
            float: volume.
        """
        return 4*np.pi*self.r**3 / 3


class Frustum(Geom):
    """Round frustum objects representing neurite segments.

    Attributes:
        ra (float): frustum bottom radius.
        rb (float): frustum top radius.
        a (ndarray): frustum bottom center, size: [3 x 1].
        b (ndarray): frustum top center, size: [3 x 1].
        center (ndarray): soma center, size: [3 x 1].
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
    def __init__(self, start, end, **kwargs) -> None:
        """Create a round frustum.

        Args:
            start (dict): start defines frustum's bottom.
                A dictionary with two keys: `position` and `radius`.
            end (dict): end defines frustum's top.
                A dictionary with two keys: `position` and `radius`.
        """
        self.ra = start['radius']
        self.rb = end['radius']
        self.a = start['position'].reshape(3, 1)
        self.b = end['position'].reshape(3, 1)
        self._translation = self.a
        self._rotation = self.rotation_matrix
        if 'frus_pdensity' in kwargs:
            self.points, self.normals = self._create_points(kwargs['frus_pdensity'])
        else:
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
                `outer`: mask of outer points.
        """
        # transform points to local local coordinate
        points = geom.points - self._translation
        points = self._rotation.T @ points
        
        # top
        top_mask = points[2, :] >= self.len_axis
        dist = LA.norm(points - np.array([[0,0,self.len_axis]]).T, axis=0)
        dist = dist - self.rb
        top_in, top_on, top_out = \
            self._create_masks(top_mask, dist, eps)
        # bottom
        bottom_mask = points[2, :] <= 0
        dist = LA.norm(points, axis=0) - self.ra
        bottom_in, bottom_on, bottom_out = \
            self._create_masks(bottom_mask, dist, eps)
        # lateral
        lateral_mask = (points[2, :] > 0) & (points[2, :] < self.len_axis)
        dist = LA.norm(points[:2, :], axis=0) - self._r(points[2, :])
        lateral_in, lateral_on, lateral_out = \
            self._create_masks(lateral_mask, dist, eps)

        # assemble masks
        inner = top_in | lateral_in | bottom_in
        on = top_on | lateral_on | bottom_on
        outer = top_out | lateral_out | bottom_out
        return inner, on, outer

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
        points[2, :] -= self.len_axis / 2
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
        return self.ra + (self.rb - self.ra) * z / self.len_axis

    def _create_points(self, frus_pdensity = 1.0):
        """Create points and normals on frustum surface.

        Args:
            frus_pdensity (float, optional):
                Number of points per unit area (approximately).
                Defaults to 1.0.

        Returns:
            tuple: contains two ndarrays
                `points`: coordinates of sampled points,
                `normals`: out-pointing normal vectors.
        """
        # number of layers on the lateral surface
        nlayer = frus_pdensity * np.max([int(self.len_axis), 5])
        # need more layers to improve mesh quality
        nlayer *= 4
        # number of points on a lateral layer
        ncircle = frus_pdensity * np.max([int(np.pi * (self.ra + self.rb)), 16])
        # number of points on the bottom and top semispheres
        nsphere = frus_pdensity * int(np.pi * (self.ra + self.rb)**2 / 2)
        nsphere = np.max([nsphere, 64])

        # create points on local frustum
        points, normals = \
            self._create_local_frustum(int(nlayer), int(ncircle), int(nsphere))
        # move the local frustum
        points = self._rotation @ points + self._translation
        normals = self._rotation @ normals
        return points, normals

    def _create_local_frustum(self, nlayer, ncircle, nsphere):
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
        # get radii of lateral circles
        samples = np.linspace(0, self.len_axis, nlayer, endpoint=False) \
            + self.len_axis / nlayer / 2
        rs = self._r(samples)
        # get lateral points and normals
        points_lateral = np.empty((3, ncircle * nlayer))
        normals_lateral = np.empty((3, ncircle * nlayer))
        theta0 = np.linspace(0, 2*np.pi, ncircle, endpoint=False)
        for ind, ir in enumerate(rs):
            if ind % 2:
                theta = theta0 + np.pi / ncircle
            else:
                theta = theta0
            indices = np.arange(ind*ncircle, (ind + 1)*ncircle)
            points_lateral[0, indices] = ir * np.cos(theta)
            points_lateral[1, indices] = ir * np.sin(theta)
            points_lateral[2, indices] = samples[ind]
            normals_lateral[:, indices] = self._rotate_local_normal(
                theta, self.local_lateral_normal)

        # get sphere
        sphere = unitsphere(2 * nsphere)
        # get top sphere
        points_top = self.rb * sphere[:, :nsphere]
        points_top[2, :] += self.len_axis
        normals_top = sphere[:, :nsphere]
        # get bottom sphere
        points_bottom = self.ra * sphere[:, nsphere:]
        normals_bottom = sphere[:, nsphere:]
        # assemble points and normals
        points = np.hstack((points_top, points_lateral, points_bottom))
        normals = np.hstack((normals_top, normals_lateral, normals_bottom))
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

    @property
    def axis(self):
        """Frustum axis pointing from start to end.

        Returns:
            ndarray: frustum axis.
        """
        ax = self.b - self.a
        return ax

    @property
    def len_axis(self):
        """Length of frustum axis."""
        return LA.norm(self.axis)

    @property
    def local_lateral_normal(self):
        """A local lateral normal vector of the frustum.
            The normal vector of point (ra, 0, 0) on the local frustum.

        Returns:
            ndarray: the normal vector, 
                size: [3 x 1].
        """
        x = np.array([self.rb-self.ra, 0, self.len_axis])
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
    
    @staticmethod
    def _create_masks(mask, dist, eps):
        """Create masks for inner, on-interface and outer points."""
        inner = mask & (dist < -eps)
        on = mask & (-eps <= dist) & (dist <= eps)
        outer = mask & (eps < dist)
        return inner, on, outer


def unitsphere(npoint):
    """Create points evenly distributed on a unit sphere.

    Args:
        npoint (int): number of sampled points.

    Returns:
        ndarray: coordinates of sampled points,
            size: [3 x npoint].
    """
    # Get evenly distributed points on the unit sphere.
    # Create angles
    golden_ratio = (1 + 5**0.5)/2
    indices = np.arange(npoint)
    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * (indices + 0.5) / npoint)
    # Create points
    points = np.zeros([3, npoint])
    points[0, :] = np.cos(theta) * np.sin(phi)
    points[1, :] = np.sin(theta) * np.sin(phi)
    points[2, :] = np.cos(phi)
    return points
