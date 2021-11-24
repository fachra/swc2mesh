import numpy as np
from numpy import linalg as LA


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
    def __init__(self) -> None:
        self.points = None
        self.normals = None
        self.keep = None
        self.color = None

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
        if self.__len__() < 1:
            x = {'min': -np.inf, 'max': -np.inf}
            y = {'min': -np.inf, 'max': -np.inf}
            z = {'min': -np.inf, 'max': -np.inf}
        else:
            p, _ = self.output()
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

    def _create_points(self):
        """Create points and normals on sphere surface.

        Returns:
            tuple: contains two ndarrays
                `points`: coordinates of sampled points,
                `normals`: out-pointing normal vectors.
        """
        npoint = int(10 * self.density * self.area)
        npoint = np.max([128, npoint])
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


class Ellipsoid(Geom):
    def __init__(self) -> None:
        super().__init__()


class Cylinder(Geom):
    def __init__(self) -> None:
        super().__init__()


class Contour(Geom):
    def __init__(self) -> None:
        super().__init__()


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
    def __init__(self, start, end, density) -> None:
        """Create a round frustum.

        Args:
            start (dict): start defines frustum's bottom.
                A dictionary with two keys: `position` and `radius`.
            end (dict): end defines frustum's top.
                A dictionary with two keys: `position` and `radius`.
        """
        super().__init__()
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
                `outer`: mask of outer points.
        """
        # transform points to local local coordinate
        points = geom.points - self._translation
        points = self._rotation.T @ points
        
        # top
        top_mask = points[2, :] >= self.h
        dist = LA.norm(points - np.array([[0,0,self.h]]).T, axis=0)
        dist = dist - self.rb
        top_in, top_on, top_out = \
            self._create_masks(top_mask, dist, eps)
        # bottom
        bottom_mask = points[2, :] <= 0
        dist = LA.norm(points, axis=0) - self.ra
        bottom_in, bottom_on, bottom_out = \
            self._create_masks(bottom_mask, dist, eps)
        # lateral
        lateral_mask = (points[2, :] > 0) & (points[2, :] < self.h)
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
        # get lateral points and normals
        npoint_lateral = int(25 * self.density * self.lateral_area 
                    * np.sqrt(self.h/np.min([self.ra,self.rb])))
        with open('info_lateral.txt', 'a') as f:
            f.write(f'npoint: {npoint_lateral}\n')
            f.write(f'area: {self.area}\n')
            f.write(f'h/r: {self.h/np.min([self.ra,self.rb])}\n')
            f.write(f'r: {np.min([self.ra,self.rb])}\n')

        npoint_lateral = np.max([npoint_lateral, 256])
        npoint_lateral = np.min([npoint_lateral, 5000])
        points_lateral, theta = self._unitfrustum(npoint_lateral)        
        normals_lateral = self._rotate_local_normal(
            theta, self.local_lateral_normal)
        # get top sphere
        nsphere = int(self.density * self.top_area)
        nsphere = np.max([npoint_lateral, 64])
        sphere = unitsphere(2 * nsphere)
        points_top = self.rb * sphere[:, :nsphere]
        points_top[2, :] += self.h
        normals_top = sphere[:, :nsphere]
        # get bottom sphere
        nsphere = int(self.density * self.bottom_area)
        nsphere = np.max([npoint_lateral, 64])
        sphere = unitsphere(2 * nsphere)
        points_bottom = self.ra * sphere[:, nsphere:]
        normals_bottom = sphere[:, nsphere:]
        # get top junction
        npoint_junc_top = 10 * int(self.density * 2*np.pi * self.rb)
        npoint_junc_top = np.max([npoint_junc_top, 32])
        normals_junc_top, theta = unitcircle(npoint_junc_top)
        points_junc_top = self.rb * normals_junc_top
        points_junc_top[2, :] += self.h
        normals_junc_top2 = self._rotate_local_normal(theta, self.local_lateral_normal)
        normals_junc_top += normals_junc_top2
        normals_junc_top = normals_junc_top / LA.norm(normals_junc_top, axis=0)
        # get bottom junction
        npoint_junc_bottom = 10 * int(self.density * 2*np.pi * self.ra)
        npoint_junc_bottom = np.max([npoint_junc_bottom, 32])
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

    def _unitfrustum(self, n):
        """Evenly distribute points on a unit frustum.

        Args:
            n (int): number of sampled points.

        Returns:
            tuple: coordinates and angles of sampled points.
        """
        # Create fibonacci lattice
        x, y = fibonacci_lattice(n)
        theta = 2 * np.pi * x

        points = np.zeros([3, n])
        rmin = np.min([self.ra, self.rb])
        rmax = np.max([self.ra, self.rb])
        if (rmax - rmin) / self.slant_h > 1:
            r = lambda h: rmin + (rmax - rmin) * h / self.h
            slant = rmin * self.slant_h / (rmax - rmin)
            # Create points, distribute more points on the rmax side
            temp = np.sqrt(slant / (self.slant_h + slant))
            y = (self.slant_h * y + slant) / (self.slant_h + slant)
            z = self.h * (np.sqrt(y) - temp) / (1 - temp)
            points[0, :] = np.cos(theta) * r(z)
            points[1, :] = np.sin(theta) * r(z)
            points[2, :] = z
        else:
            # frustum is similar to a cylinder
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
    def lateral_area(self):
        return np.pi * self.slant_h * (self.ra + self.rb)

    @property
    def top_area(self):
        return np.pi * self.rb**2

    @property
    def bottom_area(self):
        return np.pi * self.ra**2

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
    def _create_masks(mask, dist, eps):
        """Create masks for inner, on-interface and outer points."""
        inner = mask & (dist < -eps)
        on = mask & (-eps <= dist) & (dist <= eps)
        outer = mask & (eps < dist)
        return inner, on, outer


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
        a (float): semi-axis length.
        b (float): semi-axis length.
        c (float): semi-axis length.

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

def cylinder(n, r, h, only_lateral=True):
    """Evenly distribute points on a cylinder surface.

        Cylinder lateral surfaceis defined by:
            x^2/r^2 + y^2/r^2 = 1,
            z in range(-h/2, h/2).

    Args:
        n (int): number of sampled points.
        r (float): cylinder radius.
        h (float): cylinder height.
        only_lateral (bool): only include lateral surface.

    Returns:
        tuple: coordinates of sampled points and their normals.
    """
    if not only_lateral:
        n_disk = int(n*r/(h+r)/2)
        n -= 2*n_disk
    if r!=0 and h!=0:
        r, h = np.abs(r), np.abs(h)
        # Create angles
        x, y = fibonacci_lattice(n)
        theta = 2 * np.pi * x
        z = h * (y - 0.5)
        # Create points and normals
        normals = np.zeros([3, n])
        normals[0, :] = np.cos(theta)
        normals[1, :] = np.sin(theta)
        normals[2, :] = 0
        points = r * normals
        points[2, :] = z
        if not only_lateral:
            points_bottom = unitdisk(n_disk) - np.array([[0,0,h/2]]).T
            normals_bottom = np.zeors((3, n_disk))
            normals_bottom[2, :] = -1
            points_top = unitdisk(n_disk) + np.array([[0,0,h/2]]).T
            normals_top = np.zeors((3, n_disk))
            normals_top[2, :] = 1
    else:
        raise ValueError('Invalid cylinder parameters.')
    return points, normals
