import abc
import firedrake as fd
from firedrake.petsc import PETSc


class BaseTestcase:
    """
    Base class for testcases.
    """
    def __init__(self, opts):
        self.opts = opts

    @abc.abstractmethod
    def get_mesh(self):
        """
        Return a mesh for the testcase.
        """
        pass

    @abc.abstractmethod
    def set_ics(self, model):
        """
        Set fields for initial conditions and coefficients
        as required for the specific model.

        Options:
        nrefs - int, set the number of mesh refinement levels
        starpatch - if present, extends the halos to accommodate star patches.
        """


class W5Testcase(BaseTestcase):
    def __init__(self, opts):
        super().__init__(opts)
        self.R0 = 6371220.
        self.Omega = 7.292e-5
        self.g = 9.8
        self.H = 5960.

    def get_mesh(self):
        nrefs = self.opts.getInt(
            'mesh_nrefs', 5)
        starpatch = self.opts.hasName('mesh_starpatch')
        if starpatch:
            dps = {
                "partition": True,
                "overlap_type":
                (fd.DistributedMeshOverlapType.VERTEX, 2)}
        else:
            dps = None
        mesh = fd.IcosahedralSphereMesh(radius=self.R0,
                                        refinement_level=nrefs,
                                        degree=1,
                                        distribution_parameters=dps)
        x = fd.SpatialCoordinate(mesh)
        mesh.init_cell_orientations(x)
        self.mesh = mesh
        return mesh

    def set_ics(self, model):
        """
        Set initial conditions for velocity v0, layer thickness D0,
        and bathymetry b
        """

        x, y, z = fd.SpatialCoordinate(self.mesh)
        u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
        u_max = fd.Constant(u_0)
        R0 = self.R0
        Omega = self.Omega
        g = self.g
        H = self.H
        u_expr = fd.as_vector([-u_max*y/R0, u_max*x/R0, 0.0])
        eta_expr = - ((R0*Omega*u_max + u_max*u_max/2.0)*(z*z/(R0*R0)))/g
        # Topography.
        rl = fd.pi/9.0
        lambda_x = fd.atan2(y/R0, x/R0)
        lambda_c = -fd.pi/2.0
        phi_x = fd.asin(z/R0)
        phi_c = fd.pi/6.0
        minarg = fd.min_value(pow(rl, 2),
                              pow(phi_x-phi_c, 2) + pow(lambda_x-lambda_c, 2))
        bexpr = 2000.0*(1 - fd.sqrt(minarg)/rl)
        model.b.interpolate(bexpr)
        model.u0.interpolate(u_expr)
        model.D0.interpolate(eta_expr + H - bexpr)


class W6Testcase(BaseTestcase):
    def __init__(self, opts):
        super().__init__(opts)
        self.R0 = 6371220.
        self.Omega = 7.292e-5
        self.g = 9.8
        self.H = 5960.
        self.opts = opts

    def get_mesh(self):
        nrefs = self.opts.getInt(
            'mesh_nrefs', 5)
        starpatch = self.opts.hasName('mesh_starpatch')
        if starpatch:
            dps = {
                "partition": True,
                "overlap_type":
                (fd.DistributedMeshOverlapType.VERTEX, 2)}
        else:
            dps = None
        mesh = fd.IcosahedralSphereMesh(radius=self.R0,
                                        refinement_level=nrefs,
                                        degree=1,
                                        distribution_parameters=dps)
        x = fd.SpatialCoordinate(mesh)
        mesh.init_cell_orientations(x)
        self.mesh = mesh
        return mesh

    def set_ics(self, model):
        """
        Set initial conditions for velocity v0, layer thickness D0,
        and bathymetry b
        """
        x, y, z = fd.SpatialCoordinate(self.mesh)
        lon = fd.atan2(y, x)
        ll = (x**2 + y**2)**0.5
        lat = fd.atan2(z, ll)

        def perp(u):
            outward_normals = fd.CellNormal(self.mesh)
            return fd.cross(outward_normals, u)

        # code copied from Gusto
        R = fd.Constant(4)
        K = fd.Constant(7.847e-6)  # Frequency parameter, in sec^-1
        w = K
        H0 = fd.Constant(8000.)
        psi = fd.Function(model.E)
        R0 = self.R0
        Omega = self.Omega
        g = self.g
        psiexpr = -R0**2 * w * fd.sin(lat) + \
            R0**2 * K * fd.cos(lat)**R * fd.sin(lat) * fd.cos(R*lon)
        psi.interpolate(psiexpr)
        u_expr = perp(fd.grad(psi))
        model.u0.project(u_expr)
        # Initialising the depth field
        A = (w / 2) * (2 * Omega + w) * fd.cos(lat)**2 + \
            0.25 * K**2 * fd.cos(lat)**(2 * R) * \
            ((R + 1) * fd.cos(lat)**2 + (2 * R**2 - R - 2) -
             2 * R**2 * fd.cos(lat)**(-2))
        B_frac = (2 * (Omega + w) * K) / ((R + 1) * (R + 2))
        B = B_frac * fd.cos(lat)**R * ((R**2 + 2 * R + 2) -
                                       (R + 1)**2 * fd.cos(lat)**2)
        C = (1 / 4) * K**2 * fd.cos(lat)**(2 * R) * \
            ((R + 1)*fd.cos(lat)**2 - (R + 2))
        Dexpr = H0 + R0**2 * (A + B*fd.cos(lon*R) + C * fd.cos(2 * R * lon))/g
        model.D0.interpolate(Dexpr)


def get_testcase(opts):
    testcase = opts.getString('testcase', 'w6')
    testcase_opts = PETSc.Options('testcase_')
    if testcase == 'w5':
        return W5Testcase(testcase_opts)
    elif testcase == 'w6':
        return W6Testcase(testcase_opts)
    else:
        raise NotImplementedError('testcase '+testcase)
