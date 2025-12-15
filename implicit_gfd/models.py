import abc
from firedrake.petsc import PETSc
import firedrake as fd
from irksome import Dt, MeshConstant
from math import fabs
from testcases import get_testcase

def both(u):
    return 2*fd.avg(u)

class BaseModel:
    def __init__(self, testcase, opts):
        """
        Model base class for implicit fluids.
        """
        self.mesh = testcase.get_mesh()
        self.testcase = testcase
        self.opts = opts
        self.allocate()
        self.build_eqn()
        self.set_initial_conditions()

    @abc.abstractmethod
    def allocate(self):
        """
        Allocate U0 and any Function coefficients
        required in the equation system.
        """
        pass

    @abc.abstractmethod
    def set_initial_conditions(self):
        """
        Set initial conditions and any coefficients.
        """
        pass

    @abc.abstractmethod
    def build_eqn(self):
        """
        Construct the equation system and boundary conditions.
        """
        pass

    @property
    @abc.abstractmethod
    def U0(self):
        """
        Return the Function describing the model state,
        updated by the timestepper.
        """
        pass

    @property
    @abc.abstractmethod
    def eqn(self):
        """
        Return the UFL form describing the model equation system,
        written in terms of U0, and the boundary conditions.
        """
        pass

class BaseSWEModel(BaseModel):
    """
    Base class for shallow water models.
    """
    
    def allocate(self):
        mesh = self.mesh
        opts = PETSc.Options()
        self.family = opts.getString('functionspaces_family', 'BDM')
        self.degree = opts.getInt('functionspaces_degree', 2)

        self.V = fd.FunctionSpace(mesh, self.family, self.degree)
        if self.family == 'BDM':
            self.Edegree = self.degree + 1
            self.Qdegree = self.degree - 1
            self.Qfamily = "DG"
            self.Efamily = "CG"
        else:
            raise NotImplementedError('family '+self.family)
        self.Q = fd.FunctionSpace(mesh, self.Qfamily, self.Qdegree)
        self.E = fd.FunctionSpace(mesh, self.Efamily, self.Edegree) 
        # Initial condition fields and coefficients
        self.u0 = fd.Function(self.V)
        self.D0 = fd.Function(self.Q)
        self.b = fd.Function(self.Q, name="Topography")


class GSWEModel(BaseSWEModel):
    """
    Implicit the shallow water model using the G formulation.
    """
    def allocate(self):
        super().allocate()
        W = fd.VectorFunctionSpace(self.mesh, self.family, self.degree, dim=2)
        self._U0 = fd.Function(W)
        self.W = W

    def U0(self):
        return self._U0

    def eqn(self):
        return self._eqn
        
    def build_eqn(self):
        mesh = self.mesh
        W = self.W
        dU = fd.TestFunction(W)
        du = dU[0, :]
        dG = dU[1, :]
        u = self._U0[0, :]
        G = self._U0[1, :]

        x = fd.SpatialCoordinate(mesh)
        cx, cy, cz = x
        
        # Earth parameters
        testcase = self.testcase
        Omega = fd.Constant(testcase.Omega)
        f = 2*Omega*cz/fd.Constant(testcase.R0)
        g = fd.Constant(testcase.g)
        H = fd.Constant(testcase.H)
        self.H = H
        b = self.b
        U0 = self._U0
        n = fd.FacetNormal(mesh)

        # D = H - div(G)
        # G_t + u*(div(G)-H) = 0
        # therefore
        # D_t = -div(G_t) = div(u*(div(G)-H)) = -div(u*D)
        # ie D_t + div(u*D) = 0
        F = Dt(G)
        D = H - fd.div(G)
        ubar = F/D

        from firedrake import cross, inner, dx, dot, grad, \
            dS, dx, div, sign
        
        def perp(u):
            outward_normals = fd.CellNormal(mesh)
            return cross(outward_normals, u)

        # u equation
        centred = self.opts.hasName("model_centred")
        if centred:
            Upwind = 0.5
        else:
            Upwind = 0.5 * (sign(dot(u, n)) + 1)
        eqn = inner(du, Dt(u))*dx
        eqn -= inner(perp(grad(inner(du, perp(ubar)))), u)*dx
        eqn += inner(both(perp(n)*inner(du, perp(ubar))), both(Upwind*u))*dS
        eqn += inner(du, f*perp(ubar))*dx
        eqn -= div(du)*(inner(u,u)/2 + g*(D+b))*dx

        # G equation
        # G_t + u*(div(G)-H) = 0
        eqn += fd.inner(dG, Dt(G))*fd.dx
        eqn += fd.inner(dG, u*(div(G) - H))*fd.dx
        self._eqn = eqn

    def set_initial_conditions(self):
        self.testcase.set_ics(self)
        # Elliptic problem to get G st D = H - div(G)
        WG = self.V * self.Q
        One = fd.Function(self.Q).assign(1.0)
        H = fd.assemble(self.D0*fd.dx)/fd.assemble(One*fd.dx)
        self.H.assign(H)
        assert fabs(fd.assemble((H-self.D0)*fd.dx))/fd.assemble(One*fd.dx) < 1.0e-7
        H = self.H
        uG, phi = fd.TrialFunctions(WG)
        v, q = fd.TestFunctions(WG)
        eqn = (fd.inner(uG, v) - fd.div(v)*phi
               + q*(fd.div(uG) + (self.D0 - H)))*fd.dx
        shift_J = fd.lhs(eqn + fd.Constant(1.0e-8)*q*phi*fd.dx)
        params = {
            'ksp_type': 'gmres',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
        }
        v_basis = fd.VectorSpaceBasis(constant=True, comm=fd.COMM_WORLD)
        nullspace = fd.MixedVectorSpaceBasis(WG, [WG.sub(0), v_basis])
        UG = fd.Function(WG)
        # Need to think about boundary conditions later
        hybridparams = { "mat_type": "matfree",
                         "ksp_type": "gmres",
                         "ksp_atol": 0,
                         "ksp_rtol": 1.0e-15, "ksp_monitor": None,
                         "pc_type": "python", 'pc_python_type':
                         'firedrake.HybridizationPC', #
                         'hybridization': {
                             'ksp_type': 'preonly', #
                             "ksp_error_if_not_converged": None,
                             'pc_type': 'lu',
                             'pc_factor_mat_solver_type':'mumps'}
                         }
        fd.solve(fd.lhs(eqn) == fd.rhs(eqn), UG,
                 solver_parameters = hybridparams,
                 nullspace=nullspace)
        uG, p = UG.subfunctions
        self._U0.interpolate(fd.as_tensor([self.u0, uG]))
        G = self._U0[1,:]
        res = fd.norm(fd.div(G) + (- H + self.D0))/fd.norm(One)
        assert res < 1.0e-8, res


class LocalEnergySWEModel(BaseSWEModel):
    """
    Implicit the shallow water model using the G formulation.
    """
    def allocate(self):
        super().allocate()
        # Mixed function space for model state
        # u, D, dH/du=F, G, lambda
        u_elt = fd.BrokenElement(fd.FiniteElement(self.family, fd.triangle,
                                                  self.degree))
        Vu = fd.FunctionSpace(self.mesh, u_elt)
        Vlambda = fd.FunctionSpace(self.mesh, "HDiv Trace", self.degree)
        W = Vu * self.Q * Vu * Vu * Vlambda
        self._U0 = fd.Function(W)
        self.W = W

    def build_eqn(self):
        mesh = self.mesh
        W = self.W
        du, dD, dF, dG, dll = fd.TestFunctions(W)
        u, D, F, G, ll = fd.split(self._U0)

        # Earth parameters
        testcase = self.testcase
        x = fd.SpatialCoordinate(mesh)
        cx, cy, cz = x
        Omega = fd.Constant(testcase.Omega)
        f = 2*Omega*cz/fd.Constant(testcase.R0)
        g = fd.Constant(testcase.g)
        H = fd.Constant(testcase.H)
        self.H = H
        b = self.b
        
        n = fd.FacetNormal(mesh)
        from firedrake import cross, inner, dx, dot, grad, \
            dS, dx, div, sign
        def perp(u):
            outward_normals = fd.CellNormal(mesh)
            return cross(outward_normals, u)        
        # flux equation using the "Golo trick"
        eqn = inner(Dt(G), D*dG)*dx - jump(Dt(G), n)*ll('+')*dS
        # dH/du equation
        eqn += inner(Dt(F) - u*D, dF)*dx
        ubar = Dt(F)/D
        # velocity equation
        centred = self.opts.hasName("model_centred")
        if centred:
            Upwind = 0.5
        else:
            Upwind = 0.5 * (sign(dot(u, n)) + 1)
        eqn = inner(du, Dt(u))*dx
        eqn -= inner(perp(grad(inner(du, perp(ubar)))), u)*dx
        eqn += inner(both(perp(n)*inner(du, perp(ubar))), both(Upwind*u))*dS
        eqn += inner(du, f*perp(ubar))*dx
        eqn -= div(du)*(inner(u,u)/2 + g*(D+b))*dx
        eqn += inner(du, Dt(G))*dx
        # Depth equation
        eqn += (Dt(D) + div(Dt(F)))*dx
        # lambda equation
        eqn += jump(u, n)*dll('+')*dS

def get_model(opts):
    testcase = get_testcase(opts)
    model_type = opts.getString('type', 'swe')
    model_variant = opts.getString('variant', 'G')

    if model_type == 'swe':
        if model_variant == 'G':
            model = GSWEModel(testcase, opts)
        else:
            raise NotImplementedError('variant ='+model_variant)
    else:
        raise NotImplementedError('type='+model_type)
    return model
