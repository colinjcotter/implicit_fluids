import abc
from ics import get_mesh, get_initial_conditions
from firedrake.petsc import PETSc
import firedrake as fd
from irksome import Dt, MeshConstant

def both(u):
    return 2*fd.avg(u)

class BaseModel(object, metaclass=abc.ABC):
    def __init__(self, testcase):
        """
        Model base class for implicit fluids.
        """
        self.mesh = testcase.get_mesh()
        self.testcase = testcase
        self.allocate()
        self.build_eqn()
        self.set_initial_condition()
        
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

        self.V = FunctionSpace(mesh, self.family, self.degree)
        if family == 'BDM':
            self.Qdegree = degree - 1
            self.Qfamily = "DG"
        else:
            raise NotImplementedError('family '+self.family)
        self.Q = FunctionSpace(mesh, self.Qfamily, self.Qdegree)

        # Initial condition fields and coefficients
        self.u0 = Function(V)
        self.D0 = Function(Q)
        self.b = fd.Function(Q, name="Topography")


class GSWEModel(BaseSWEModel):
    """
    Implicit the shallow water model using the G formulation.
    """
    def allocate(self):
        super().allocate(self)
        W = VectorFunctionSpace(self.mesh, self.family, self.degree)
        self._U0 = Function(W)
        self.W = W

    def build_eqn(self):
        mesh = self.mesh
        W = self.W
        dU = TestFunction(W)
        du = dU[0, :]
        dG = dG[1, :]
        u = self._U0[0, :]
        G = self._U0[1, :]

        x = fd.SpatialCoordinate(mesh)
        cx, cy, cz = x
        
        # Earth parameters
        Omega = fd.Constant(testcase.Omega)
        f = 2*Omega*cz/fd.Constant(testcase.R0)
        g = fd.Constant(testcase.g)
        H = fd.Constant(testcase.H)

        b = self.b
        U0 = self._U0
        n = fd.FacetNormal(mesh)

        # D = H - div(G)
        # G_t + u*(div(G)-H) = 0
        # therefore
        # D_t = -div(G_t) = div(u*(div(G)-H)) = -div(u*D)
        
        F = Dt(G)
        D = H - div(G)
        ubar = F/D

        from firedrake import cross, inner, dx, dot, grad, \
            dS, dx, div
        
        def perp(u):
            outward_normals = fd.CellNormal(mesh)
            return cross(outward_normals, u)

        # u equation
        centred = opts.hasName("model_centred")
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
        eqn += inner(dG, Dt(G))*dx
        eqn += inner(dG, u*div(G) - H)*dx
        self._eqn = eqn

    def set_initial_conditions(self):
        # First we get b, u0 and D0, then
        # solve for G and insert into U0
        ic_opts = PETSc.GetOptions("ics_")
        get_initial_conditions(ic_opts, u=u0, D=D0, b=b)
        
def get_model(opts):
    model_type = opts.getString('type', 'swe')
    model_variant = opts.getString('variant', 'G')

    if model_type == 'swe':
        if model_variant == 'G':
            model = GSWEModel()
        else:
            raise NotImplementedError('variant ='+model_variant)
    else:
        raise NotImplementedError('type='+model_type)
    return model
