import abc
from ics import set_initial_conditions, get_mesh
from firedrake.petsc import PETSc
import firedrake as fd
from irksome import Dt, MeshConstant

def both(u):
    return 2*fd.avg(u)

class BaseModel(object, metaclass=abc.ABC):
    def __init__(self):
        """
        Model base class for implicit fluids.
        """
        self.mesh = get_mesh()
        self.allocate()
        self.build_eqn()
        set_initial_condition(self)
        
    @abc.abstractmethod
    def allocate(self):
        """
        Allocate U0 and any Function coefficients
        required in the equation system.
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


class GSWEModel(BaseModel):
    """
    Implicit the shallow water model using the G formulation.
    """
    super().__init__()

    def allocate(self):
        mesh = self.mesh
        opts = PETSc.Options()
        family = opts.getString('functionspaces_family', 'BDM')
        degree = opts.getInt('functionspaces_degree', 2)

        V = FunctionSpace(mesh, family, degree)
        if family == 'BDM':
            Qdegree = degree - 1
            Qfamily = "DG"
        else:
            raise NotImplementedError, 'family', family
        Q = FunctionSpace(mesh, Qfamily, Qdegree)
        W = V * V #  u, G
        self._U0 = Function(W)
        self.W = W
        self.b = fd.Function(Q, name="Topography")

    def build_eqn(self):
        mesh = self.mesh
        W = self.W
        du, dG = TestFunctions(W)
        u, G = fd.split(self._U0)

        x = fd.SpatialCoordinate(mesh)
        cx, cy, cz = x
        
        # Earth parameters
        opts = PETSc.Options()
        R0 = opts.GetScalar("model_R0", 6371220.)
        Omega = opts.GetScalar("model_Omega", 7.292e-5)
        Omega = fd.Constant(Omega)
        f = 2*Omega*cz/fd.Constant(R0)
        g = opts.GetScalar("model_g", 9.8)
        g = fd.Constant(g)
        H = opts.GetScalar("model_H", 5960.)
        H = fd.Constant(H)

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
        upwind = opts.hasName("model_upwind")
        if upwind:
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

def get_model(opts):
    model_type = opts.getString('type', 'swe')
    model_variant = opts.getString('variant', 'G')

    if model_type == 'swe':
        if model_variant == 'G':
            model = GSWEModel()
        else:
            raise NotImplementedError, 'variant ='+model_variant
    else:
        raise NotImplementedError, 'type='+model_type
    return model
