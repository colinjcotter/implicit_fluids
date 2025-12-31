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

    @abc.abstractmethod
    def output(self):
        """
        Do any necessary postprocessing and
        return a tuple of fields to output.
        """
        pass

    @abc.abstractmethod
    def diagnostics(self):
        """
        Compute diagnostics and return in a dictionary.
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
        self.u0 = fd.Function(self.V, name="Velocity")
        self.D0 = fd.Function(self.Q, name="Layer Depth")
        self.b = fd.Function(self.Q, name="Topography")

        vorticity_diagnostic = opts.hasName("diagnostics_vorticity")
        if vorticity_diagnostic:
            vort = fd.TrialFunction(self.E)
            dvort = fd.TestFunction(self.E)
            self.vorticity = fd.Function(self.E, name="Relative Vorticity")
            vort_lhs = vort*dvort*fd.dx
            vort_rhs = -fd.inner(perp(fd.grad(dvort), self.u0))*fd.dx
            vort_prob = fd.LinearVariationalProblem(vort_lhs,
                                                    vort_rhs,
                                                    self.vorticity)
            vortparams = {'ksp_type':'preonly',
                       'pc_type':'lu',
                       "pc_factor_mat_solver_type": "mumps"}
            self.vort_solver = fd.LinearVariationalSolver(vort_prob,
                                                          solver_parameters=
                                                          vortparams)
        self.vorticity_diagnostic = vorticity_diagnostic

    def output(self):
        fields = [self.u0, self.D0]
        if self.vorticity_diagnostic:
            self.vort_solver.solve()
            fields.append(self.vorticity)
        return fields
    
    def diagnostics(self):
        self.output()
        u = self.u0
        D = self.D0
        b = self.b
        g = fd.Constant(self.testcase.g)
        energy = fd.assemble((fd.inner(u, u)*D + g*D*(D/2 + b))*fd.dx)
        diagnostics = {
            "energy": energy
        }
        

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
        D = H - fd.div(G)
        if self.opts.hasName("projection"):
            F = Dt(G)
            ubar = F/D
        else:
            ubar = u

        from firedrake import cross, inner, dx, dot, grad, \
            dS, dx, div, sign
        
        def perp(u):
            outward_normals = fd.CellNormal(mesh)
            return cross(outward_normals, u)

        # u equation
        centred = self.opts.hasName("centred")
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

    def output(self):
        G = self._U0[1,:]
        u = self._U0[0,:]
        self.D0.interpolate(self.H - fd.div(G))
        self.u0.interpolate(u)
        return super().output()

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
