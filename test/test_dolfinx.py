import pytest
import numpy as np
import dolfinx
import dolfinx.fem
import dolfinx.mesh
import fecr
import ufl
from mpi4py import MPI

def test_dolfinx_function_conversion():
    """Test conversion between DOLFINx functions and NumPy arrays."""
    # Create a mesh
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    
    # Create a function space
    V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
    
    # Create a function
    u = dolfinx.fem.Function(V)
    
    # Set some values using expression
    expr = dolfinx.fem.Expression(ufl.sin(ufl.pi*ufl.SpatialCoordinate(mesh)[0]), V.element.interpolation_points())
    u.interpolate(expr)
    
    # Convert to NumPy
    u_np = fecr.to_numpy(u)
    
    # Convert back to DOLFINx
    u_back = fecr.from_numpy(u_np, u)
    
    # Check if values are preserved
    u_back_np = fecr.to_numpy(u_back)
    assert np.allclose(u_np, u_back_np)

def test_dolfinx_constant_conversion():
    """Test conversion between DOLFINx constants and NumPy arrays."""
    # Create constants
    scalar_const = dolfinx.fem.Constant(1.0)
    vector_const = dolfinx.fem.Constant([1.0, 2.0, 3.0])
    
    # Convert to NumPy
    scalar_np = fecr.to_numpy(scalar_const)
    vector_np = fecr.to_numpy(vector_const)
    
    # Check values
    assert scalar_np == 1.0
    assert np.array_equal(vector_np, np.array([1.0, 2.0, 3.0]))
    
    # Convert back to DOLFINx
    scalar_back = fecr.from_numpy(scalar_np, scalar_const)
    vector_back = fecr.from_numpy(vector_np, vector_const)
    
    # Check if values are preserved
    scalar_back_np = fecr.to_numpy(scalar_back)
    vector_back_np = fecr.to_numpy(vector_back)
    
    assert scalar_back_np == scalar_np
    assert np.array_equal(vector_back_np, vector_np)

def test_dolfinx_primal_evaluation():
    """Test primal function evaluation with DOLFINx."""
    # Create mesh and function space
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
    
    # Function to compute L2 norm squared
    def l2_norm_squared(u):
        return dolfinx.fem.assemble_scalar(dolfinx.fem.form(u*u*ufl.dx))
    
    # Create a template function
    u_template = dolfinx.fem.Function(V)
    
    # Generate random values
    u_values = np.random.rand(V.dofmap.index_map.size_global)
    
    # Evaluate primal
    result_np, result_dolfinx, inputs_dolfinx, tape = fecr.evaluate_primal(
        l2_norm_squared,
        [u_template],
        u_values
    )
    
    # Check if the result is a scalar
    assert isinstance(result_np, float) or (isinstance(result_np, np.ndarray) and result_np.size == 1)

if __name__ == "__main__":
    test_dolfinx_function_conversion()
    test_dolfinx_constant_conversion()
    test_dolfinx_primal_evaluation() 