
def main():
    import os
    import torch
    import open3d as o3d
    import numpy as np
    import time
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from solver_functions import (E_inc, Green_function, Derivative_Green_function_x1, Derivative_Green_function_x2,
                                  Derivative_Green_function_x3,G_mm, G_mm_derivative,Electric_field)

    # Set the number of decimal places to 10
    torch.set_printoptions(precision=10)
    device = "cpu"
    Lambda = torch.tensor(4).to(device)
    K_0s = torch.tensor([2 * torch.pi / Lambda]).to(device)
    for i, k_0 in enumerate(K_0s):

        meshes = np.array(["data/Sphere_comsol"])
        for ite, Mesh in enumerate(meshes):
            print(Mesh)
            # Load the STL file as a TriangleMesh
            mesh_file_path = os.path.join("..", "data", Mesh + ".stl")
            mesh = o3d.io.read_triangle_mesh(mesh_file_path)
            # Compute the normals for the mesh
            mesh.compute_vertex_normals()
            # scaling
            Scale = 1  # mm
            # Get vertices and normals as numpy arrays
            vertices = np.asarray(mesh.vertices) * Scale  # en mm
            vertices = torch.tensor(vertices).to(device, dtype=torch.complex128).to(device)
            triangles = np.asarray(mesh.triangles)
            triangles = torch.tensor(triangles, dtype=torch.complex128).to(device).real.long()

            # Compute triangle centers on GPU
            triangle_centers = (vertices[triangles[:, 0]] + vertices[triangles[:, 1]] + vertices[triangles[:, 2]]) / 3

            triangle_centers = triangle_centers.to(device)
            N = triangle_centers.shape[0]
            N = torch.tensor(N).to(device)
            print(N)

            num_triangles = triangles.shape[0]
            tangent_vectors = []
            normal_vectors = []
            areas = []

            for i in range(num_triangles):
                v0 = vertices[triangles[i, 0]]
                v1 = vertices[triangles[i, 1]]
                v2 = vertices[triangles[i, 2]]
                edge0 = v1 - v0
                edge1 = v2 - v0
                t0 = v0 - triangle_centers[i]
                area = (0.5 * torch.norm(torch.cross(edge0, edge1)))
                tangent1 = t0 / torch.norm(t0)
                edge1 = edge1 / torch.norm(edge1)
                normal = torch.cross(edge0, edge1) / (torch.norm(torch.cross(edge0, edge1)))
                tangent2 = torch.cross(normal, tangent1) / (torch.norm(torch.cross(normal, tangent1)))
                normal_vectors.append(normal)
                tangent_vectors.append(torch.stack((tangent1, tangent2)))
                areas.append(area)

            normal_vectors = torch.stack(normal_vectors)
            normal_vectors = normal_vectors.to(device)
            tangent_vectors = torch.stack(
                tangent_vectors)  # You might need to adjust this depending on the desired shape
            areas = torch.tensor(areas)
            areas = areas.to(device)

            # Initials
            d = torch.tensor([0, 0, 1], dtype=torch.complex128).to(
                device)  # polarization direction of the incident field
            k = torch.tensor([1, 0, 0], dtype=torch.complex128).to(
                device)  # propagation direction of the incident field
            nl1 = (torch.tensor([1, 0, 0], dtype=torch.complex128)).unsqueeze(0).repeat(N, 1).to(device)
            nl2 = (torch.tensor([0, 1, 0], dtype=torch.complex128)).unsqueeze(0).repeat(N, 1).to(device)
            nl3 = (torch.tensor([0, 0, 1], dtype=torch.complex128)).unsqueeze(0).repeat(N, 1).to(device)

            # Extract the two tangent vectors
            tangent_vector_1 = tangent_vectors[:, 0, :]
            tangent_vector_2 = tangent_vectors[:, 1, :]
            # Compute the E_inc values for all triangle centers
            E_inc_values = E_inc(d, k_0, k, triangle_centers).to(dtype=torch.complex128)

            # Perform batch dot product using torch.bmm (batch matrix-matrix product)
            # Before that, we need to adjust dimensions
            tangent_vector_1 = tangent_vector_1.unsqueeze(2).to(
                dtype=torch.complex128)  # add an extra dimension for bmm
            tangent_vector_2 = tangent_vector_2.unsqueeze(2).to(
                dtype=torch.complex128)  # add an extra dimension for bmm
            E_inc_values = E_inc_values.unsqueeze(1)  # adjust dimensions for bmm

            # Calculate Beta_1 and Beta_2

            Beta_1 = -torch.bmm(E_inc_values, tangent_vector_1).squeeze(2).reshape(-1, 1)
            Beta_2 = -torch.bmm(E_inc_values, tangent_vector_2).squeeze(2).reshape(-1, 1)

            # Combine as before
            Beta_3 = torch.zeros((N, 1), dtype=torch.complex128).to(device)
            Beta = torch.cat((Beta_1, Beta_2, Beta_3), dim=0).to(device)

            start_time1 = time.time()

            # Assuming triangle_centers is already a tensor of shape (N, 3)
            triangle_centers_i = triangle_centers.unsqueeze(1).expand(N, N, 3).to(device, dtype=torch.complex128)
            triangle_centers_j = triangle_centers.unsqueeze(0).expand(N, N, 3).to(device, dtype=torch.complex128)
            # Compute mask for the i != j condition
            mask = torch.eye(N, device=device) == 0  # Assuming you're using a device variable for GPU/CPU

            # counting time step

            # Calculate Green functions and derivatives
            g = torch.where(mask, Green_function(k_0, triangle_centers_i, triangle_centers_j),
                            torch.zeros((N, N), device=device)).to(dtype=torch.complex128)
            dg1 = torch.where(mask, Derivative_Green_function_x1(k_0, triangle_centers_i, triangle_centers_j),
                              torch.zeros((N, N), device=device)).to(dtype=torch.complex128)
            dg2 = torch.where(mask, Derivative_Green_function_x2(k_0, triangle_centers_i, triangle_centers_j),
                              torch.zeros((N, N), device=device)).to(dtype=torch.complex128)
            dg3 = torch.where(mask, Derivative_Green_function_x3(k_0, triangle_centers_i, triangle_centers_j),
                              torch.zeros((N, N), device=device)).to(dtype=torch.complex128)

            end_time1 = time.time()
            elapsed_time1 = end_time1 - start_time1
            print("Elapsed time1:", elapsed_time1, "seconds")
            start_time2 = time.time()

            diag_G_mm = G_mm(areas, k_0, triangle_centers, normal_vectors, N,device).to(device)
            diag_G_mm_derivative_nl1 = G_mm_derivative(nl1, areas, k_0, triangle_centers, normal_vectors, N,device).to(device)
            diag_G_mm_derivative_nl2 = G_mm_derivative(nl2, areas, k_0, triangle_centers, normal_vectors, N,device).to(device)
            diag_G_mm_derivative_nl3 = G_mm_derivative(nl3, areas, k_0, triangle_centers, normal_vectors, N,device).to(device)

            end_time2 = time.time()
            elapsed_time2 = end_time2 - start_time2
            print("Elapsed time2:", elapsed_time2, "seconds")
            start_time3 = time.time()

            end_time3 = time.time()
            elapsed_time3 = end_time3 - start_time3
            print("Elapsed time3:", elapsed_time3, "seconds")
            start_time4 = time.time()

            # Create the Z_XX matrices using broadcasting
            # Create the Z_XX matrices using broadcasting
            Z_11 = (tangent_vectors[:, 0, 0][:, None] * g).to(device)
            Z_12 = (tangent_vectors[:, 0, 1][:, None] * g).to(device)
            Z_13 = (tangent_vectors[:, 0, 2][:, None] * g).to(device)
            Z_21 = (tangent_vectors[:, 1, 0][:, None] * g).to(device)
            Z_22 = (tangent_vectors[:, 1, 1][:, None] * g).to(device)
            Z_23 = (tangent_vectors[:, 1, 2][:, None] * g).to(device)
            Z_31 = dg1.to(device)
            Z_32 = dg2.to(device)
            Z_33 = dg3.to(device)

            end_time4 = time.time()
            elapsed_time4 = end_time4 - start_time4
            print("Elapsed time4:", elapsed_time4, "seconds")
            start_time5 = time.time()
            # Set the diagonal elements using the diagonal matrices

            Z_11[torch.arange(Z_11.shape[0]), torch.arange(Z_11.shape[0])] = tangent_vectors[:, 0, 0] * diag_G_mm
            Z_12[torch.arange(Z_12.shape[0]), torch.arange(Z_12.shape[0])] = tangent_vectors[:, 0, 1] * diag_G_mm
            Z_13[torch.arange(Z_13.shape[0]), torch.arange(Z_13.shape[0])] = tangent_vectors[:, 0, 2] * diag_G_mm
            Z_21[torch.arange(Z_21.shape[0]), torch.arange(Z_21.shape[0])] = tangent_vectors[:, 1, 0] * diag_G_mm
            Z_22[torch.arange(Z_22.shape[0]), torch.arange(Z_22.shape[0])] = tangent_vectors[:, 1, 1] * diag_G_mm
            Z_23[torch.arange(Z_23.shape[0]), torch.arange(Z_23.shape[0])] = tangent_vectors[:, 1, 2] * diag_G_mm
            Z_31[torch.arange(Z_31.shape[0]), torch.arange(Z_31.shape[0])] = diag_G_mm_derivative_nl1.to(Z_31.dtype)
            Z_32[torch.arange(Z_32.shape[0]), torch.arange(Z_32.shape[0])] = diag_G_mm_derivative_nl2.to(Z_32.dtype)
            Z_33[torch.arange(Z_33.shape[0]), torch.arange(Z_33.shape[0])] = diag_G_mm_derivative_nl3.to(Z_33.dtype)

            end_time5 = time.time()
            elapsed_time5 = end_time5 - start_time5
            print("Elapsed time5:", elapsed_time5, "seconds")
            start_time6 = time.time()

            # Create the Z matrix by stacking the Z_XX matrices using PyTorch
            Z_top_row = torch.cat((Z_11, Z_12, Z_13), dim=1)
            Z_middle_row = torch.cat((Z_21, Z_22, Z_23), dim=1)
            Z_bottom_row = torch.cat((Z_31, Z_32, Z_33), dim=1)

            Z = torch.cat((Z_top_row, Z_middle_row, Z_bottom_row), dim=0).to(device)

            end_time6 = time.time()
            elapsed_time6 = end_time6 - start_time6
            print("Elapsed time6:", elapsed_time6, "seconds")

            start_time7 = time.time()
            # Convert PyTorch tensors to NumPy arrays.
            Z_np = Z.cpu().numpy()
            print(torch.linalg.eigvals(Z))
            Beta_np = Beta.cpu().numpy()
            torch.cuda.empty_cache()
            Alpha_gpu = np.linalg.solve(Z_np, Beta_np)

            end_time7 = time.time()
            elapsed_time7 = end_time7 - start_time7
            print("Elapsed time7:", elapsed_time7, "seconds")
            start_time8 = time.time()
            alpha_calc = torch.from_numpy(Alpha_gpu).to(device).squeeze(1)
            alpha1, alpha2, alpha3 = torch.chunk(alpha_calc, chunks=3, dim=0)


            ##########################################################################################
            #                                        PLOT                                            #
            ##########################################################################################

            # Plotting Es in XY plan
            # We exclude the calculation of the scattering electric field outside the domain
            N_grid = 500
            # Create a mesh grid
            x = np.linspace(-5 * Scale, 5 * Scale, N_grid)
            y = np.linspace(-5 * Scale, 5 * Scale, N_grid)
            X, Y = np.meshgrid(x, y)

            K_vector_plane_cor = 0
            K_vector_cor_val = np.ones((N_grid, N_grid)) * K_vector_plane_cor
            combined_matrix = np.dstack((K_vector_cor_val, X, Y))
            torch_tensor = torch.from_numpy(combined_matrix).to(device)
            E_x, E_y, E_z, G_f = Electric_field(torch_tensor,triangle_centers,alpha1,alpha2,alpha3,k_0,N)
            E_x = E_x.cpu().numpy()
            E_y = E_y.cpu().numpy()
            E_z = E_z.cpu().numpy()
            G_f = G_f.cpu().numpy()
            # Find the index of the highest value

            end_time8 = time.time()
            elapsed_time8 = end_time8 - start_time8
            print("Elapsed time8:", elapsed_time8, "seconds")
            # Create a figure with subplots

            Ex_real = mcolors.TwoSlopeNorm(vmin=np.real(E_x).min(), vcenter=0, vmax=np.real(E_x).max())
            Ey_real = mcolors.TwoSlopeNorm(vmin=np.real(E_y).min(), vcenter=0, vmax=np.real(E_y).max())
            Ez_real = mcolors.TwoSlopeNorm(vmin=np.real(E_z).min(), vcenter=0, vmax=np.real(E_z).max())
            Ex_imag = mcolors.TwoSlopeNorm(vmin=np.imag(E_x).min(), vcenter=0, vmax=np.imag(E_x).max())
            Ey_imag = mcolors.TwoSlopeNorm(vmin=np.imag(E_y).min(), vcenter=0, vmax=np.imag(E_y).max())
            Ez_imag = mcolors.TwoSlopeNorm(vmin=np.imag(E_z).min(), vcenter=0, vmax=np.imag(E_z).max())
            plt.figure(figsize=(24, 8))

            # Create the contour plot for the electric field magnitude
            plt.subplot(241)
            # Assuming X, Y, E_x, E_y, and E_z are already defined
            # Calculate the function
            values = np.sqrt(np.real(E_x * E_x.conj()) + np.real(E_y * E_y.conj()) + np.real(E_z * E_z.conj()))

            # Create the mask where X and Y are between -1 and 1
            mask = X ** 2 + Y ** 2 <= 1

            # Apply the mask to the values
            values[mask] = 0
            contour1 = plt.contourf(X, Y, values, levels=900, cmap='RdBu_r')
            plt.colorbar(contour1)
            plt.xlabel('Y')
            plt.ylabel('Z')
            plt.title('Total Electric Field Magnitude')
            # Generate theta values
            theta = np.linspace(0, 2 * np.pi, 100)
            # Generate circle points
            circle_x = np.cos(theta)
            circle_y = np.sin(theta)
            # Fill the circle with gray color
            plt.fill(circle_x, circle_y, color='gray', alpha=0.2)

            highest_value_of_E_norm = np.amax(np.real(values))
            lowest_value_of_E_norm = np.amin(np.real(values))
            print("Highest value of E_norm:", highest_value_of_E_norm)
            print("lowest value of E_norm:", lowest_value_of_E_norm)

            # Plot E_x

            plt.subplot(242)
            values = np.real(E_x)

            # Create the mask where X and Y are between -1 and 1
            mask = X ** 2 + Y ** 2 <=  1

            # Apply the mask to the values
            values[mask] = 0
            highest_value_of_E_x = np.amax(values)
            lowest_value_of_E_x = np.amin(values)
            print("Highest value of E_x:", highest_value_of_E_x)
            print("lowest value of E_x:", lowest_value_of_E_x)
            contour2 = plt.contourf(X, Y, values, levels=900, cmap='RdBu_r', norm=Ex_real)
            plt.colorbar(contour2, extend='both')
            plt.xlabel('Y')
            plt.ylabel('Z')
            plt.title('Real part Electric Field Component E_x')
            theta = np.linspace(0, 2 * np.pi, 100)
            # Generate circle points
            circle_x = np.cos(theta)
            circle_y = np.sin(theta)
            # Fill the circle with gray color
            plt.fill(circle_x, circle_y, color='gray', alpha=0.5)

            plt.subplot(243)
            values = np.real(E_y)

            # Create the mask where X and Y are between -1 and 1
            mask = X ** 2 + Y ** 2 <=  1

            # Apply the mask to the values
            values[mask] = 0
            highest_value_of_E_y = np.amax(values)
            lowest_value_of_E_y = np.amin(values)
            print("Highest value of E_y:", highest_value_of_E_y)
            print("lowest value of E_y:", lowest_value_of_E_y)
            contour2 = plt.contourf(X, Y, values, levels=900, cmap='RdBu_r', norm=Ey_real)
            plt.colorbar(contour2, extend='both')
            plt.xlabel('Y')
            plt.ylabel('Z')
            plt.title('Real part Electric Field Component E_y')
            theta = np.linspace(0, 2 * np.pi, 100)
            # Generate circle points
            circle_x = np.cos(theta)
            circle_y = np.sin(theta)
            # Fill the circle with gray color
            plt.fill(circle_x, circle_y, color='gray', alpha=0.5)

            plt.subplot(244)
            values = np.real(E_z)

            # Create the mask where X and Y are between -1 and 1
            mask = X ** 2 + Y ** 2 <=  1

            # Apply the mask to the values
            values[mask] = 0
            highest_value_of_E_z = np.amax(values)
            lowest_value_of_E_z = np.amin(values)
            print("Highest value of E_z:", highest_value_of_E_z)
            print("lowest value of E_z:", lowest_value_of_E_z)
            contour2 = plt.contourf(X, Y, values, levels=900, cmap='RdBu_r', norm=Ez_real)
            plt.colorbar(contour2, extend='both')
            plt.xlabel('Y')
            plt.ylabel('Z')
            plt.title('Real part Electric Field Component E_z')
            theta = np.linspace(0, 2 * np.pi, 100)
            # Generate circle points
            circle_x = np.cos(theta)
            circle_y = np.sin(theta)
            # Fill the circle with gray color
            plt.fill(circle_x, circle_y, color='gray', alpha=0.5)

            # Create the contour plot for the electric field magnitude
            plt.subplot(245)
            # Assuming X, Y, E_x, E_y, and E_z are already defined
            # Calculate the function
            values = np.sqrt(np.real(G_f * np.conj(G_f)))

            # Create the mask where X and Y are between -1 and 1
            mask = X ** 2 + Y ** 2 <=  1

            # Apply the mask to the values
            values[mask] = 0
            contour1 = plt.contourf(X, Y, values, levels=900, cmap='RdBu_r')
            plt.colorbar(contour1)
            plt.xlabel('Y')
            plt.ylabel('Z')
            plt.title('Total Green function Magnitude')
            # Overlay the masked area in gray
            # Generate theta values
            theta = np.linspace(0, 2 * np.pi, 100)
            # Generate circle points
            circle_x = np.cos(theta)
            circle_y = np.sin(theta)
            # Fill the circle with gray color
            plt.fill(circle_x, circle_y, color='gray', alpha=0.5)

            highest_value_of_E_norm = np.amax(values)
            lowest_value_of_E_norm = np.amin(values)
            print("Highest value of G_f:", highest_value_of_E_norm)
            print("lowest value of G_f:", lowest_value_of_E_norm)
            # Plot E_x

            plt.subplot(246)
            values = np.imag(E_x)
            # values = np.arctan(np.imag(E_x) / np.real(E_x))
            # Create the mask where X and Y are between -1 and 1
            mask = X ** 2 + Y ** 2 <=  1

            # Apply the mask to the values
            values[mask] = 0
            highest_value_of_E_x = np.amax(values)
            lowest_value_of_E_x = np.amin(values)
            print("Highest value of E_x:", highest_value_of_E_x)
            print("lowest value of E_x:", lowest_value_of_E_x)
            contour2 = plt.contourf(X, Y, values, levels=900, cmap='RdBu_r', norm=Ex_imag)
            plt.colorbar(contour2, extend='both')
            plt.xlabel('Y')
            plt.ylabel('Z')
            plt.title('Imaginary part Electric Field Component E_x')
            theta = np.linspace(0, 2 * np.pi, 100)
            # Generate circle points
            circle_x = np.cos(theta)
            circle_y = np.sin(theta)
            # Fill the circle with gray color
            plt.fill(circle_x, circle_y, color='gray', alpha=0.5)

            plt.subplot(247)
            values = np.imag(E_y)
            # values = np.arctan(np.imag(E_y) / np.real(E_y))
            # Create the mask where X and Y are between -1 and 1
            mask = X ** 2 + Y ** 2 <=  1

            # Apply the mask to the values
            values[mask] = 0
            highest_value_of_E_y = np.amax(values)
            lowest_value_of_E_y = np.amin(values)
            print("Highest value of E_y:", highest_value_of_E_y)
            print("lowest value of E_y:", lowest_value_of_E_y)
            contour2 = plt.contourf(X, Y, values, levels=900, cmap='RdBu_r', norm=Ey_imag)
            plt.colorbar(contour2, extend='both')
            plt.xlabel('Y')
            plt.ylabel('Z')
            plt.title('Imaginary part  Electric Field Component E_y')
            theta = np.linspace(0, 2 * np.pi, 100)
            # Generate circle points
            circle_x = np.cos(theta)
            circle_y = np.sin(theta)
            # Fill the circle with gray color
            plt.fill(circle_x, circle_y, color='gray', alpha=0.5)

            plt.subplot(248)
            values = np.imag(E_z)
            # values = np.arctan(np.imag(E_z)/np.real(E_z))
            # Create the mask where X and Y are between -1 and 1
            mask = X ** 2 + Y ** 2 <= 1

            # Apply the mask to the values
            values[mask] = 0
            highest_value_of_E_z = np.amax(values)
            lowest_value_of_E_z = np.amin(values)
            print("Highest value of E_z:", highest_value_of_E_z)
            print("lowest value of E_z:", lowest_value_of_E_z)
            contour2 = plt.contourf(X, Y, values, levels=900, cmap='RdBu_r', norm=Ez_imag)
            plt.colorbar(contour2, extend='both')
            plt.xlabel('Y')
            plt.ylabel('Z')
            plt.title('Imaginary part Electric Field Component E_z')
            theta = np.linspace(0, 2 * np.pi, 100)
            # Generate circle points
            circle_x = np.cos(theta)
            circle_y = np.sin(theta)
            # Fill the circle with gray color
            plt.fill(circle_x, circle_y, color='gray', alpha=0.5)

            # Add a title for the entire figure
            plt.suptitle('Electric Field Analysis')

            # Display the plots
            plt.tight_layout()
            manager = plt.get_current_fig_manager()
            manager.window.state('zoomed')  # For TkAgg backend
            plt.show()

if __name__ == "__main__":
    main()