import csv
import numpy as np
import matplotlib.pyplot as plt

def write_elastic_energy_csv(data, filename="elastic_energy_total.csv", fill_value=None):
    """
    Writes a list of nested dictionaries to a CSV file with elastic energy data.

    Parameters:
    - data (list of dicts): List where each dict maps energy types to edge data.
    - filename (str): The CSV filename.
    - fill_value (float or str): Value to fill missing entries (default: None).
    """
    # Collect all unique energy types and edge IDs
    energy_types = set()
    edge_ids = set()

    for entry in data:
        for energy_type, edges in entry.items():
            energy_types.add(energy_type)
            edge_ids.update(edges.keys())

    energy_types = sorted(energy_types)  # Sort for consistent column order
    edge_ids = sorted(edge_ids)

    print("Energy Types:", energy_types)
    print("Edge IDs:", edge_ids)

    # Define CSV headers and add two extra columns for totals
    headers = ["Index", "EdgeID"]
    for et in energy_types:
        headers.extend([f"{et}Strain", f"{et}Energy"])
    headers.extend(["TotalHingeEnergy", "TotalStretchEnergy"])

    # Write to CSV
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for idx, entry in enumerate(data):
            # Compute total energies for this entry
            total_hinge = sum(
                energy for (strain, energy) in entry.get("Hinge", {}).values()
                if isinstance(energy, (int, float))
            )
            total_stretch = sum(
                energy for (strain, energy) in entry.get("Stretch", {}).values()
                if isinstance(energy, (int, float))
            )
            for edge_id in edge_ids:
                row = [idx, edge_id]
                for et in energy_types:
                    # Look up the (strain, energy) for the current edge and energy type.
                    strain, energy = entry.get(et, {}).get(edge_id, (fill_value, fill_value))
                    row.extend([strain, energy])
                # Append the total energies to the row.
                row.extend([total_hinge, total_stretch])
                writer.writerow(row)
                
def get_entry(state, springs, energy, nodes_to_edge_id):
    ret = {}
    strain = energy.get_strain(state)
    energy_arr = energy.get_energy_linear_elastic(state, False)
    for i, spring in enumerate(springs):
        # Both hinge and stretch first 2 nodes are around main edge
        edge_id = nodes_to_edge_id(spring.nodes_ind[:2])
        ret[int(edge_id)] = (strain[i], energy_arr[i][0])
    return ret

def get_node_to_edge_id(robot):
    def helper(edge):
        a, b = edge
        match = np.where((robot._SoftRobot__edges == [a, b]).all(
            axis=1) | (robot._SoftRobot__edges == [b, a]).all(axis=1))[0]
        return match[0] if match.size > 0 else None
    return helper

def plot_edge_ids(coordinates, edges):
    # 2D Plot with connectivity
    plt.figure(figsize=(8,6))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', marker='o')

    # Label each node with its ID (using zero-based indexing)
    for i, (x, y, _) in enumerate(coordinates):
        plt.text(x, y, f' {i}', color='red', fontsize=18, fontweight='bold')

    # Draw each edge as a line between nodes (using x and y coordinates)
    for edge_id, (n1, n2) in enumerate(edges):
        x_coords = [coordinates[n1, 0], coordinates[n2, 0]]
        y_coords = [coordinates[n1, 1], coordinates[n2, 1]]
        plt.plot(x_coords, y_coords, 'k-', linewidth=1)
        
        # Compute the midpoint for labeling the edge
        mid_x = (x_coords[0] + x_coords[1]) / 2
        mid_y = (y_coords[0] + y_coords[1]) / 2
        plt.text(mid_x, mid_y, f' {edge_id}', color='magenta', fontsize=18)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Plot of Coordinates and Connectivity with Labels')
    plt.grid(True)
    plt.show()

    # 3D Plot with connectivity
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
            c='blue', marker='o')

    # Label each node in 3D with its ID
    for i, (x, y, z) in enumerate(coordinates):
        ax.text(x, y, z, f' {i}', color='red', fontsize=15, fontweight='bold')

    # Draw each edge in 3D (using x, y, and z coordinates)
    for edge_id, (n1, n2) in enumerate(edges):
        x_coords = [coordinates[n1, 0], coordinates[n2, 0]]
        y_coords = [coordinates[n1, 1], coordinates[n2, 1]]
        z_coords = [coordinates[n1, 2], coordinates[n2, 2]]
        ax.plot(x_coords, y_coords, z_coords, 'k-', linewidth=1)
        
        # Compute the midpoint for labeling the edge in 3D
        mid_x = (x_coords[0] + x_coords[1]) / 2
        mid_y = (y_coords[0] + y_coords[1]) / 2
        mid_z = (z_coords[0] + z_coords[1]) / 2
        ax.text(mid_x, mid_y, mid_z, f' {edge_id}', color='magenta', fontsize=15)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of Coordinates and Connectivity with Labels')
    plt.show()