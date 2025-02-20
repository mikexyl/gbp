import numpy as np

def read_g2ofile(fileName):
    measurements = []
    poses_ID1s = []
    poses_ID2s = []
    infos = []
    initials = {}

    with open(fileName, 'r') as file:
        for line in file:
            # Split line into tag and data
            parts = line.strip().split()
            if parts[0] == "VERTEX_SE3:QUAT":
                # Parse ID and initial estimate
                id = int(parts[1])
                x, y, z = map(float, parts[2:5])
                qx, qy, qz, qw = map(float, parts[5:])
                initials[id] = (x, y, z, qw, qx, qy, qz)
            elif parts[0] in ["EDGE3", "EDGE_SE3:QUAT"]:

                # Parse IDs
                id1, id2 = int(parts[1]), int(parts[2])
                poses_ID1s.append(id1)
                poses_ID2s.append(id2)

                # Parse measurement (x, y, z, quaternion)
                x, y, z = map(float, parts[3:6])
                qx, qy, qz, qw = map(float, parts[6:10])
                measurements.append((x, y, z, qw, qx, qy, qz))  # Quaternion in (w, x, y, z) order

                # Parse information matrix
                m = np.zeros((6, 6))
                upper_triangle = map(float, parts[10:])
                indices = np.triu_indices(6)
                for (i, j), value in zip(zip(*indices), upper_triangle):
                    m[i, j] = value
                    m[j, i] = value  # Symmetric matrix

                # Transform g2o information matrix to GTSAM format
                info = np.zeros((6, 6))
                info[:3, :3] = m[3:6, 3:6]  # Info rotation
                info[3:6, 3:6] = m[:3, :3]  # Info translation
                info[3:6, :3] = m[:3, 3:6]  # Off-diagonal translation-rotation
                info[:3, 3:6] = m[3:6, :3]  # Off-diagonal rotation-translation
                infos.append(info)
            
            else:
                continue

    n_points = len(set(poses_ID1s + poses_ID2s))
    n_edges = len(measurements)

    return n_points, n_edges, initials, measurements, poses_ID1s, poses_ID2s, infos