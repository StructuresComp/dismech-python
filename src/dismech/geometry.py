import dataclasses
import itertools
import os
import typing

import numpy as np


GEOMETRY_FLOAT = np.float64
GEOMETRY_INT = np.int64


class Geometry:
    """
    Generate, save, and load custom node geometries
    """

    def __init__(self, rod_nodes: np.ndarray, shell_nodes: np.ndarray,
                 rod_edges: np.ndarray, rod_shell_joint_edges: np.ndarray,
                 face_nodes: np.ndarray):
        """
        acts as createGeometry.m
        """
        # save important params
        self.__rod_nodes = rod_nodes
        self.__shell_nodes = shell_nodes
        self.__rod_edges = rod_edges
        self.__rod_shell_joint_edges = rod_shell_joint_edges
        self.__face_nodes = face_nodes

        # general counting
        n_rod_nodes = np.size(rod_nodes, 0)
        n_shell_nodes = np.size(shell_nodes, 0)
        n_nodes = n_rod_nodes + n_shell_nodes
        n_rod_edges = np.size(rod_edges, 0)
        n_rod_shell_joints = np.size(rod_shell_joint_edges, 0)
        n_faces = np.size(face_nodes, 0)

        # using custom concat to deal with empty array
        self.__nodes = self.__safe_concat((rod_nodes, shell_nodes))

        # Initialize shell-related arrays
        nEdges = 3 * n_faces
        shell_edges = np.zeros((nEdges, 2), dtype=GEOMETRY_INT)
        hinges = np.zeros((nEdges, 4), dtype=GEOMETRY_INT)
        third_node = np.zeros(nEdges, dtype=GEOMETRY_INT)

        edge_faces = np.zeros((nEdges, 2), dtype=GEOMETRY_INT)
        face_shell_edges = np.zeros((n_faces, 3), dtype=GEOMETRY_INT)
        As = np.zeros(n_faces)

        self.__sign_faces = np.zeros((n_faces, 3), dtype=GEOMETRY_INT)
        self.__face_unit_norms = np.zeros((3, n_faces))

        # track # of shell/hinge edges
        s_i = 0
        h_i = 0

        # Calculate shell and hinge edges
        for i in range(n_faces):
            n1, n2, n3 = face_nodes[i, :]
            p1 = self.__nodes[n1, :]
            p2 = self.__nodes[n2, :]
            p3 = self.__nodes[n3, :]

            # face normal and area
            face_cross = np.cross(p2-p1, p3-p1)
            face_norm = np.linalg.norm(face_cross)
            As[i] = face_norm / 2
            self.__face_unit_norms[:, i] = face_cross * 1 / face_norm

            # edge between first 2 nodes
            permutations = [(n2, n3, n1), (n3, n1, n2), (n1, n2, n3)]

            # iterate over each edge pair
            for j, (n1, n2, n3) in enumerate(permutations):
                edge = (n1, n2)
                edge_neg = (n2, n1)

                # free standing edge (using walrus/short circuiting/slicing to optimize search)
                if not np.any(exist_arr := np.all(shell_edges[:s_i] == edge_neg, axis=1)) and \
                        not np.any(exist_arr := np.all(shell_edges[:s_i] == edge, axis=1)):
                    shell_edges[s_i, :] = edge
                    third_node[s_i] = n3
                    face_shell_edges[i, j] = s_i
                    self.__sign_faces[i, j] = 1
                    edge_faces[s_i, :] = [i, i]
                    s_i += 1
                # hinge with a prexisting edge
                else:
                    # [0][0] as where returns tuple of indices
                    exist_id = np.where(exist_arr)[0][0]
                    existing_edge = shell_edges[exist_id, :]
                    exist_n3 = third_node[exist_id]
                    hinges[h_i, :] = [n1, n2, exist_n3, n3]
                    face_shell_edges[i, j] = exist_id

                    # sign depends on what direction of existing edge
                    if np.array_equiv(existing_edge, edge):
                        self.__sign_faces[i, j] = 1
                    else:
                        self.__sign_faces[i, j] = -1

                    edge_faces[exist_id, 1] = i

                    h_i += 1

        # Trim unused values
        shell_edges = shell_edges[:s_i, :]
        self.__hinges = hinges[:h_i, :]

        # Ghost edges for rod shell joint bent-twist springs
        ghost_rod_shell_joint_edges = [np.array([0, 0])]  # jugaad
        for i in range(n_rod_shell_joints):
            s_node = rod_shell_joint_edges[i][1]
            s_faces = []

            # find faces with s_node
            for j in range(n_faces):
                if s_node in face_nodes[j, :]:
                    s_faces.append(j)

            # add any edges that are not already considered
            s_edges = []
            for j in range(len(s_faces)):
                temp_edges = face_shell_edges[s_faces[j], :]
                for k in range(3):
                    if not np.any(np.all(ghost_rod_shell_joint_edges == shell_edges[temp_edges[k], :], axis=0)):
                        if temp_edges[k] not in s_edges:
                            s_edges.append(temp_edges[k])

            ghost_rod_shell_joint_edges += list(shell_edges[s_edges])

        # remove jugaad
        self.__rod_shell_joint_edges_total = np.concat(
            (rod_shell_joint_edges, ghost_rod_shell_joint_edges[1:]), 0)

        # bend-twist springs
        if rod_edges.size or rod_shell_joint_edges.size:
            bend_twist_springs = []  # N e N e N
            bend_twist_signs = []
            rod_edges_modified = self.__safe_concat(
                (rod_edges, self.__rod_shell_joint_edges_total))

            for i in range(n_nodes):
                # find edges that point in/out of the center node
                into = np.where(rod_edges_modified[:, 1] == i)[0]
                outof = np.where(rod_edges_modified[:, 0] == i)[0]

                # bend springs are created between two edges that are centered
                # add as a tuple of (edge_1, edge_2), (sign1, sign2), (n1, n3) and edges to choose
                pairs = []

                # 1. all combinations of two edges that point into the node
                if into.size >= 2:
                    pairs.append(
                        ((0, 0), (1, -1), np.array(list(itertools.combinations(into, 2)))))

                # 2. all combinations of two edges that point out of the node
                if outof.size >= 2:
                    pairs.append(
                        ((1, 1), (-1, 1), np.array(list(itertools.combinations(outof, 2)))))

                # 3. all combinations of an edge into/out of the node
                if outof.size and into.size:
                    grid_into, grid_outof = np.meshgrid(
                        into, outof, indexing='ij')
                    pairs.append(
                        # transpose and ravel to match order of Matlab
                        ((0, 1), (1, 1), np.column_stack((grid_into.T.ravel(), grid_outof.T.ravel()))))

                # by pairing the combinations with indices, we can use one loop
                for ((n1, n2), (s1, s2), spring_edges) in pairs:
                    spring_nodes = np.stack((rod_edges_modified[spring_edges[:, 0], n1], i * np.ones(
                        len(spring_edges)), rod_edges_modified[spring_edges[:, 1], n2]), axis=-1)
                    bend_twist_springs.append(np.stack(
                        (spring_nodes[:, 0], spring_edges[:, 0], spring_nodes[:, 1], spring_edges[:, 1], spring_nodes[:, 2]), axis=-1))
                    bend_twist_signs.append(np.stack(
                        (s1 * np.ones(len(spring_edges)), s2 * np.ones(len(spring_edges))), axis=-1))

            self.__bend_twist_springs = np.concat(bend_twist_springs)
            self.__bend_twist_signs = np.concat(bend_twist_signs)
        else:
            self.__bend_twist_springs = np.empty(0)
            self.__bend_twist_signs = np.empty(0)

        # sequence edges
        self.__edges = self.__safe_concat(
            (rod_edges, self.__rod_shell_joint_edges_total))

        # only add unique shell_edges
        if self.__edges.size:
            for i in range(len(shell_edges)):
                if not np.any(np.all(self.__edges == shell_edges[i, :], axis=1)):
                    self.__edges = np.concat(
                        (self.__edges, shell_edges[i, :].reshape(-1, 2)))
        else:
            self.__edges = shell_edges

        self.__shell_edges = self.__edges[n_rod_edges + n_rod_shell_joints:]

        # stretch springs
        self.__rod_stretch_springs = self.__safe_concat(
            (rod_edges, rod_shell_joint_edges))
        self.__shell_stretch_springs = self.__shell_edges

        # face edges
        self.__face_edges = np.zeros((n_faces, 3))
        for i in range(n_faces):
            n1, n2, n3 = face_nodes[i]

            # edge between first 2 nodes
            permutations = [(n2, n3), (n3, n1), (n1, n2)]

            # pick pos/neg edge depending on sign_faces
            for j, (n1, n2) in enumerate(permutations):
                self.__face_edges[i][j] = np.where((self.edges == [n1, n2]).all(axis=1))[0][0] \
                    if self.__sign_faces[i][j] > 0 else np.where((self.edges == [n2, n1]).all(axis=1))[0][0]

        # Twist angles
        # TODO: Make this mutable
        self.__twist_angles = np.zeros(
            n_rod_edges + np.size(self.shell_edges, 0))

    @staticmethod
    def __safe_concat(arrs: typing.Tuple[np.ndarray]) -> np.ndarray:
        """
        np.concat does not function properly with empty arrays
        """
        # TODO: make it general
        arr1, arr2 = arrs
        if arr1.size and arr2.size:
            return np.concat((arr1, arr2), 0)
        elif arr1.size:
            return arr1
        elif arr2.size:
            return arr2
        else:
            return np.empty(0)

    @staticmethod
    def from_txt(fname: str) -> "Geometry":
        """
        Reads from a .txt file and returns a Geometry object. Uses the same convention as the Matlab version.
        """
        if not os.path.exists(fname) or not os.path.isfile(fname):
            raise ValueError('{} is not a valid path'.format(fname))

        # Constants
        valid_headers = {'*rodnodes': 0, '*shellnodes': 1,
                         '*rodedges': 2, '*rodshelljointedges': 3, '*facenodes': 4}
        h_len = [3, 3, 2, 2, 3]  # expected entry length
        h_dtype = [GEOMETRY_FLOAT, GEOMETRY_FLOAT,
                   GEOMETRY_INT, GEOMETRY_INT, GEOMETRY_INT]

        # Flags
        h_flag = [False for _ in range(len(valid_headers))]  # list of flag
        cur_h = -1  # tracks current header

        params = [np.empty(0, dtype=np.float64)
                  for _ in range(len(valid_headers))]  # initial arrays
        temp_array = []  # temp linked list

        with open(fname, 'r') as f:
            while (line := f.readline()) != '':
                line = line[:-1]  # trim /n

                # * denotes header
                if line[0] == "*":
                    if (h_id := valid_headers.get(line.lower())) is None:
                        raise ValueError('unknown header: {}'.format(line))

                    # check and mark header
                    if h_flag[h_id]:
                        raise ValueError('{} header used twice'.format(line))
                    h_flag[h_id] = True

                    # add previous parameter
                    if len(temp_array) > 0:
                        params[cur_h] = np.array(
                            temp_array, dtype=h_dtype[cur_h])

                        # Python is 0 indexed
                        if h_dtype[cur_h] == GEOMETRY_INT:
                            params[cur_h] -= 1

                        temp_array = []

                    cur_h = h_id
                # '#' is a comment
                elif line[0] == "#":
                    continue
                else:
                    if len(vals := line.split(',')) != h_len[cur_h]:
                        raise ValueError(
                            "{} should have {} values".format(vals, h_len[cur_h]))

                    temp_array.append([float(val) for val in vals])

        # add last parameter
        if len(temp_array) > 0:
            params[cur_h] = np.array(temp_array, dtype=h_dtype[cur_h])

            # Python is 0 indexed
            if h_dtype[cur_h] == GEOMETRY_INT:
                params[cur_h] -= 1

        return Geometry(*params)

    # Read-only properties (direct analogs to createGeometry.m output)

    @property
    def nodes(self):
        return self.__nodes

    @property
    def edges(self):
        return self.__edges

    @property
    def rod_nodes(self):
        return self.__rod_nodes

    @property
    def shell_nodes(self):
        return self.__shell_nodes

    @property
    def rod_edges(self):
        return self.__rod_edges

    @property
    def shell_edges(self):
        return self.__shell_edges

    @property
    def rod_shell_joint_edges(self):
        return self.__rod_shell_joint_edges

    @property
    def rod_shell_joint_edges_total(self):
        return self.__rod_shell_joint_edges_total

    @property
    def face_nodes(self):
        return self.__face_nodes

    @property
    def face_edges(self):
        return self.__face_edges

    @property
    def rod_stretch_springs(self):
        return self.__rod_stretch_springs

    @property
    def shell_stretch_springs(self):
        return self.__shell_stretch_springs

    @property
    def bend_twist_springs(self):
        return self.__bend_twist_springs

    @property
    def bend_twist_signs(self):
        return self.__bend_twist_signs

    @property
    def hinges(self):
        return self.__hinges

    @property
    def sign_faces(self):
        return self.__sign_faces

    @property
    def face_unit_norms(self):
        return self.__face_unit_norms

    @property
    def twist_angles(self):
        return self.__twist_angles
