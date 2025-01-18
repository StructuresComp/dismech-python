import dataclasses
import os

import numpy as np


@dataclasses.dataclass
class GeomParams:
    rod_r0: float
    shell_h: float

    axs: float = None
    jxs: float = None
    ixs1: float = None
    ixs2: float = None


class Geometry:
    """
    Generate, save, and load custom node geometries
    """

    def __init__(self, rod_nodes, shell_nodes, rod_edges, rod_shell_joint_edges, face_nodes):
        # TODO: translate rest of createGeometry.m
        self.__nodes = np.concat((rod_nodes, shell_nodes))

    @staticmethod
    def from_txt(fname: str) -> "Geometry":
        """
        Reads from a .txt file and returns a Geometry object. Uses the same convention as the Matlab version.
        """
        if not os.path.exists(fname) or not os.path.isfile(fname):
            raise ValueError('{} is not a valid path'.format(fname))

        # Constants
        valid_headers = {'*rodnodes': 0, '*shellnodes': 1,
                         '*rodedges': 2, 'rodshelljointedges': 3, '*facenodes': 4}
        h_len = [3, 3, 2, 2, 2, 3]  # expected entry length

        # Flags
        h_flag = [False for _ in range(len(valid_headers))]  # list of flag
        cur_h = -1  # tracks current header

        params = [np.empty(0, dtype=np.float64)
                  for _ in range(len(valid_headers))]  # initial arrays
        temp_array = []  # temp linked list

        with open(fname, 'r') as f:
            while (line := f.readline()) != '':
                line = line[:-1]  # trim /n

                if line[0] == "*":
                    # * denotes header
                    if (h_id := valid_headers.get(line.lower())) is None:
                        raise ValueError('unknown header: {}'.format(line))

                    # check and mark header
                    if h_flag[h_id]:
                        raise ValueError('{} header used twice'.format(line))
                    h_flag[h_id] = True

                    # add previous parameter
                    if len(temp_array) > 0:
                        params[cur_h] = np.array(temp_array)
                        temp_array = []

                    cur_h = h_id
                elif line[0] == "#":
                    # '#' is a comment
                    continue
                else:

                    if len(vals := line.split(',')) != h_len[cur_h]:
                        raise ValueError(
                            "{} should have {} values".format(vals, h_len[cur_h]))

                    temp_array += [float(val) for val in vals]

        # add last parameter
        if len(temp_array) > 0:
            params[cur_h] = np.array(temp_array)
            # No need to reset temp_array

        return Geometry(*params)
