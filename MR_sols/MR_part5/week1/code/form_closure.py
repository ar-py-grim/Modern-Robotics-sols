from collections import namedtuple
import numpy as np
from scipy.optimize import linprog

# Represents a contact point with its normal
PointNormal = namedtuple("PointNormal", ["point", "normal"])


def _calculate_F_matrix(point_normals):
    """Calculate the wrench matrix by combining wrench vectors for each contact
       point columns
    """
    ret = [_calculate_single_F(pn.point, pn.normal) for pn in point_normals]
    return np.array(ret).T


def _calculate_single_F(pt, normal):
    """Calculate a wrench for a contact point"""
    ang = np.array([np.cos(normal),
                    np.sin(normal)])
    M = np.cross(pt, ang)
    return np.r_[M, ang]


def _form_closure_impl(F):
    """Calculate if the given contact point give form closure"""
    print(f"\nCheck form closure with:\n{F}\n")
    f = np.ones(4)
    A = -np.identity(4)
    b = -np.ones(4)
    Aeq = F
    beq = [0, 0, 0]
    k = linprog(f, A, b, Aeq, beq,  method='highs-ipm')
    return k


def _create_normals_array(normals):
    """Create an array of point normals by giving the x- and y-coordinates
       and the contact normal angle."""
    return [PointNormal(np.array([line[0], line[1]]),
                        line[2])
            for line in normals]


def form_closure(normals):
    """Calculate if the contacts lead to form closure"""
    F = _calculate_F_matrix(_create_normals_array(normals))
    return _form_closure_impl(F).success


def main():
    # case 1
    print("\nCase 1")
    normals = [[0, 0, np.pi],
             [0, 0, np.pi * 3 / 2],
             [2, 1, 0],
             [2, 1, np.pi / 2]]
    res = form_closure(normals)
    print(f"Form closure: {res}")

    # case 2
    print("\n\nCase 2")
    normals = [[0, 0, np.pi],
               [0, 0, np.pi * 3 / 2],
               [2, 1, 0],
               [2, 0, np.pi * 3 / 2]]
    res = form_closure(normals)
    print(f"Form closure: {res}")


if __name__ == "__main__":
    main()


# Tests

def test_form_closure_ok():
    F = np.array([[0, 0, -1, 2],
                  [-1, 0, 1, 0],
                  [0, -1, 0, 1]])

    assert _form_closure_impl(F).success


def test_form_closure_not_ok():
    F = np.array([[0, 0, 0, -2],
                  [-1, 0, 1, 0],
                  [0, -1, 0, -1]])
    assert not _form_closure_impl(F).success


def test_calculate_single_F():
    pt_1 = np.array([0, 0])
    norm_1 = np.pi
    res = _calculate_single_F(pt_1, norm_1)
    np.testing.assert_array_almost_equal(res, [0, -1, 0])


def test_calculate_F_matrix_0():
    contact_normals = [
        PointNormal(np.array([0, 0]), np.pi),
        PointNormal(np.array([0, 0]), np.pi * 3 / 2),
        PointNormal(np.array([2, 1]), 0),
        PointNormal(np.array([2, 1]), np.pi / 2),
    ]

    ret = _calculate_F_matrix(contact_normals)
    np.testing.assert_array_almost_equal(ret, [
     [0, 0, -1, 2],
     [-1, 0, 1, 0],
     [0, -1, 0, 1]])


def test_calculate_F_matrix_1():
    contact_normals = [
        PointNormal(np.array([0, 0]), np.pi),
        PointNormal(np.array([0, 0]), np.pi * 3 / 2),
        PointNormal(np.array([2, 0]), 0),
        PointNormal(np.array([2, 0]), np.pi * 3 / 2),
    ]

    ret = _calculate_F_matrix(contact_normals)
    np.testing.assert_array_almost_equal(ret, [
     [0, 0, 0, -2],
     [-1, 0, 1, 0],
     [0, -1, 0, -1]])