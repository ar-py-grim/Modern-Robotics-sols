from collections import namedtuple
from typing import List
import numpy as np
from scipy.optimize import linprog


StaticMass = namedtuple("StaticMass", ["m", "x_pos", "y_pos"])
Contact = namedtuple("Contact", ["body_a", "body_b", "x_contact", "y_contact", "normal_angle", "friction_coefficient"])

# Examples from book

# collapses
m_0 =[StaticMass(2, 25, 35),
      StaticMass(5, 66, 42)]
contacts_0 = [Contact(1, 0, 0, 0, np.pi / 2, 0.1),
              Contact(1, 2, 60, 60, np.pi, 0.5),
              Contact(2, 0, 72, 0, np.pi / 2, 0.5),
              Contact(2, 0, 60, 0, np.pi / 2, 0.5)]

# stable
m_1 = [StaticMass(2, 25, 35),
       StaticMass(10, 66, 42)]
contacts_1 = [Contact(1, 0, 0, 0, np.pi / 2, 0.5),
              Contact(1, 2, 60, 60, np.pi, 0.5),
              Contact(2, 0, 72, 0, np.pi / 2, 0.5),
              Contact(2, 0, 60, 0, np.pi / 2, 0.5)]


# Other Examples

# _________________
# \    | *(1,5)  /
#  \   |       / (4,5)
#   \  |     /-------------/
#    \ |   /  \           /
#     \| /     \   *     /
#      v        \ (10,2)/
# --------------------------
#     (0,0)    (5,0)   (12,0)

# collapses
m_2 = [StaticMass(5, 1, 5),
       StaticMass(4, 10, 2)]
contacts_2 = [Contact(1, 0, 0, 0, np.pi/2, 0.3),
              Contact(1, 2, 4, 5, np.pi/4, 0.4),
              Contact(2, 0, 5, 0, np.pi/2, 0.4),
              Contact(2, 0, 12, 0, np.pi/2, 0.4)]

# stable
m_3 = [StaticMass(2, 1, 5),
       StaticMass(4, 10, 2)]
contacts_3 = [Contact(1, 0, 0, 0, np.pi/2, 0.7),
              Contact(1, 2, 4, 5, np.pi/4, 0.5),
              Contact(2, 0, 5, 0, np.pi/2, 0.4),
              Contact(2, 0, 12, 0, np.pi/2, 0.4)]


def find_contacts(contacts: List[Contact], body: int):
    """Find all contact of a given body"""
    return [c for c in contacts if c.body_a == body or c.body_b == body]


def stability_check(m, contacts):
    """Check the stability of given body masses and
       the contacts."""
    g = 9.81
    num_bodies = len(m)
    for i in range(num_bodies):
        # find all contacts of the current boy
        contact_bodies = find_contacts(contacts, i + 1)

        # friction cone has two edges
        num_k = len(contact_bodies) * 2

        # linear programming
        f = np.ones(num_k)
        A = -np.identity(num_k)
        b = -np.ones(num_k)
        F = []

        # calculate F matrix
        for contact_b in contact_bodies:
            x = contact_b.x_contact
            y = contact_b.y_contact

            # normal angle
            ang = contact_b.normal_angle
            if i + 1 == contact_b.body_b:
                # adjust angle if it's other body
                ang = ang - np.pi

            friction_coeff = contact_b.friction_coefficient

            # the angle
            theta = np.arctan2(friction_coeff, 1)

            # the columns of the F matrix handled as rows here and transposed later
            f1 = [np.sin(ang+theta) * x - np.cos(ang+theta) * y,
                  np.cos(ang+theta),
                  np.sin(ang+theta)]
            f2 = [np.sin(ang-theta) * x - np.cos(ang-theta) * y,
                  np.cos(ang-theta),
                  np.sin(ang-theta)]

            # add the columns
            F.extend([f1, f2])

        # convert to numpy array (transpose as columns were added as rows)
        F = np.array([np.array(xi) for xi in F]).T

        Aeq = F
        # calculate the mass
        static_mass = m[i]
        beq = [static_mass.m * static_mass.x_pos * g, 0, static_mass.m*g]
        # check fol stability with linear programming
        k = linprog(f, A, b, Aeq, beq, method='highs-ipm')
        return k


def check_stability_with_output(m, contacts):
    """Check stability of a given assembly and print results"""
    print("Checking stability with:")
    print("Masses:")
    for ms in m:
        print(ms)
    print("Contacts:")
    for c in contacts:
        print(c)

    k = stability_check(m, contacts)
    result = "The assembly remains standing" if k.success else "would collapse"
    print(result)
    print()


if __name__ == "__main__":
    """Run examples"""
    print("= Examples from book =\n")
    check_stability_with_output(m_0, contacts_0)
    check_stability_with_output(m_1, contacts_1)
    print("\n= Additional examples =\n")
    check_stability_with_output(m_2, contacts_2)
    check_stability_with_output(m_3, contacts_3)


# Tests

def test_find_contact_bodies():
    res = find_contacts(contacts_0, 1)
    assert res == [Contact(1, 0, 0, 0, np.pi / 2, 0.1),
                   Contact(1, 2, 60, 60, np.pi, 0.5)]

    res = find_contacts(contacts_0, 2)
    assert res == [Contact(1, 2, 60, 60, np.pi, 0.5),
                   Contact(2, 0, 72, 0, np.pi / 2, 0.5),
                   Contact(2, 0, 60, 0, np.pi / 2, 0.5)]


def test_stability_check_collapsing_0():
    k = stability_check(m_0, contacts_0)
    assert not k.success


def test_stability_check_stable_0():
    k = stability_check(m_1, contacts_1)
    assert k.success


def test_stability_check_collapsing_1():
    k = stability_check(m_2, contacts_2)
    assert not k.success


def test_stability_check_stable_1():
    k = stability_check(m_3, contacts_3)
    assert k.success
