import numpy as np
import modern_robotics as mr
import csv

def simulate(thetalist, duration, integration_per_sec, filename):
    n_joints = 6
    dt = 1/integration_per_sec  # Time step for integration
    
    # UR5 robot parameters
    M01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]])
    M12 = np.array([[0, 0, 1, 0.28], [0, 1, 0, 0.13585], [-1, 0, 0, 0], [0, 0, 0, 1]])
    M23 = np.array([[1, 0, 0, 0], [0, 1, 0, -0.1197], [0, 0, 1, 0.395], [0, 0, 0, 1]])
    M34 = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.14225], [0, 0, 0, 1]])
    M45 = np.array([[1, 0, 0, 0], [0, 1, 0, 0.093], [0, 0, 1, 0], [0, 0, 0, 1]])
    M56 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.09465], [0, 0, 0, 1]])
    M67 = np.array([[1, 0, 0, 0], [0, 0, 1, 0.0823], [0, -1, 0, 0], [0, 0, 0, 1]])
    Mlist = [M01, M12, M23, M34, M45, M56, M67]
    G1 = np.diag([0.010267495893, 0.010267495893,  0.00666, 3.7, 3.7, 3.7])
    G2 = np.diag([0.22689067591, 0.22689067591, 0.0151074, 8.393, 8.393, 8.393])
    G3 = np.diag([0.049443313556, 0.049443313556, 0.004095, 2.275, 2.275, 2.275])
    G4 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
    G5 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
    G6 = np.diag([0.0171364731454, 0.0171364731454, 0.033822, 0.1879, 0.1879, 0.1879])
    Glist = [G1, G2, G3, G4, G5, G6]
    Slist = np.array([[0,         0,         0,         0,        0,        0],
                      [0,         1,         1,         1,        0,        1],
                      [1,         0,         0,         0,       -1,        0],
                      [0, -0.089159, -0.089159, -0.089159, -0.10915, 0.005491],
                      [0,         0,         0,         0,  0.81725,        0],
                      [0,         0,     0.425,   0.81725,        0,  0.81725]])
    
    g = np.array([0, 0, -9.81])

    # Initial conditions
    taulist = np.zeros(n_joints)
    Ftip = np.zeros(n_joints)
    dthetalist = np.zeros(n_joints)

    steps = int(duration*integration_per_sec)
    theta = []

    for _ in range(steps):

        theta.append(list(thetalist))
        ddthetalist = mr.ForwardDynamics(thetalist,dthetalist,taulist,g,Ftip,Mlist,Glist,Slist)

        # Updating to get new states
        thetalist,dthetalist = mr.EulerStep(thetalist,dthetalist,ddthetalist,dt)
        

    # Save to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(theta)


simulate([0, 0, 0, 0, 0, 0], 3, 100, 'MR_part3\peer_graded_assignment_week2\simulation1.csv')
simulate([0, -1, 0, 0, 0, 0], 5, 100, 'MR_part3\peer_graded_assignment_week2\simulation2.csv')