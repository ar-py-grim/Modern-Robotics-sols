import modern_robotics as mr
import numpy as np
import csv

def IKinBodyIterates(T_desired, theta_guess, M, Blist, eomg, ev):

    """Computes inverse kinematics in the body frame per iteration for an open chain robot

    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns

    :param M: The home configuration of the end-effector

    :param T_desired: The desired end-effector configuration Tsd

    :param theta_guess: An initial guess of joint angles that are close to
                       satisfying Tsd

    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg

    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev

    :return theta_matrix: All the Joint angles that were achieved with sucessive iterations 
                          within the specified tolerances,

    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.

    Uses an iterative Newton-Raphson root-finding method.

    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed."""


    theta_matrix = []
   
    itr, maxiterations  = 0, 25
    print(f"initial guess: {theta_guess}","\n")
    thetalist = np.array(theta_guess).copy()                # create empty array of same size 
    Tb = mr.FKinBody(M, Blist, thetalist)
    Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(mr.FKinBody(M, Blist, thetalist)), T_desired)))

    mag_omega_b = np.linalg.norm([Vb[0], Vb[1], Vb[2]])
    mag_v_b = np.linalg.norm([Vb[3], Vb[4], Vb[5]])
    err = mag_omega_b > eomg or mag_v_b > ev

    with open('MR_part2_week4/iterates.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            while err and itr < maxiterations:

                thetalist = thetalist \
                            + np.dot(np.linalg.pinv(mr.JacobianBody(Blist, thetalist)), Vb)
                Tb = mr.FKinBody(M, Blist, thetalist)                 # end-effector configuration
                Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(mr.FKinBody(M, Blist, thetalist)), T_desired)))
                mag_omega_b = np.linalg.norm([Vb[0], Vb[1], Vb[2]])   # angular error magnitude
                mag_v_b = np.linalg.norm([Vb[3], Vb[4], Vb[5]])       # linear error magnitude
                theta_matrix.append(thetalist)
                writer.writerow(thetalist)
                print(f"Iteration {itr+1}:")
                print(f"joint vector: {thetalist}")
                print(f"SE(3) end-effector config: {Tb}")
                print(f"error twist V_b: {Vb}")
                print(f"angular error magnitude ||omega_b||: {mag_omega_b}")
                print(f"linear error magnitude ||v_b||: {mag_v_b}","\n")
                err = mag_omega_b > eomg or mag_v_b > ev
                itr = itr + 1

    return (theta_matrix,not err)


def main():
    
    w1,w2,l1,l2,h1,h2 = 0.109,0.082,0.425,0.392,0.089,0.095  # all in meters (for UR5 arm)

    T_desired = np.array([[0, 1, 0, -0.5],
                [0, 0, -1, 0.1],
                [-1, 0, 0, 0.1],
                [0, 0, 0, 1]])

    ew, ev, pi = 0.001, 0.0001, np.pi

    B_list = np.array([[0, 1, 0, w1+w2, 0, l1+l2], # joint1
                       [0, 0, 1, h2, -l1-l2, 0],
                       [0, 0, 1, h2, -l2, 0],
                       [0, 0, 1, h2, 0, 0],
                       [0, -1, 0, -w2, 0, 0],
                       [0, 0, 1, 0, 0, 0]]).T      # joint6
    
    
    thetalist0 = np.array([pi, 0, pi/4, pi, 0, pi/3])

    M = np.array([[-1, 0, 0, l1+l2],
                  [0, 0, 1, w1+w2],
                  [0, 1, 0, h1-h2],
                  [0, 0, 0, 1]])

    matrix,result = IKinBodyIterates(T_desired, thetalist0, M, B_list, ew, ev)

    if result==True:

        print("Inverse Kinematics Successful!")
        print("Matrix of Joint Angles at Each Iteration:")
        for i, joint_angles in enumerate(matrix, 1):
            print(f"Iteration {i}: {joint_angles}")
        

if __name__ == "__main__":
    main()

    