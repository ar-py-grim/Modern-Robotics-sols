import modern_robotics as mr
import numpy as np

print(mr.QuinticTimeScaling(Tf=5,t=3))
X_sart = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
X_end = np.array([[0,0,1,1],[1,0,0,2],[0,1,0,3],[0,0,0,1]])
print(f"for method=3: {mr.ScrewTrajectory(Xstart=X_sart,Xend=X_end,Tf=10,N=10,method=3)}")
print(f"for method=5: {mr.CartesianTrajectory(Xstart=X_sart,Xend=X_end,Tf=10,N=10,method=5)}")