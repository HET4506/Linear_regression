import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
import numpy as np

dataset=pd.read_csv("Advertising.csv")

x=dataset["TV"]
y=dataset["Sales"]

print(f"{len(x)} \n {len(y)}")

plt.figure(figsize=(10,5))
plt.plot(x,y,marker="o",color="red",linestyle="none")
plt.ylabel("Sales")
plt.xlabel("TV advertisment")
plt.title("training set")
plt.show()

m=len(x)
w_vals=np.linspace(-5000,5000,100)
b_vals=np.linspace(-300000,300000,100)

W,B=np.meshgrid(w_vals,b_vals)
J_wb=np.zeros_like(W)

for i in range (W.shape[0]):
    for j in range (W.shape[1]):

        y_hat=W[i,j]*x+B[i,j]
        error=y_hat-y

        J_wb[i,j]=(1/(2*m))*(error**2).sum()

fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(W,B,J_wb)
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_zlabel("J(w,b)")
plt.show()

plt.figure(figsize=(8,4))
plt.contour(W,B,J_wb,levels=50)
plt.xlabel("w")
plt.ylabel("b")
plt.show()

alpha=5e-5
w=0;b=0
num_iter=5000
temp=np.zeros(num_iter)
temp_x=np.arange(num_iter)
for k in range(num_iter):

    y_hat=w*x+b
    error=y_hat-y
    J_wb=(1/(2*m))*(error**2).sum()
    temp[k]=J_wb

    gradient_w=(1/m)*(error*x).sum()
    gradient_b=(1/m)*error.sum()

    w=w-alpha*gradient_w
    b=b-alpha*gradient_b

print(f"w={w}\nb={b}")
print(f"equation of line: {w:.4f} x + {b:.4f}")
plt.figure(figsize=(8,4))
plt.plot(temp_x,temp,color="orange")
plt.title("decrease in cost function")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(x,y,marker="o",color="green",linestyle="none")
plt.plot(x,w*x+b,color="maroon")
plt.xlabel("Advertisment")
plt.ylabel("Sales")
plt.show()


a=float(input("Enter number of Advertisments: "))
prediction=w*a+b
print(f"Total sales based on {a} number of ads is: ${prediction:.4f}")