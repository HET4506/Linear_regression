import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d as Axes3D

dataset=pd.read_csv("AmesHousing.csv")

x=dataset["Gr Liv Area"]
y=dataset["SalePrice"]
nx=len(x)
ny=len(y)

print(" length of x ",nx,"\n","length of y ",ny)

plt.figure(figsize=(8,4))
plt.plot(x,y,color="red",marker="x",linestyle="none")
plt.xlabel("Area of house")
plt.ylabel("Sale Price")
plt.title("traning set")
plt.show()

w_vals=np.linspace(-1000,1000,100)
b_vals=np.linspace(-500000,700000,100)
W,B=np.meshgrid(w_vals,b_vals)
m=nx
J_wb=np.zeros_like(W)

for i in range (W.shape[0]):
    for j in range (W.shape[1]):
        y_hat=W[i,j]*x+B[i,j]
        error=y_hat-y
        J_wb[i,j]=(1/(2*m))*(error**2).sum()


plt.ion()
fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(B,W,J_wb)
ax.set_xlabel("b")
ax.set_ylabel("w")
ax.set_zlabel("J(w,b)")
plt.show()

plt.figure(figsize=(8,6))
plt.contour(B, W, J_wb, levels=50)
plt.xlabel("b")
plt.ylabel("w")
plt.show()

alpha=5e-7
ini_w=0;ini_b=0
num_iter=5000
temp=np.zeros(num_iter)
temp_x=np.arange(num_iter)
w=ini_w;b=ini_b
for k in range (num_iter):
    y_hat=w*x+b
    error=y_hat-y
    J_wb=(1/(2*m))*(error**2).sum()
    temp[k]=J_wb
    
    gradient_w=(1/m)*(error*x).sum()
    gradient_b=(1/m)*error.sum()
    
    w=w-alpha*gradient_w
    b=b-alpha*gradient_b

plt.figure(figsize=(8,4))
plt.plot(temp_x,temp,color="orange")
plt.title("decrease in cost function")
plt.show()
print("value of w ",w,"\nvalue of b ",b)

print(f"model equation: {w:.4f} x + {b:.4f}")
plt.figure(figsize=(10,8))
plt.plot(x,y_hat,color="maroon")
plt.plot(x,y,linestyle="none",color="green",marker="x")
plt.ylabel("Sale Price")
plt.xlabel("Area")
plt.show()

a=float(input("Enter the Area of house: "))
prediction=w*a+b
print(f"According the the House Area: {a} the price should be approx {prediction:.4f}")