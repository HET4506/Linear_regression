import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("AmesHousing.csv")
x=dataset[["Gr Liv Area","Overall Qual","Bedroom AbvGr","Year Built"]].values
y=dataset["SalePrice"].values
print(x.shape)


x_mean = x.mean(axis=0)
x_std  = x.std(axis=0)

x = (x - x_mean) / x_std


w=np.zeros(x.shape[1])
b=0
m=x.shape[0]
alpha=0.3
J_wb=[]



num_iter=500

for i in range(num_iter):
    y_hat=np.dot(x,w)+b
    error=y_hat-y

    gradient_w=(1/m)*np.dot(x.T,error)
    gradient_b=(1/m)*error.sum()

    w=w-alpha*gradient_w
    b=b-alpha*gradient_b

    J=(1/(2*m))*(np.sum(error**2))

    J_wb.append(J)

print("x_mean:", x_mean)
print("x_std :", x_std)

print("x_new (raw):", x_new)
print("x_new_norm :", x_new_norm)

print("w:", w)
print("b:", b)


plt.plot(J_wb)
plt.xlabel("Iterations")
plt.ylabel("Cost J")
plt.title("Training Loss")
plt.show()

gr_liv_area = float(input("Gr Liv Area: "))
overall_qual = float(input("Overall Qual: "))
bedroom_abvgr = float(input("Bedroom AbvGr: "))
year_built = float(input("Year Built: "))

x_new = np.array([gr_liv_area, overall_qual, bedroom_abvgr, year_built])

x_new_norm = (x_new - x_mean) / x_std

y_pred = np.dot(x_new_norm, w) + b

print(f"Predicted House Price: ₹{y_pred:,.2f}")

plt.figure()
plt.plot(dataset["Gr Liv Area"],y,color="green",linestyle="none",marker="o",alpha=0.5)
plt.title("gr liv area")
plt.show()

plt.figure()
plt.plot(dataset["Overall Qual"],y,color="red",linestyle="none",marker="o",alpha=0.5)
plt.title("Overall Qual")
plt.show()

plt.figure()
plt.plot(dataset["Bedroom AbvGr"],y,color="blue",linestyle="none",marker="o",alpha=0.5)
plt.title("Bedroom AbvGr")
plt.show()

plt.figure()
plt.plot(dataset["Year Built"],y,color="orange",linestyle="none",marker="o",alpha=0.5)
plt.title("Year Built")
plt.show()