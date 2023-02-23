import sys
import time
import numpy as np
import sympy as sp
import matplotlib
import matplotlib.pyplot as plt
import scipy.integrate as spi
from scipy.integrate import odeint
from scipy.misc import derivative

start = time.time()

def timer():
  print(str(time.time()-start))

def space():
  print("")

def wait():
  time.sleep(0.5)

def waitspace():
  wait()
  space()

f = lambda x: x/(np.cos(x) + x**2 + 1)

print("hello!")
waitspace()
print("welcome to noah's advanced calculator")
waitspace()

# What kind of problem will be solved
print("what kind of problem do you want to solve? (1)derivative, (2)integration, (3)first order ODE, or (4)second order ODE")
space()

prob = input()

# start of DERIVATIVE
if prob == "1":
  waitspace()
  print("good choice")
  waitspace()

  # eqn or number/plot result
  print("would you like an (1)numerical or (2)plot/approximate solution?")
  space()
  dmethod = input()

  # numerical result (der.)
  if dmethod == "1":

    #x = sp.Symbol("x")
    x = float(input("what x value? "))
    N = 1000
    h = 1/N
    temp = (f(x+h/2)/h - f(x-h/2)/h)
    print(str(temp))


    # DERIVATIVE EQUATION - analytical
    #print(str(sp.diff(f,x)))

  # plot result (der.)
  elif dmethod == "2":

    # slope calculation
    xlist = []
    ylist = []
    xi = float(input("initial x value: "))
    xf = float(input("final x value: "))
    N = int(input("step size: "))
    h = (xf - xi)/N

    #x list making
    for i in range (0,N): 
      xlist.append(xi + i*h)
    
    # making y list
    for i in range (0, len(xlist)):
      temp = (f(xlist[i]+h/2)/h - f(xlist[i] - h/2)/h)
      ylist.append(temp)


    # der. plot
    y = np.linspace(-10,10)
    plt.plot(xlist,ylist)
    plt.show()
    sys.exit()
    

# start of INTEGRATION
elif prob == "2":
  waitspace()
  print("good choice")
  space()

# TRAP vs BOX method input
  print("would you like to solve via (1)trapezoid or (2)box method?")
  space()
  intmethod = input()
  
  # TRAPEZOID METHOD
  if intmethod == "1":
    space()

    # interval and step size
    a = int(input("a interval bound: "))
    b = int(input("b interval bound: "))
    N = int(input("step size: "))

    # x and y values for trapezoid method
    x = np.linspace(a,b,N+1)
    y = f(x)

    # x and y values to plot function
    X = np.linspace(a,b,100)
    Y = f(X)
    plt.plot(X,Y)

    # implementing step size into plot
    for i in range(N):
      xs = [x[i],x[i],x[i+1],x[i+1]]
      ys = [0,f(x[i]),f(x[i+1]),0]
      plt.fill(xs,ys,'b',edgecolor='b',alpha=0.2)

    # area of traps
    trap_area = x = np.linspace(a,b,N+1)
    y = f(x)
    right_y = y[1:]
    left_y = y[:-1] 
    dx = (b - a)/N
    AR = (dx/2) * np.sum(right_y + left_y)
    print("Summation of the area of trapezoids", AR)

    # final trap plot
    plt.fill(xs,ys,'b',edgecolor='b',alpha=0.2)
    plt.title('Trapezoid Method, Step Size = {}'.format(N))
    plt.show()
    sys.exit()
    

  # BOX METHOD
  elif intmethod == "2":
    print("box method")


# start of FIRST ORDER ODE
elif prob == "3":
  waitspace()
  print("good choice")
  waitspace()
  print("which method would you like to use? (1)euler's method or (2)runge kutta?")
  space()
  Fmethod = input()

  #EULER METHOD 1st order ODE
  if Fmethod == "1":
   # F = lambda x,y: (x*y)
    
    x_0 = int(input("x initial: "))
    x_f = int(input("x final: "))
    y_0 = int(input("y initial: "))
    N = int(input("step size: "))
    h = (y_0-x_0)/N

    xlist = [x_0]
    ylist = [y_0]

    for i in range(0, N):
        xlist.append(x_0 + h*f(xt,yt))

    # y values
    for i in range(0,len(xlist)-1):
        yt = ylist[-1]
        xt = xlist[i]
        ylist.append(y_0 + i * h)

    plt.plot(xlist,ylist)
    plt.show()
    sys.exit()
  
  # RUNGE KUTTA 1st order ODE
  elif Fmethod == "2":
   # f = lambda x, y: (x * y)**0.5

  # initial conditions
    x_0 = int(input("x initial: "))
    x_f = int(input("x final: "))
    y_0 = int(input("y initial: "))
    N = int(input("step size: "))

    h = (x_f - x_0) / N

    # x values
    xlist = [x_0]

    for i in range(1, N + 1):
        xlist.append(x_0 + i * h)

    # y values
    ylist = [y_0]

    # k values
    for i in range(1,len(xlist)):
        yt = ylist[-1]
        xt = xlist[i]

        k1 = h * f(xt,yt)
        k2 = h * f(xt + h/2, yt + k1/2)
        k3 = h * f(xt + h/2, yt + k2/2)
        k4 = h * f(xt + h, yt +k3)

        k = (k1 + 2*k2 + 2*k3 + k4)/6
        ylist.append(yt + k)

    # plotting
    plt.plot(xlist, ylist)
    plt.show()
    sys.exit()

# start of SECOND ORDER ODE
elif prob == "4":
  waitspace()
  print("good choice")
  waitspace()

  # standard linear 2nd order eqn
  def dP_dx(P,x):
    return [P[1], 2*P[0] - P[1] + x + np.sin(2*x)]
  
  # initial vector condition
  P0 = [1,0]

  # limit of integration
  a = int(input("lower boundary: "))
  b = int(input("upper boundary: "))

  # x and y values used
  xs = np.arange(a, b, 0.01)
  Ps = odeint(dP_dx, P0, xs)
  ys = Ps[:, 1]

  fig =plt.subplots(figsize=(10,10))
  plt.plot(xs,ys)
  plt.xlabel("x")
  plt.ylabel("f(x)")
  plt.show()
  waitspace()
  sys.exit()


else:
  print("that is not a valid answer...")
  waitspace()
  sys.exit() 