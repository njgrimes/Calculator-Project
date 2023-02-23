#TRAPEZOID METHOD
    x = np.linspace(-0.5,1.5,100)
    y = np.exp(-x**2)
    plt.plot(x,y)
    x0 = int(input("x initial: "))
    x1 = int(input("x final: "))

    #integration equation being solved
    y0 = np.exp(-x0**2); y1 = np.exp(-x1**2);
    A = 0.5*(y1 + y0)*(x1 - x0)
    print("area: ", A)

    #plot developed
    plt.fill_between([x0,x1],[y0,y1])
    plt.xlim([-0.5,1.5]); plt.ylim([0,1.5]);
    plt.show()
