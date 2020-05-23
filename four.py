## Mahdi cheikhrouhou
""" Bibliothèques nécessaires """
import matplotlib.pyplot as plt
from IPython.display import Image, display
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib
import seaborn as sns
import numpy as np
import scipy 
import imageio
import cv2

class resout_tridiag:
    def __init__(self):
        pass
    
    def get_a(self,A,i):
      #fonction pour le bloc A_i
      return A[i*Nx:(i+1)*Nx,(i-1)*Nx:i*Nx]
    
    def get_b(self,A,i):
      #fonction pour le bloc B_i
      return A[i*Nx:(i+1)*Nx,(i)*Nx:(i+1)*Nx]
    
    def get_c(self,A,i):
      #fonction pour le bloc C_i
      return A[i*Nx:(i+1)*Nx,(i+1)*Nx:(i+2)*Nx]
    
    def get_d(self,D,i):
      #fonction pour le bloc D_i
      return D[i*Nx:(i+1)*Nx,0]
    
    # fonction principale
    def solve(self,A,F):
      d=np.copy(F)

      caches_w=[0] # pour stocker les Wi
      caches_b=[self.get_b(A,0)] # pour stocker les Bi
      caches_d=[self.get_d(d,0)] # pour stocker les Di

      for i in range(1,4*ny-1):
        caches_w.append(np.dot(self.get_a(A,i),np.linalg.inv(caches_b[i-1])))
        caches_b.append(self.get_b(A,i) - np.dot(caches_w[i],self.get_c(A,i-1)))
        caches_d.append(self.get_d(d,i) - np.dot(caches_w[i],caches_d[i-1]))

      T_new = np.zeros((N,1))
      T_new[N-Nx:N,0] = np.dot(np.linalg.inv(caches_b[4*ny-2]),caches_d[4*ny-2])

      for i in range(4*ny-3,-1,-1):
        T_new[i*Nx:(i+1)*Nx,0] = np.dot( np.linalg.inv(caches_b[i]),  caches_d[i] - np.dot(self.get_c(A,i),T_new[(i+1)*Nx:(i+2)*Nx,0]) )
      return T_new

def resout_solve(A,F):
  return np.linalg.solve(A,F)

def resout_sparse(A,F):
  AA = scipy.sparse.csr_matrix(A)
  return scipy.sparse.linalg.spsolve(AA,F)

def show3D(A,title):
    # pour représenter l'image en 3D
    m = [i*hx for i in range(A.shape[1])]
    p = [i*hy for i in range(A.shape[0])]
    X,Y = np.meshgrid(m, p)
    fig = plt.figure(figsize=(8,6))
    axe = fig.add_subplot(1,1,1, projection='3d')
    axe.view_init(30, 10)
    plt.title(title);
    pp = axe.plot_surface(X, Y, A, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
    cb = fig.colorbar(pp, shrink=0.5)
    axe.set_xlabel("X")
    axe.set_ylabel("Y")
    plt.show()
    
def animate3D(l, name,fps=5.0):
    # pour créer le fichier .gif d'une list l d'images
    kwargs_write = {'fps':fps, 'quantizer':'nq'}
    imageio.mimsave('./'+name+'.gif', l, fps=fps)
    
    
def image3D(A,title):
    # pour créer une image 3D
    m = [i*hx for i in range(A.shape[1])]
    p = [i*hy for i in range(A.shape[0])]
    X,Y = np.meshgrid(m, p)
    fig = plt.figure(figsize=(8,6))
    axe = fig.add_subplot(1,1,1, projection='3d')
    axe.view_init(30, 10)
    plt.title(title);
    axe.set_xlabel("X")
    axe.set_ylabel("Y")
    axe.set_xlim(0, Lx)
    axe.set_ylim(0, Ly)
    pp = axe.plot_surface(X, Y, A, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
    cb = fig.colorbar(pp, shrink=0.5)
    fig.canvas.draw() 
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image

def rotate3D(A,title):  
    # pour créer une liste d'image qui fait une tour de 360 dégrés
    l=[]
    for i in range(24):
        m = [i*hx for i in range(A.shape[1])]
        p = [i*hy for i in range(A.shape[0])]
        X,Y = np.meshgrid(m, p)
        fig = plt.figure(figsize=(8,6))
        axe = fig.add_subplot(1,1,1, projection='3d')
        plt.title(title);
        axe.set_xlabel("X")
        axe.set_ylabel("Y")
        axe.set_xlim(0, Lx)
        axe.set_ylim(0, Ly)
        pp = axe.plot_surface(X, Y, A, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
        cb = fig.colorbar(pp, shrink=0.5)
        axe.view_init(20, i*15)
        fig.canvas.draw() 
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        l.append(image)
    return l


def show2D(A,title):
  plt.figure(figsize=(8,6))
  plt.title(title)
  sns.heatmap(A, cmap="coolwarm") # l'image est inversée
  plt.xlabel("Y")
  plt.ylabel("X")
  plt.show()

def display_gif(name, fps=3):
    gif =imageio.mimread('./'+name+'.gif')
    nums = len(gif)
    imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in gif]
    i = 0
    while True:
        cv2.imshow("gif (tapez 'f' pour terminer)", imgs[i])
        if cv2.waitKey(100)&0xFF == ord("f"):
            break
        i = (i+1)%nums
        plt.pause(1/fps)
    cv2.destroyAllWindows()

def repesenter_stationnaire(A,title):
    cond_aff = int(input("0:représentation 2D | 1 : représentation 3d : "))
    if(cond_aff==0):
        show2D(A,title)
    else:
        cond_tour = int(input("0:schéma fixe |1 :Schéma tournant : "))
        if(cond_tour):
            print("il va prendre quelques secondes (taper 'f' pour fermer le gif)")
            l=rotate3D(A,title)
            name = title+"_gif"
            animate3D(l,name)
            display_gif(name)
        else:
            show3D(A,title)
def represneter_temps(l,title):
    print("il va prendre quelques secondes pour affichier le schéma (taper 'f' pour fermer le gif)")
    name = title+"gif"
    animate3D(l,name)
    display_gif(name,fps=10)
    

def const_a1():
    #bloc a1
    d1 = 2*(alpha_a+beta_a) * np.ones((Nx))
    d1[0] = alpha_a + 2 * beta_a # premier terme 
    d1[Nx-1] = alpha_a + 2 * beta_a # dernier terme
    d2 = (-alpha_a) * np.ones((Nx-1)) # diagonale supérieure et  diagonale inférieure
    a1 = np.diag(d1)+np.diag(d2,-1)+np.diag(d2,1)
    return a1

def const_b1():
    #bloc b1
    d3 = (-beta_a) * np.ones((Nx)) # diagonale
    b1 = np.diag(d3)
    return b1

def const_a2():
    #bloc a2:
    a2=np.zeros((Nx,Nx))
    d10 = (2*alpha_a + 2* beta_a) * np.ones((nx)) # premier bloc
    d20 =(2*alpha_a + 2* beta_a) * np.ones((nx)) # dernier bloc
    d10[0] -= alpha_a # premier terme
    d20[-1]-=alpha_a # dernier terme
    d30=np.zeros((Nx-2*nx)) # bloc au milieu
    a2+=np.diag(np.concatenate((d10,d30,d20))) # ajout du premier et denier bloc
    d30=((2*alpha_a +  beta_a + beta_c))*np.ones((Nx-2*nx)) # bloc au milieu
    d10 =  np.zeros((nx)) # premier bloc
    d20 = np.zeros((nx)) # dernier bloc
    d30[0]=(2*alpha_a + 2* beta_a) # 1er terme du bloc au milieu
    d30[-1]=(2*alpha_a + 2* beta_a) # dernier temer du bloc au milie
    a2+=np.diag(np.concatenate((d10,d30,d20))) # ajout du bloc au milieu
    d10 = (-alpha_a)*np.ones((Nx-1)) # diagonale supérieure et  diagonale inférieure
    a2+=np.diag(d10,1)+np.diag(d10,-1)
    return a2

def const_b2():
    #bloc b2
    d10 = (-beta_a) * np.ones((nx+1)) # 1er bloc
    d20 = (-beta_c) * np.ones((Nx-2*(nx+1))) # 2eme bloc
    b2 = np.diag((np.concatenate((d10,d20,d10)))) # ajout des blocs
    return b2

def const_a3():
    a3=np.zeros((Nx,Nx))
    d10 = (2*alpha_a + 2* beta_a) * np.ones((nx)) # 1er bloc
    d20 =(2*alpha_a + 2* beta_a) * np.ones((nx)) # dernier bloc
    d10[0] -= alpha_a # 1er terme
    d20[-1]-=alpha_a #dernier terme
    d30=np.zeros((Nx-2*nx)) # bloc au milieu
    a3+=np.diag(np.concatenate((d10,d30,d20))) # ajout du 1er et dernier blocs
    d30= (2*alpha_c + 2* beta_c)*np.ones((Nx-2*nx)) # bloc au milieu
    d10 =  np.zeros((nx)) # 1er bloc
    d20 = np.zeros((nx)) # dernier bloc
    d30[0]=alpha_a+alpha_c+2*beta_a # 1er terme du bloc au milieu
    d30[-1]=alpha_a+alpha_c+2*beta_a  # dernier terme du bloc au milieu
    a3+=np.diag(np.concatenate((d10,d30,d20)))  # ajout du bloc au milieu
    d10  = (-alpha_a)*np.ones((nx)) # 1er -dernier bloc du diagonal sup-inf
    d20  = (-alpha_c)*np.ones((Nx-1-2*(nx))) # bloc au milieu du diagonal sup-inf
    d30 = np.concatenate((d10,d20,d10)) # formation du diag sup-inf
    a3+=np.diag(d30,1)+np.diag(d30,-1)  # ajout des diag sup-inf
    return a3

def const_A(a1,b1,a2,b2,a3):
    #matrice A
    A=np.zeros((N,N))

    # ajout de a1
    d10 =  np.ones((ny-1))
    d20 = np.zeros((4*ny-1-2*(ny-1)))
    d30 = np.concatenate((d10,d20,d10))
    d30 = np.diag(d30)
    A += np.kron(d30,a1)
    #print(a1)

    #ajout de a3
    d10 =  np.zeros((ny-1))
    d20 = np.ones((4*ny-1-2*(ny-1)))
    d20[0]=0
    d20[-1]=0
    d30 = np.concatenate((d10,d20,d10))
    d30 = np.diag(d30)
    A += np.kron(d30,a3)

    #ajout de a2
    d10 =  np.zeros((ny-1))
    d20 = np.zeros((4*ny-1-2*(ny-1)))
    d20[0]=1.
    d20[-1]=1.
    d30 = np.concatenate((d10,d20,d10))
    d30 = np.diag(d30)
    A += np.kron(d30,a2)

    #ajout de b1
    d10 = np.ones((ny-1))
    d20 = np.zeros((4*ny-2-2*(ny-1)))
    d30 = np.concatenate((d10,d20,d10))

    d301 = np.diag(d30,1)
    d302 = np.diag(d30,-1)
    A+= np.kron(d301,b1) + np.kron(d302,b1)

    #ajout de b2
    d10 = np.zeros((ny-1))
    d20 = np.ones((4*ny-2-2*(ny-1)))
    d30 = np.concatenate((d10,d20,d10))
    d301 = np.diag(d30,1)
    d302 = np.diag(d30,-1)
    A+= np.kron(d301,b2) + np.kron(d302,b2)
    return A
def const_F0():
    #Matrice F0
    F0 = np.zeros((N,1))
    F0[0:Nx] =  beta_a * ( Tb * np.ones((Nx,1)) )
    F0[N-Nx:N] = beta_a * ( Th * np.ones((Nx,1)))
    return F0
def const_Tst(A,F):
    Tst_temp =  resout_sparse(A,F)
    return Tst_temp

def reshaper(T):
    return T.reshape((Ny,Nx))

def gamma1_gamma3(T_2D):
    gamma1 = Tb * np.ones((1,4*nx+1),dtype=float)
    gamma3 = Th * np.ones((1,4*nx+1),dtype=float)
    T_2D = np.concatenate((gamma1, T_2D), axis = 0)
    T_2D = np.concatenate((T_2D, gamma3), axis = 0)

def h(i,j):
  return (j-1)*Nx+i

def calc_F(pos):
  #pos est un tuple de la forme (xi,yi)
  Fi = np.zeros((N,1))
  xi,yi = pos # on récupere xi et yi du variable pos
  for i in range(Nx):
    for j in range(1,4*ny):
      lig = h(i,j) # position dans F
      x = i*hx # calcul de x
      y = j*hy # calcul de y
      nb = ((x-xi)**2 + (y-yi)**2)/(2*0.05**2) # calcul du terme dans l'exp
      Fi[lig,0]+= np.exp(-nb) # calcul du fi(x,y)
  Fi = Fi/(2*(0.05)**2*np.pi)
  return Fi

def const_T_resistance(r,pos,Tst0):
    T_temp = np.copy(Tst0)
    for i in range(len(r)):
      Fi = calc_F(pos[i])
      T_temp += r[i]*const_Tst(A,Fi)
      #print( max(resoud_tridiag(A,r[i]*Fi)))
    return T_temp

#fonction qui calcule B^n en log(n)*O(multiplication matricielle)
def puissance_log(B,n):
    res =  np.eye((B.shape[0]))
    cur = np.copy(B)
    while(n):
        if(n%2):
            res = np.dot(res,cur)
        n//=2
        cur = np.dot(cur,cur)
    return res
    
def const_B_explicite(dt):
    B = np.eye((4*nx+1)*(4*ny-1)) -  dt*A
    return B
def const_B_inv_implicite(dt):
    B = np.eye((4*nx+1)*(4*ny-1)) +  dt*A
    B_inv = np.linalg.inv(B)
    return B_inv

def const_T_0():
    return (Tb+Th)*0.5*np.ones((N,1))

def evolution_en_temps(B,Tst0, T_0,m=10000):
    Tst0 = Tst0.reshape((N,1))
    T0 = T_0 -Tst0
    lastT = np.copy(T0)
    lastT = lastT.reshape((N,1))
    res=np.eye((B.shape[0]))
    l=[] # pour stocker les courbes de Ttr en fonction de t dans différents instants
    l_T=[] # pour stocker les courbes de T en fonction de t dans différents instants
    T1=np.copy(T0) ## le résultat
    for i in range(20):
        res = puissance_log(B,i)
        T1 = np.dot(res,lastT)
        T1_2D  = T1.reshape((4*ny-1,4*nx+1))
        l.append(image3D(T1_2D,r"Ttr pour $t={}\Delta t$".format(i))) # ajouter l'image à la liste
        l_T.append(image3D(T1_2D+Tst0.reshape((Ny,Nx)),r"T pour $t={}\Delta t$".format(i)))
        
    for i in range(1,20):
        res = puissance_log(B,i*100)
        T1 = np.dot(res,lastT)
        T1_2D  = T1.reshape((4*ny-1,4*nx+1))
        l.append(image3D(T1_2D,r"Ttr pour $t={}\Delta t$".format(i*100))) # ajouter l'image à la liste
        l_T.append(image3D(T1_2D+Tst0.reshape((Ny,Nx)),r"T pour $t={}\Delta t$".format(i*100)))
        
    for i in range(2,m//1000+1):
        res = puissance_log(B,i*1000)
        T1 = np.dot(res,lastT)
        T1_2D  = T1.reshape((4*ny-1,4*nx+1))
        l.append(image3D(T1_2D,r"Ttr pour $t={}\Delta t$".format(i*1000))) # ajouter l'image à la liste
        l_T.append(image3D(T1_2D+Tst0.reshape((Ny,Nx)),r"T pour $t={}\Delta t$".format(i*1000)))
        
    for i in range(2,10):
        res = puissance_log(B,i*10000)
        T1 = np.dot(res,lastT)
        T1_2D  = T1.reshape((4*ny-1,4*nx+1))
        l.append(image3D(T1_2D,r"Ttr pour $t={}\Delta t$".format(i*10000))) # ajouter l'image à la liste
        l_T.append(image3D(T1_2D+Tst0.reshape((Ny,Nx)),r"T pour $t={}\Delta t$".format(i*10000)))
    print("solution calculée")
    return (l,l_T)

#calcul de a_tilde(ij):
def calcul_aij(Ti,Tj):
    ans=0.0
    for i in range(2*nx):
        for j in range(2*ny):
            posi = nx+i
            posj = ny+j
            terme1 = Ti[h(posi,posj),0]*Tj[h(posi,posj),0]
            terme2 = Ti[h(posi,posj+1),0]*Tj[h(posi,posj+1),0]
            terme3 = Ti[h(posi+1,posj),0]*Tj[h(posi+1,posj),0]
            terme4 = Ti[h(posi+1,posj+1),0]*Tj[h(posi+1,posj+1),0]
            ans+=(hx*hy/4)*(terme1+terme2+terme3+terme4)
    return ans
#calcul de A_tilde
def calcul_A(l_T):
    #l_T est une liste qui contient les Tst_i
    A_tilde = np.zeros((len(l_T),len(l_T)))
    for i in range(len(l_T)):
        for j in range(i,len(l_T)):
            if(i==j):
                A_tilde[i,j]=calcul_aij(l_T[i],l_T[i])
            else :
                A_tilde[i,j]=A_tilde[j,i]=calcul_aij(l_T[i],l_T[j])
    return A_tilde
#calcul de bi
def calcul_bi(Ti,T0,Tc):
    ans=0.0
    for i in range(2*nx):
        for j in range(2*ny):
            posi = nx+i
            posj = ny+j
            terme1 = (T0[h(posi,posj),0]-Tc)*Ti[h(posi,posj),0]
            terme2 = (T0[h(posi,posj+1),0]-Tc)*Ti[h(posi,posj+1),0]
            terme3 = (T0[h(posi+1,posj),0]-Tc)*Ti[h(posi+1,posj),0]
            terme4 = (T0[h(posi+1,posj+1),0]-Tc)*Ti[h(posi+1,posj+1),0]
            ans+=(hx*hy/4)*(terme1+terme2+terme3+terme4)
    return ans
#calcul de b tilde
def calcul_b(l_T,T0,Tc):
    b_tilde = np.zeros((len(l_T),1))
    for i in range(len(l_T)):
        b_tilde[i,0]=calcul_bi(l_T[i],T0,Tc)
    return b_tilde
# calcul de  l'erreur : 
def calcul_e(T,Tc):
    ans=0.0
    for i in range(2*nx):
        for j in range(2*ny):
            posi = nx+i
            posj = ny+j
            terme1 = (T[h(posi,posj),0]-Tc)**2
            terme2 = (T[h(posi,posj+1),0]-Tc)**2
            terme3 = (T[h(posi+1,posj),0]-Tc)**2
            terme4 = (T[h(posi+1,posj+1),0]-Tc)**2
            ans+=(hx*hy/4)*(terme1+terme2+terme3+terme4)
    return ans

def const_T_opt(pos,Tst0,T_c):
    Nr = len(pos)
    l_T = [] # pour stocker les Tst_i
    for i in range(Nr):
        Fi = calc_F(pos[i])
        Ti = const_Tst(A,Fi)
        Ti = Ti.reshape((Ti.shape[0],1))
        l_T.append(Ti)
    A_tilde = calcul_A(l_T) # calcul de A tilde
    Tst0=Tst0.reshape((Tst0.shape[0],1))
    b_tilde = calcul_b(l_T,Tst0,T_c) # calcul de b tilde
    r = resout_sparse(A_tilde,-b_tilde) # résolution de système
    r= r.reshape((Nr,1))
    T_opt  = np.copy(Tst0) 
    for i in range(Nr):
        T_opt += r[i,0]*l_T[i] # calcul de T finale
    return T_opt
    
    

        

Lx = float(input("Lx = "))
Ly = float(input("Ly = "))
Ka = float(input("la conductivité dans la régione a : Ka = "))
Kc = float(input("la conductivité dans la régione c : Kc = "))
nx = int(input("nx = "))
ny = int(input("ny = "))
Tb = float(input("Tb = "))
Th = float(input("Th = "))

N = (4*nx+1)*(4*ny-1)
Nx = (4*nx+1)
Ny = (4*ny-1)

hx = Lx/(4*nx) # pas dans la directon x
hy = Ly/(4*ny) # pas dans la direction y

alpha_a = Ka / (hx**2)
beta_a = Ka / (hy**2)
alpha_c = (Kc) / (hx**2)
beta_c = (Kc) /(hy**2)

a1 = const_a1()
b1 = const_b1()
a2 = const_a2()
b2 = const_b2()
a3 = const_a3()
A = const_A(a1,b1,a2,b2,a3)
F0 = const_F0()
print("--------------------Calcul de la température stationnaire--------------------------")
Tst0 = const_Tst(A,F0)
Tst_2D = reshaper(Tst0)
gamma1_gamma3(Tst_2D)
print("--------------------Représentation de la température stationnaire-------------")
repesenter_stationnaire(Tst_2D,"Température Stationnaire")

print("--------------------Calcul de la température stationnaire avec résistance--------------------------")
cond_resistance = int(input("1 : pour ajouter des résistances| 0 sinon :  "))
if(cond_resistance==1):
    Nr = int(input("Donner le nombre de résistances : Nr = "))
    r = []
    pos= []
    for i in range(Nr):
        x,y,re = map(float,input("Donner les informations de la résistance {} sous la forme a b r avec x = a*Lx et y = b*Ly : ".format(i+1)).split())
        r.append(re)
        x*=Lx
        y*=Ly
        pos.append((x,y))
    Tst_resistance = const_T_resistance(r,pos,Tst0)
    Tst_2D_resistance = reshaper(Tst_resistance)
    gamma1_gamma3(Tst_2D_resistance)
    print("--------------------Représentation de la température stationnaire avec résistance-------------")
    repesenter_stationnaire(Tst_2D_resistance,"Température Stationnaire avec résistance(s)")
print("--------------------Résolution du problème transitoire--------------------------")
cond_implicite = int(input("choisir la méthode de résolution  : 0:explicite|1:implicite : "))
T_0 = const_T_0()
dt=0.00001
if(cond_implicite==0):
    cond = 1/(4*max((Ka,Kc))*(1/(hx**2)  + 1/(hy**2)))
    dt = float(input("donner dt tel que dt<%.8f : dt = "%cond))
    B = const_B_explicite(dt)
    print("--------------------Calcul de la température stationnaire--------------------------")
    l,l_T = evolution_en_temps(B,Tst0, T_0)
    print("--------------------Représentation la température stationnaire--------------------------")
    represneter_temps(l,"solution transitoire")
    print("--------------------Représentation la température totale--------------------------")
    represneter_temps(l_T,"solution totale")
elif(cond_implicite==1):
    dt = float(input("donner dt sans condition : dt = "))
    B_inv = const_B_inv_implicite(dt)
    print("--------------------Calcul de la température stationnaire--------------------------")
    l,l_T = evolution_en_temps(B_inv,Tst0, T_0)
    print("--------------------Représentation la température stationnaire--------------------------")
    represneter_temps(l,"solution transitoire")
    print("--------------------Représentation la température totale--------------------------")
    represneter_temps(l_T,"solution totale")

print("--------------------Résolution du problème inverse--------------------------")
T_c = float(input("Donner :  T_c  = "))
Nr = int(input("Donner le nombre de résistances : Nr = "))
pos= []
for i in range(Nr):
    x,y= map(float,input("Donner les informations de la résistance {} sous la forme a b avec x = a*Lx et y = b*Ly : ".format(i+1)).split())
    x*=Lx
    y*=Ly
    pos.append((x,y))
print("--------------------Calcul de la température optimale--------------------------")
T_opt = const_T_opt(pos,Tst0,T_c)
T_opt_2D = reshaper(T_opt)
gamma1_gamma3(T_opt_2D)
print("--------------------Représentation de la température optimale-------------")
repesenter_stationnaire(T_opt_2D,"Température Stationnaire")


    
    




