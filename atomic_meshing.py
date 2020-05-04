import numpy as np
from shapely.geometry import box, Polygon, Point
from scipy.optimize import minimize
from skimage.io import imread
from mesh_init import initial_lattice
import time

#TODO
def distance(v1, v2):
    return np.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)

class Atomic_Mesh_Generator:

    def __init__(self, atoms, image = imread("../test_images/circle.png",as_gray=True), nominal_distance = 10, beta = 0.5):
        self.atoms = atoms
        self.image = image

        # nominal distance (force at which two atoms go from being repulsive to being attractive)
        self.d = nominal_distance  # could be determined by resolution

        #relative contribution of atomic and image potential to the total potential energy
        self.beta = beta

        #the scale of the image
        self.img_scale = 2


    # scalar potential of two atoms modelled by a polynomial function
    # which has its minimum at u = 1
    #
    def atomic_potential(self,atom1, atom2):
        dist = distance(atom1, atom2)

        #TODO if speedup necessary create a table lookup here for precomputed distances

        u = dist / self.d  # normalized distance

        if 0 < u < 3 / 2:
            return 153 / 256 - 9 / 8 * u + 19 / 24 * u ** 3 - 5 / 16 * u ** 4
        else:
            return 0

    #attractive force of an image feature to an atom (darker pixels attract stronger) minimum is at -1 to enforce regions of interest with minimization
    #TODO experiment here with function (magnets can be modelled by 1/u^2)
    def image_feature_force(self,atom,pixel_pos,pixel_val):
        dist = distance(atom,pixel_pos)
        u = dist / self.d  # normalized distance
        if 0 < u < 3/2:
            return -1 * pixel_val * (1 / (1 + u))
        else:
            return 0


    #TODO visualize
    def total_potential_energy(self, atoms):
        P = 0
        #tm_total = time.time()
        mean_ipf = 0
        mean_apf = 0
        for i,a1 in enumerate(atoms):
            atomic_potential_field = 0
            #tm = time.time()
            for j,a2 in enumerate(atoms):
                if i != j:
                    atomic_potential_field += (1-self.beta)*self.atomic_potential(a1,a2)
            #print("apf time ",time.time() - tm)
            image_potential_field = 0
            tm = time.time()
            #TODO the image potential energy really depends on the density and total amount of pixels
            #only take every nth pixel
            n = 100
            for row in range(0,self.image.shape[0],n):
                for col in range(0,self.image.shape[1],n):
                    rows = self.image.shape[0]
                    pixel_val = self.image[row][col]
                    if pixel_val > 0:
                        image_potential_field += self.image_feature_force(a1,(rows-row,col),pixel_val)
                    #print(a1,rows-row,col,image_potential_field)
            #print("ipf time",time.time()-tm)
            mean_ipf += image_potential_field
            mean_apf += atomic_potential_field
            P = P + 0.5 * (atomic_potential_field + self.beta * image_potential_field)
        #print("time total ",time.time() - tm_total)
        print(mean_ipf,mean_apf)
        return P

    def total_potential_energy_1D(self, atoms):
        return self.total_potential_energy(atoms.reshape(-1,2))

    def random_perturb(self):
        self.atoms = self.atoms + np.random.normal(0,self.d/5,size=self.atoms.shape)

    """
    We could test different minimizers here. For example cobyla with a constraint 
    of atoms being inside boundary
    """
    def optimize_lattice(self):
        while True:
            P_init = self.total_potential_energy(self.atoms)
            print("Total potential energy", P_init)
            self.random_perturb()
            res = minimize(self.total_potential_energy_1D, self.atoms.ravel(),method='L-BFGS-B', jac='2-point',tol = 0.01)
            print("success ",res.success)
            self.atoms = res.x.reshape(-1,2)
            #stopping condition
            print("difference in potential energy ",P_init - self.total_potential_energy(self.atoms), P_init*0.01)
            if np.abs(P_init - self.total_potential_energy(self.atoms)) < np.abs(P_init * 0.01):
                break
        return self.atoms


def image_boundary():
    pass

#quick test with grid around circle
atoms = np.array(initial_lattice(cellsize=50)["vertices"])
img = imread("../test_images/circle.png",as_gray=True)
print(atoms.shape,img.shape,img.max())
generator = Atomic_Mesh_Generator(atoms,img,nominal_distance=40,beta=0.5)
optimized_lattice = generator.optimize_lattice()

import turtle
import view

turtle.speed(0)
turtle.tracer(0)
turtle.setworldcoordinates(0,0,2000,1000)
turtle.bgpic("../test_images/circle.png")
boundary = Polygon([(0,0),(500,0),(0,500)])

#view.draw_graph(turtle.Turtle(),graph = {"vertices":initial_lattice(cellsize=50,boundary=boundary)["vertices"],"edges":[]})
view.draw_graph(turtle.Turtle(),graph = {"vertices":optimized_lattice,"edges":[]})
turtle.mainloop()