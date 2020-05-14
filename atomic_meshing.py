import numpy as np
from shapely.geometry import box, Polygon, Point
from scipy.optimize import minimize
from skimage.io import imread
from mesh_init import initial_lattice
import time
import turtle


#TODO
def distance(v1, v2):
    return np.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)

"""
A class that calculates and minimizes the total energy function of a lattice given an image.
"""
class Atomic_Mesh_Generator:

    def __init__(self, atoms, image, nominal_distance = 10, beta = 0.5):
        self.atoms = atoms
        if type(image) == np.ndarray:
            self.image = image
        else:
            self.image = imread(image,as_gray=True)

        # nominal distance (force at which two atoms go from being repulsive to being attractive)
        self.d = nominal_distance  # could be determined by resolution

        #relative contribution of atomic and image potential to the total potential energy
        self.beta = beta

        #the scale of the image
        self.img_scale = 2

        self.f_evals = 0


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
        self.f_evals += 1
        print(self.f_evals)
        P = 0
        #tm_total = time.time()
        mean_ipf = 0
        mean_apf = 0
        #print("pixels :", np.array(np.nonzero(self.image)).T.size / n)
        for i,a1 in enumerate(atoms):
            atomic_potential_field = 0
            tm = time.time()
            for j,a2 in enumerate(atoms):
                if i != j:
                    atomic_potential_field += (1-self.beta)*self.atomic_potential(a1,a2)
            print("apf time ",time.time() - tm)

            image_potential_field = 0
            tm = time.time()
            #TODO limit amount of pixels to whats necessary
            n = 100
            for i,pix in enumerate(np.array(np.nonzero(self.image)).T):
                if i % n == 0:
                    x = pix[0]
                    y = pix[1]
                    pixel_val = self.image[x][y]
                    if pixel_val > 0:
                        image_potential_field += self.image_feature_force(a1,(x,y),pixel_val)
            #print("ipf time",time.time()-tm)
            mean_ipf += image_potential_field
            mean_apf += atomic_potential_field
            P = P + 0.5 * (atomic_potential_field + self.beta * image_potential_field)
        #print("time total ",time.time() - tm_total)
        #print("ImagePF/AtomPF",mean_ipf,mean_apf)
        return P

    def total_potential_energy_1D(self, atoms):
        return self.total_potential_energy(atoms.reshape(-1,2))

    def random_perturb(self):
        self.atoms = self.atoms + np.random.normal(0,self.d/5,size=self.atoms.shape)

    """
    Could test different minimizers here. For example L-BFGS with a constraint 
    of atoms being inside boundary
    """
    def optimize_lattice(self):
        while True:
            P_init = self.total_potential_energy(self.atoms)
            print("Total potential energy", P_init)
            self.random_perturb()
            res = minimize(self.total_potential_energy_1D, self.atoms.ravel(),method='l-BFGS-B', jac='2-point',options={"maxfun":10,"disp":True})
            print("success ",res.success)
            self.atoms = res.x.reshape(-1,2)
            #stopping condition
            print("difference in potential energy ",P_init - self.total_potential_energy(self.atoms), P_init*0.01)
            if np.abs(P_init - self.total_potential_energy(self.atoms)) < np.abs(P_init * 0.01):
                break
        return self.atoms

    """
    Single call of the function 'minimze'. Can be used to display moving atoms. 
    """
    def optimization_step(self):
        tm = time.time()
        P_init = self.total_potential_energy(self.atoms)
        print("Total potential energy", P_init)
        res = minimize(self.total_potential_energy_1D, self.atoms.ravel(), method='L-BFGS-B', jac='2-point',
                       options={"maxiter": 5, "disp": True})
        print("success ", res.success)
        self.atoms = res.x.reshape(-1, 2)
        print("difference in potential energy ", P_init - self.total_potential_energy(self.atoms), P_init * 0.01)
        print(time.time() - tm,"seconds")
        return self.atoms


turtle.speed(0)
turtle.tracer(0)
turtle.setworldcoordinates(0,0,2000,1000)
turtle.bgpic("../test_images/circle.png")
boundary = Polygon([(0,0),(500,0),(0,500)])

#quick test with grid around circle
atoms = np.array(initial_lattice(cellsize=50)["vertices"])
print(atoms.size)
img = imread("../test_images/circle.png",as_gray=True)
print(atoms.shape,img.shape,img.max())
#generator = Atomic_Mesh_Generator(atoms,img,nominal_distance=40,beta=0.5)
#optimized_lattice = generator.optimize_lattice()

#view.draw_graph(turtle.Turtle(),graph = {"vertices":initial_lattice(cellsize=50,boundary=boundary)["vertices"],"edges":[]})
#view.draw_graph(turtle.Turtle(),graph = {"vertices":optimized_lattice,"edges":[]})
