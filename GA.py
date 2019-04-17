import matplotlib.pyplot as plt
import numpy as np

cities = 50
mutate_rate = 1
pop_size = 50
cross_rate=0.1
selections = 100000000000000

class GA(object):
    def __init__(self, cities, cross_rate, mutate_rate, pop_size):
        self.cities = cities
        self.mutate_rate = mutate_rate
        self.pop_size = pop_size
        self.cross_rate = cross_rate
        self.city_position = np.random.rand(cities, 2)
        self.pop = np.vstack([np.random.permutation(cities) 
                                for _ in range(pop_size)])
        
    def translate(self, popul):     
        line_x = np.empty_like(popul, dtype=np.float64)
        line_y = np.empty_like(popul, dtype=np.float64)
        for i, d in enumerate(popul):
            city_coord = self.city_position[d]
            line_x[i, :] = city_coord[:, 0]
            line_y[i, :] = city_coord[:, 1]
        return line_x, line_y
    
    def get_distance(self, line_x, line_y):
        total_distance = np.empty((line_x.shape[0]), dtype=np.float64)
        for i, (xs, ys) in enumerate(zip(line_x, line_y)):
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
            total_distance[i] += np.sum(np.sqrt(np.square(xs[-1]-xs[1]) + np.square(ys[-1]-ys[1])))
        return total_distance
    
    def min_distance(self, pops, total_distance):
        pop_best=np.random.randint(80,100)
        pop_n_bad=100-pop_best
        pop_prob_best=(pop_best*self.pop_size)/100
        pop_prob_bad=(pop_n_bad*self.pop_size)/100
        pop_prob_best=round(pop_prob_best)
        pop_prob_bad=round(pop_prob_bad)
        pop=[]
        dic={x:y for i ,(x,y) in enumerate(zip(total_distance,pops))}
        sorted_x = sorted(dic.items())
        k=0
        for i,j in sorted_x:
            pop.append(j)
            k+=1
            if k == pop_prob_best:
                break
        k=0
        if pop_prob_bad !=0:               
            for i,j in reversed(sorted_x):
                pop.append(j)
                k+=1
                if k == pop_prob_bad:
                    break
        return pop
        
    def evolve(self):
        childs=[]
        for i in self.pop:
            childs.append(i)
        pop_copy=self.pop.copy()
        for parent in pop_copy:
            child = self.cross(parent,pop_copy)
            child = self.mutate(child)
            childs.append(child)
            pops = np.array(childs)
        lx, ly = self.translate(pops)    
        total_distance = self.get_distance(lx, ly)
        parents = self.min_distance(pops,total_distance)
        self.pop[:] = np.array(parents)
             
    def cross(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            i = np.random.randint(0, self.pop_size, size=1) 
            cross_points = np.random.randint(0, 2, self.cities).astype(np.bool)
            keep_city = parent[~cross_points] 
            swap_points = np.isin(pop[i].ravel(), keep_city, invert=True)
            swap_city = pop[i, swap_points]
            parent[:] = np.concatenate((keep_city, swap_city))
        return parent
    
    def mutate(self, child):
        if np.random.rand() < self.mutate_rate:
            point=[]
            swap_point_1 = np.random.randint(0, self.cities/2)
            swap_point_2 = np.random.randint(0, self.cities/2)
            point.append(swap_point_1)
            point.append(swap_point_2)
            point=sorted(point)
            g=[child[i] for i in range(point[0])]
            point_rev=child[::-1]
            g1=[point_rev[i] for i in range(point[1])]
            child=[i for i in child if i not in g if i not in g1]
            child=np.concatenate((g,child[::-1]))
            child=np.concatenate((child,g1[::-1]))
            child=[int(i) for i in child]
        return child

def plotting(total_d,generation, city_position):
    x = list(city_position[:, 0])
    y = list(city_position[:, 1])
    x.append(x[0]); y.append(y[0])
    plt.cla()
    plt.scatter(city_position[:, 0], city_position[:, 1], s=20, c='b')
    plt.plot(x, y, 'g-')
    plt.title("Total distance=%.2f" % total_d)
    plt.savefig('images/%d.png' % generation)    

        
ga = GA(cities,cross_rate,mutate_rate,pop_size)
td=0   
ln=0 
for sel in range(selections):
    ga.evolve()
    lx, ly = ga.translate(ga.pop)
    total_distance = ga.get_distance(lx, ly)
    if total_distance[0] != td:
        td=total_distance[0]
        print(total_distance[0])
        plotting(total_distance[0],sel,ga.city_position[ga.pop[0]])
       
lx, ly = ga.translate(ga.pop)
total_distance = ga.get_distance(lx, ly)
print(total_distance[0])
plotting(total_distance[0],sel,ga.city_position[ga.pop[0]])  
