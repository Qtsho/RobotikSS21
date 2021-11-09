import math
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians, pi
"""

LIDAR to 2D grid map example


"""

import math
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

EXTEND_AREA = 1.0


def file_read(f):
    """
    Reading LIDAR laser beams (angles and corresponding distance data)
    """
    with open(f) as data:
        measures = [line.split(",") for line in data]
    angles = []
    distances = []
    for measure in measures:
        angles.append(float(measure[0]))
        distances.append(float(measure[1]))
    angles = np.array(angles)
    distances = np.array(distances)
    return angles, distances


def bresenham(start, end):
    """
    Implementation of Bresenham's line drawing algorithm
    See en.wikipedia.org/wiki/Bresenham's_line_algorithm
    Bresenham's Line Algorithm
    Produces a np.array from start and end (original from roguebasin.com)
    >>> points1 = bresenham((4, 4), (6, 10))
    >>> print(points1)
    np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
    """
    # setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)  # determine how steep the line is
    if is_steep:  # rotate line
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    # swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1  # recalculate differentials
    dy = y2 - y1  # recalculate differentials
    error = int(dx / 2.0)  # calculate error
    y_step = 1 if y1 < y2 else -1
    # iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:  # reverse the list if the coordinates were swapped
        points.reverse()
    points = np.array(points)
    return points


def calc_grid_map_config(ox, oy, xy_resolution):
    """
    Calculates the size, and the maximum distances according to the the
    measurement center
    """
    min_x = round(min(ox) - EXTEND_AREA / 2.0)
    min_y = round(min(oy) - EXTEND_AREA / 2.0)
    max_x = round(max(ox) + EXTEND_AREA / 2.0)
    max_y = round(max(oy) + EXTEND_AREA / 2.0)
    xw = int(round((max_x - min_x) / xy_resolution))
    yw = int(round((max_y - min_y) / xy_resolution))
    print("Die Größe der Belegtheitskarte ist ", xw, "x", yw, ".")
    return min_x, min_y, max_x, max_y, xw, yw


def atan_zero_to_twopi(y, x):
    angle = math.atan2(y, x)
    if angle < 0.0:
        angle += math.pi * 2.0
    return angle




def generate_ray_casting_grid_map(ox, oy, xy_resolution, breshen=True):
    """
    The breshen boolean tells if it's computed with bresenham ray casting
    (True) or with flood fill (False)
    """
    min_x, min_y, max_x, max_y, x_w, y_w = calc_grid_map_config(
        ox, oy, xy_resolution)
    # default 0.5 -- [[0.5 for i in range(y_w)] for i in range(x_w)]
    occupancy_map = np.ones((x_w, y_w)) / 2
    center_x = int(
        round(-min_x / xy_resolution))  # center x coordinate of the grid map
    center_y = int(
        round(-min_y / xy_resolution))  # center y coordinate of the grid map
    # occupancy grid computed with bresenham ray casting
    if breshen:
        for (x, y) in zip(ox, oy):
            # x coordinate of the the occupied area
            ix = int(round((x - min_x) / xy_resolution))
            # y coordinate of the the occupied area
            iy = int(round((y - min_y) / xy_resolution))
            laser_beams = bresenham((center_x, center_y), (
                ix, iy))  # line form the lidar to the occupied point
            for laser_beam in laser_beams:
                occupancy_map[laser_beam[0]][
                    laser_beam[1]] = 0.0  # free area 0.0
            occupancy_map[ix][iy] = 1.0  # occupied area 1.0
            occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
            occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
            occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area
    # occupancy grid computed with with flood fill
    else:
        occupancy_map = init_flood_fill((center_x, center_y), (ox, oy),
                                        (x_w, y_w),
                                        (min_x, min_y), xy_resolution)
        flood_fill((center_x, center_y), occupancy_map)
        occupancy_map = np.array(occupancy_map, dtype=float)
        for (x, y) in zip(ox, oy):
            ix = int(round((x - min_x) / xy_resolution))
            iy = int(round((y - min_y) / xy_resolution))
            occupancy_map[ix][iy] = 1.0  # occupied area 1.0
            occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
            occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
            occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area
    return occupancy_map, min_x, max_x, min_y, max_y, xy_resolution


def main():
    """
    Example usage
    """
    print(__file__, "start")
    xy_resolution = 0.02  # x-y grid resolution
    ang, dist = file_read("csv/lidar_datei.csv")
    ox = np.sin(ang) * dist
    oy = np.cos(ang) * dist
    occupancy_map, min_x, max_x, min_y, max_y, xy_resolution = \
        generate_ray_casting_grid_map(ox, oy, xy_resolution, True)
    xy_res = np.array(occupancy_map).shape
    plt.figure(1, figsize=(10, 4))
    plt.subplot(122)
    plt.imshow(occupancy_map, cmap="PiYG_r")
    # cmap = "binary" "PiYG_r" "PiYG_r" "bone" "bone_r" "RdYlGn_r"
    plt.clim(-0.4, 1.4)
    plt.gca().set_xticks(np.arange(-.5, xy_res[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(-.5, xy_res[0], 1), minor=True)
    plt.grid(True, which="minor", color="w", linewidth=0.6, alpha=0.5)
    plt.colorbar()
    plt.subplot(121)
    plt.plot([oy, np.zeros(np.size(oy))], [ox, np.zeros(np.size(oy))], "ro-")
    plt.axis("equal")
    plt.plot(0.0, 0.0, "ob")
    plt.gca().set_aspect("equal", "box")
    bottom, top = plt.ylim()  # return the current y-lim
    plt.ylim((top, bottom))  # rescale y axis, to match the grid orientation
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()


def file_read(f):
    """
    Reading LIDAR laser beams (angles and corresponding distance data)
    """
    measures = [line.split(",") for line in open(f)]
    angles = []
    distances = []
    for measure in measures:
        angles.append(float(measure[0]))
        distances.append(float(measure[1]))
    angles = np.array(angles)
    distances = np.array(distances)
    return angles, distances

ang, dist = file_read("csv/lidar_datei.csv")
ox = np.sin(ang) * dist
oy = np.cos(ang) * dist

plt.figure(figsize=(6,10))
plt.plot([oy, np.zeros(np.size(oy))], [ox, np.zeros(np.size(oy))], "o:r") 

plt.axis("equal")
## flip then plot
bottom, top = plt.ylim()  # return the current ylim
print (plt.ylim())
plt.ylim((top, bottom)) # rescale y axis, to match the measure orientation 
print (plt.ylim())
plt.grid(True)
plt.title('Laser Messungen')
plt.show()
def bresenham(start, end):
    """
    Implementation of Bresenham's line drawing algorithm
    See en.wikipedia.org/wiki/Bresenham's_line_algorithm
    Bresenham's Line Algorithm
    Produces a np.array from start and end (original from roguebasin.com)
    >>> points1 = bresenham((4, 4), (6, 10))
    >>> print(points1)
    np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
    """
    # setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)  # determine how steep the line is
    if is_steep:  # rotate line
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    # swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1  # recalculate differentials
    dy = y2 - y1  # recalculate differentials
    
    error = int(dx / 2.0)  # calculate error
    y_step = 1 if y1 < y2 else -1
    # iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:  # reverse the list if the coordinates were swapped
        points.reverse()
    points = np.array(points)
    return points



map1 = np.ones((50, 50)) * 0.5 # Karteninitialisierung mit Vorwissen
line = bresenham((2, 2), (40, 30)) # eine Brsenham-Linie erstellen: Hinweis für die Aufgabe.
for l in line:
    map1[l[0]][l[1]] = 1
    
plt.imshow(map1, cmap = 'Greys') 
plt.colorbar()
plt.title('Bresenham Line')
plt.show()


def generate_ray_casting_grid_map(ox, oy, xy_resolution):
    """
    Kartieren-Funktion
    """
    #Verwenden der Funktion calc_grid_map_config() (in anderem Scrikt) zum Erstellen einer Grid Map
    min_x, min_y, max_x, max_y, x_w, y_w = calc_grid_map_config(ox, oy, xy_resolution)
    
    #Initialisieren der Grid Map mit p = 0,5 / logodd is 0
    occupancy_map = np.zeros((x_w, y_w))  #PRIOR = 0
    
    #Berechnen der Mittelpunktskoordinate der Karte
    center_x = int(round(-min_x / xy_resolution))  # Zentrum x-Koordinate der Rasterkarte
    center_y = int(round(-min_y / xy_resolution))  # Zentrum y-Koordinate der Rasterkarte
    
    l_occ = np.log(0.65/0.35) #logodd besetzt
    l_free = np.log(0.35/0.65) #logodd frei
    r = 1 #kann verändern

    #Schleife 3 mal.
    for update in range(3): 
        #while not done:
        #
        #Schleife durch alle Laserpunkt
        for (x, y) in zip(ox, oy):
            #x-Koordinate des belegten Bereichs
            ix = int(round((x - min_x) / xy_resolution))
            #y-Koordinate des belegten Bereichs
            iy = int(round((y - min_y) / xy_resolution))
            """
            Aufgabe:
            Aktualisieren der Belegtheitskarte mit inversem Sensormodell und  Bresenham ray-casting
            
            Hinweis: 
            + Verwenden Sie bresenham(), um den Strahl von der Mittelpunktskoordinate (center_x, center_y) nach (ix,iy) 
            zu berechnen. Wie die vorgesehene Figur gezeicht wurdet und diese durch das Aktualisieren des Grid ersetzen.
            Aktualisieren der 2D nparray occupancy_map mit lfree (mit einem for loop, sehen Sie wie map1 aktulisiert wird)
            
            + Aktualisieren der 2D nparray occupancy_map mit l_occ (mit ix, iy)
            
            +Fühlen Sie sich frei, r zu ändern. Und fühlen Sie sich auch frei, zu aktualisieren, wie viele Gitter um die 
            Messung, solange es eine schöne Grid-Karte zu erzeugen.
            
            + 6-10 lines of code.
            
            + Ref: Occupancy Mapping Algorithm: http://ais.informatik.uni-freiburg.de/teaching/ws12/mapping/pdf/slam11-gridmaps-4.pdf
            
            Ergebnis: 
            occupancy_map 2Dnparray: aktualisierte Karte m 

             """   
          
            
            # YOUR CODE HERE
            
            line = bresenham((center_x, center_y), (ix, iy))
            
            occupancy_map[ix][iy] += l_occ 
            occupancy_map[ix+r][iy+r] += l_occ
            occupancy_map[ix+r][iy] += l_occ
            occupancy_map[ix][iy+r] += l_occ
            
            for l in line:
                occupancy_map[l[0]][l[1]] += l_free   
            
           
    return occupancy_map, min_x, max_x, min_y, max_y, xy_resolution


'''
MAIN CODE
Initialisieren der Karte

Sie können mit der Gridauflösung herumspielen.
'''
xyreso = 0.01  # x-y Gridauflösung
yawreso = math.radians(3.1)  # yaw angle resolution [rad]
ang, dist = file_read("csv/lidar_datei.csv")
ox = np.sin(ang) * dist #nparray von x Koordinaten
oy = np.cos(ang) * dist #nparray von y Koordinaten.
print (ox)
occupancy_map, minx, maxx, miny, maxy, xyreso = generate_ray_casting_grid_map(ox, oy, xyreso)
xyres = np.array(occupancy_map).shape



#das Belegungsraster aufzeichnen
plt.figure(figsize=(20,10))
plt.subplot(122)
plt.imshow(occupancy_map, cmap = 'Greys') 
plt.clim(-0.4, 1.4)
plt.gca().set_xticks(np.arange(-.5, xyres[1], 1), minor = True)
plt.gca().set_yticks(np.arange(-.5, xyres[0], 1), minor = True)
plt.colorbar() 
plt.title('Belegtheitskarte')

#die Messung aufzeichnen

plt.subplot(121)
plt.plot([oy, np.zeros(np.size(oy))], [ox, np.zeros(np.size(oy))], "o:r") # Linien von 0,0 bis zu den Messungen
plt.axis("equal")
plt.plot(0.0, 0.0, "ob")#lidar position
plt.gca().set_aspect("equal", "box")
bottom, top = plt.ylim()  # gibt den aktuellen y-Lim zurück
plt.ylim((top, bottom))  # y-Achse neu skalieren, um die Gitterausrichtung anzupassen
plt.grid(True)
plt.title('Messungen')
plt.show()

    
