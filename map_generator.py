import numpy as np
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
from opensimplex import noise2 as SimplexNoise
from opensimplex import random_seed as SimplexNoiseSeed
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
from tqdm import tqdm  # Import de tqdm pour la barre de progression
from collections import defaultdict


class MapParameters:
    """Classe pour stocker les paramètres de génération de la carte."""

    def __init__(self, width=200, height=200,
                 scale_octave=2.0,
                 octaves=6,
                 voronoi_scale=4,
                 amplitude_func="Pow",
                 water_sea_level=0.2,
                 mountaigns_level=0.8,
                 altitude_transform_mixing=0.2,
                 altitude_transform_exponent=1.0,
                 distribution_exponent=1.0,
                 show_progress_bar=None,
                 raininess=0.5,
                 rain_shadow=0.2,
                 evaporation=0.3,
                 angle_variation_winds = 15.0,
                 num_angles_winds = 5,
                 water_flow_threshold=0.005):
        """
        raininess : Contrôle la quantité de pluie générée par l'humidité des nuages.
            Valeur recommandée : 0.5 à 1.0
            Une valeur plus élevée signifie plus de pluie dans les zones où les nuages transportent de l'humidité.

        rain_shadow : Contrôle l'effet de pluie orographique (pluie causée par l'élévation des montagnes).
            Valeur recommandée : 0.2 à 0.5
            Une valeur plus élevée accentue les précipitations sur le côté exposé aux montagnes et assèche davantage les zones d'ombre.

        evaporation : Contrôle la quantité d'humidité ajoutée au-dessus des zones d'eau.
            Valeur recommandée : 0.1 à 0.3
            Une valeur plus élevée augmente l'humidité des zones maritimes ou lacustres, ce qui peut entraîner plus de pluie dans les zones proches de l'eau.
        """

        self.width = width
        self.height = height
        self.scale_octave = scale_octave
        self.amplitude_func = amplitude_func #"Linear" / "Pow"
        self.octaves = octaves
        self.voronoi_scale = voronoi_scale #Permet de créer un diagrame voronoi avec width*height/voronoi_scale points**2
        self.voronoi_size = int(width*height/voronoi_scale**2)
        self.nrivers = max(int(0.1*width),int(0.1*height))

        self.water_sea_level = water_sea_level
        self.water_deepsea_level = min(0.1,water_sea_level)
        self.mountaigns_level = mountaigns_level

        self.altitude_transform_mixing = altitude_transform_mixing
        self.distribution_exponent = distribution_exponent
        self.altitude_transform_exponent = altitude_transform_exponent

        if isinstance(show_progress_bar,bool):
            self.show_progress_bar = show_progress_bar
        else :
            self.show_progress_bar = (width*height>1E6)

        self.raininess = raininess
        self.rain_shadow = rain_shadow
        self.evaporation = evaporation

        self.num_angles_winds = num_angles_winds
        self.angle_variation_winds = angle_variation_winds

        self.water_flow_threshold = water_flow_threshold




class MapGenerator:
    """Classe pour générer et gérer les cartes."""

    def __init__(self, parameters: MapParameters):
        self.parameters = parameters
        self.altitude = None  # Carte générée

        self.voronoi_moisture = None

    def generate_altitude_map(self,generator="Simplex"):
        """Génère une carte d'altitude en utilisant le bruit Perlin."""
        params = self.parameters
        altitude = np.zeros((params.height, params.width))
        moisture_var = np.zeros((params.height, params.width))

        noise_func = PerlinNoise(octaves=params.scale_octave)
        noise_func_moisture = PerlinNoise(octaves=params.scale_octave)
        if generator == "Simplex" :
            SimplexNoiseSeed()
            noise_func = lambda XY : SimplexNoise(XY[0]*params.scale_octave,XY[1]*params.scale_octave)
            SimplexNoiseSeed()
            noise_func_moisture = lambda XY : SimplexNoise(XY[0]*params.scale_octave,XY[1]*params.scale_octave)

        range_iter = tqdm(range(params.height), desc="Génération de la carte d'altitude",
                          unit="ligne") if params.show_progress_bar else range(params.height)
        for y in range_iter:
            for x in range(params.width):
                alt_xy = 0
                m_xy = 0
                W = 0
                for i in range(params.octaves):
                    k = 2**i
                    ks = 2 ** i
                    if params.amplitude_func == "Linear" :
                        ks = i+1


                    e = 1/ks * noise_func([x/params.width*k,
                                          y/params.height*k])

                    m = 1/ks * noise_func_moisture([x/params.width*k,
                                          y/params.height*k])
                    W += 1/ks
                    alt_xy += e
                    m_xy += m

                altitude[y][x] = alt_xy / W
                moisture_var[y][x] = m_xy / W


        # Normalisation
        self.altitude = (altitude - np.min(altitude)) / (np.max(altitude) - np.min(altitude))
        self.moisture_var = (moisture_var - np.min(moisture_var)) / (np.max(moisture_var) - np.min(moisture_var))
    def build_voronoi_grid(self):
        """Construit une grille de Voronoï."""
        params = self.parameters

        dx = 1.1

        vsize = params.voronoi_size
        hsize = int(np.sqrt(vsize))
        X,Y = np.meshgrid(np.linspace(0,params.width,hsize),
                          np.linspace(0,params.height,hsize))

        XY = np.array([X.flatten(),Y.flatten()]).T
        vsize = len(XY)
        points = XY + np.random.rand(vsize, 2) * np.array([params.width/(dx*hsize), params.height/(dx*hsize)])
        points[:, 0] = np.clip(points[:, 0], 0, params.width)
        points[:, 1] = np.clip(points[:, 1], 0, params.height)
        # points = np.append(points,
        #                    [[-10,-10],
        #                            [params.width+10,-10],
        #                            [params.width+10,params.height+10],
        #                            [-10,params.height+10]],
        #                    axis=0)
        vor = Voronoi(points)

        # Création d'une carte à partir des régions
        voronoi_map = np.zeros((params.height, params.width),dtype=int)
        # for y in range(params.height):
        #     for x in range(params.width):
        #         distances = np.linalg.norm(points - np.array([x, y]), axis=1)
        #         voronoi_map[y, x] = np.argmin(distances)
        kd_tree = KDTree(points)
        range_iter = tqdm(range(params.height), desc="Initialisation du mapping",
                          unit="ligne") if params.show_progress_bar else range(params.height)
        for y in range_iter:
            for x in range(params.width) :
                _,idx = kd_tree.query([x,y])
                voronoi_map[y,x] = idx

        self.voronoi_grid = voronoi_map   # Normalisation
        self.voronoi_obj = vor

        self.frontier_cells = np.unique(voronoi_map[0].tolist() + \
                              voronoi_map[params.height-1].tolist() +\
                              voronoi_map[:,0].tolist() +\
                              voronoi_map[:,params.width-1].tolist())

        cell_voronoi = np.array(self.voronoi_obj.ridge_points)
        # self.voronoi_next_cells = []
        # range_iter = tqdm(range(self.voronoi_obj.npoints), desc="Création des connexions de cellules",
        #                   unit="ligne") if params.show_progress_bar else range(self.voronoi_obj.npoints)
        # for cp in range_iter :
        #     connected = [ci for ci in cell_voronoi[cell_voronoi[:, 0] == cp, 1] ] + \
        #                 [ci for ci in cell_voronoi[cell_voronoi[:, 1] == cp, 0] ]
        #     self.voronoi_next_cells.append(connected)
        connections = [[] for i in range(self.voronoi_obj.npoints)]

        range_iter = tqdm(cell_voronoi,
                          desc="Création des connexions de cellules",
                          unit="arête",
                          disable=not params.show_progress_bar)
        for p1,p2 in range_iter :
            connections[p1].append(p2)
            connections[p2].append(p1)
        self.voronoi_next_cells = connections





    def build_voronoi_mapping(self):
        """Calcule la valeur moyenne de l'altitude pour chaque cellule Voronoï."""
        if self.voronoi_grid is None or self.altitude is None:
            raise ValueError("Les cartes Voronoï ou d'altitude doivent être générées avant de calculer les moyennes.")

        # Identifiez les cellules uniques du Voronoï
        unique_cells = np.array(self.voronoi_obj.points)
        params = self.parameters
        # Stocker les moyennes d'altitude
        mean_altitudes = np.zeros(self.voronoi_obj.npoints)
        mean_moisture_var = np.zeros(self.voronoi_obj.npoints)
        points = self.voronoi_obj.points

        range_iter = tqdm(range(self.voronoi_obj.npoints), desc="Mapping du Voronoi",
                          unit="ligne") if self.parameters.show_progress_bar else range(self.voronoi_obj.npoints)
        for i in range_iter:
            # Masque pour les pixels de la cellule actuelle
            mask = (self.voronoi_grid == i)
            if mask.any() :
                # Moyenne des altitudes dans cette cellule
                mean_altitudes[i] = np.mean(self.altitude[mask])
                mean_moisture_var[i] = np.mean(self.moisture_var[mask])
            else :
                x,y = np.clip(int(points[i][0]),0,params.width-1),np.clip(int(points[i][1]),0,params.height-1)
                mean_altitudes[i] = self.altitude[y,x]
                mean_moisture_var[i] = self.moisture_var[y,x]

        self.__voronoi_altitude = mean_altitudes
        self.voronoi_altitude = mean_altitudes
        self.voronoi_moisture_var = (mean_moisture_var - mean_moisture_var.min())/(mean_moisture_var.max() - mean_moisture_var.min())
        self.voronoi_moisture_var = (self.voronoi_moisture_var-0.5)*0.3


    def transform_elevation(self, shaping="Euclidean2"):
        funct = {"SquareBump" : lambda x,y : 1-(1-x**2)*(1-y**2),
                 "Euclidean2" : lambda x,y : np.minimum(1,(x**2+y**2)/np.sqrt(2)),
                 "Coast": self.angled_coast,
                 "Estuary": self.estuary,
                }

        if shaping not in funct :
            raise ValueError(f"La fonction {shaping} n'est pas disponible dans la liste {list(funct.keys())}")

        params = self.parameters
        X = 2*self.voronoi_obj.points[:,0]/params.width-1
        Y = 2*self.voronoi_obj.points[:,1]/params.height-1
        altitude_transform = (1 - funct[shaping](X, Y))**params.altitude_transform_exponent
        self.voronoi_altitude = self.__voronoi_altitude * (1-params.altitude_transform_mixing) + \
                                altitude_transform * params.altitude_transform_mixing
        self.voronoi_altitude = (self.voronoi_altitude-self.voronoi_altitude.min())/ \
                                (self.voronoi_altitude.max()-self.voronoi_altitude.min())
        self.voronoi_altitude = self.voronoi_altitude**self.parameters.distribution_exponent

    # Nouvelle fonction pour générer une côte avec un angle aléatoire
    def angled_coast(self, x, y):
        angle = np.random.uniform(low=0, high=2 * np.pi)  # Angle aléatoire en radians
        a = np.cos(angle)  # Coefficient pour x
        b = np.sin(angle)  # Coefficient pour y
        return np.clip(a * x + b * y,0,1)**1.2  # Distance à la ligne droite définie par l'angle

    # Nouvelle fonction pour simuler un estuaire
    def estuary(self, x, y):
        # Axe principal de l'estuaire (direction horizontale ou inclinée)
        river_angle = np.pi / 4  # Par défaut, un angle de 45° (modifiable)
        river_width = 0.2  # Largeur de l'estuaire
        a = np.cos(river_angle)
        b = np.sin(river_angle)

        # Simuler un dégradé autour de la rivière
        river_distance = np.abs(a * x + b * y)  # Distance au centre de la rivière
        estuary_effect = np.maximum(0, 1 - river_distance / river_width)  # Largeur contrôlée

        # Ajouter un effet de bord pour un estuaire qui se rétrécit vers l'intérieur
        tapering = 1 - (1 + x) / 2  # Dépend de la position horizontale

        return (estuary_effect * tapering)  # Combine les effets



    def fillup_depression(self,printfull=False):
        """
        Corrige les dépressions dans un diagramme de Voronoï.
        """

        voronoi_alt = self.voronoi_altitude
        voronoi_next_cells = self.voronoi_next_cells

        stop_condition = False
        exclude_index = set(np.where(voronoi_alt<= self.parameters.water_deepsea_level)[0])
        exclude_index.update(self.frontier_cells)
        points_to_inspect = np.array(
                    list(set(range(self.voronoi_obj.npoints)).\
                                difference(exclude_index))
                                )
        iter = 0

        while not stop_condition :
            iter += 1

            min_alt_next_cell = np.array(
                [min(voronoi_alt[voronoi_next_cells[i]]) for i in points_to_inspect])

            to_correct = min_alt_next_cell>voronoi_alt[points_to_inspect]

            corrected_index = points_to_inspect[to_correct]
            if printfull :
                print(f"     >>> iter {iter} - points {len(points_to_inspect)} - corrections {len(corrected_index)}")
            if len(corrected_index) > 0:
                voronoi_alt[corrected_index] = min_alt_next_cell[to_correct]+0.01

                new_points_to_inspect = set()
                for ci in corrected_index :
                    new_points_to_inspect.update(set(voronoi_next_cells[ci]).difference(exclude_index) )
                    new_points_to_inspect.add(ci)

                points_to_inspect = np.array(list(new_points_to_inspect))

            else :
                stop_condition = True

            # points_to_inspect = points_to_inspect[np.argsort(voronoi_alt[points_to_inspect])]
            # correct =0
            # for i in points_to_inspect:
            #     if i in self.frontier_cells:
            #         continue
            #     if voronoi_alt[i] <= self.parameters.water_deepsea_level:
            #         continue
            #
            #     alt_next = voronoi_alt[voronoi_next_cells[i]]
            #     alt_min = np.min(alt_next)
            #
            #     if alt_min > voronoi_alt[i]:
            #         correct+=1
            #         voronoi_alt[i] = alt_min + 0.01  # Correction légère
            #         stop_condition = False
            #         new_points_to_inspect.update(voronoi_next_cells[i])
            #         new_points_to_inspect.add(i)
            #
            # print(f"     >>> iter {iter} - points {len(points_to_inspect)} - corrections {correct} - {np.sum(to_correct)}")
            # points_to_inspect = np.array(list(new_points_to_inspect))

        # Normalisation finale
        self.voronoi_altitude = (voronoi_alt - voronoi_alt.min()) / (voronoi_alt.max() - voronoi_alt.min())

    def __calculate_wind_direction(self):
        # Calcul du barycentre des zones d'eau
        sea_cells = np.where(self.voronoi_altitude < self.parameters.water_sea_level)[0]
        if len(sea_cells) == 0:
            raise ValueError("No sea cells found to calculate wind direction.")

        sea_x = self.voronoi_obj.points[sea_cells][:, 0]
        sea_y = self.voronoi_obj.points[sea_cells][:, 1]

        xb, yb = np.mean(sea_x), np.mean(sea_y)

        # Centre de la carte
        xc, yc = self.parameters.width / 2, self.parameters.height / 2

        # Direction du vent (du barycentre des zones d'eau vers le centre)
        wind_vector = np.array([xc - xb, yc - yb])
        wind_vector /= np.linalg.norm(wind_vector)  # Normalisation

        return wind_vector

    def __assignRainfall(self,wind_direction):
        humidity_array = np.zeros(self.voronoi_obj.npoints)
        rainfall_array = np.zeros(self.voronoi_obj.npoints)
        # Calcul des projections des cellules dans la direction du vent
        projections = np.dot(self.voronoi_obj.points, wind_direction)
        wind_order = np.argsort(projections)

        # Parcours des cellules dans l'ordre du vent
        for cell in wind_order:
            neighbors = self.voronoi_next_cells[cell]

            # Moyenne de l'humidité des cellules amont (déjà traversées par le vent)
            humidity_sum = 0
            count = 0
            for neighbor in neighbors:
                if projections[neighbor] < projections[cell]:
                    humidity_sum += humidity_array[neighbor]
                    count += 1

            if count > 0:
                humidity_array[cell] = humidity_sum / count

            # Pluie générée par l'humidité
            rainfall_array[cell] += self.parameters.raininess * humidity_array[cell]

            # Gestion des bords
            if cell in self.frontier_cells:
                humidity_array[cell] = 1.0

            # Évaporation pour les cellules sous le niveau de la mer
            if self.voronoi_altitude[cell] < self.parameters.water_sea_level:
                evaporation = self.parameters.evaporation * (
                        self.parameters.water_sea_level - self.voronoi_altitude[cell])
                humidity_array[cell] += evaporation

            # Précipitations orographiques (effet de barrière des montagnes)
            if humidity_array[cell] > (1.0 - self.voronoi_altitude[cell]):
                orographic_rainfall = self.parameters.rain_shadow * (
                        humidity_array[cell] - (1.0 - self.voronoi_altitude[cell])
                )
                rainfall_array[cell] += self.parameters.raininess * orographic_rainfall
                humidity_array[cell] -= orographic_rainfall
        return rainfall_array


    def build_rainfall(self):
        self.voronoi_rainfall = np.zeros(self.voronoi_obj.npoints,dtype=float)

        base_direction = self.__calculate_wind_direction()
        base_angle_rad = np.arctan2(base_direction[1], base_direction[0])
        weight_sum = 0

        num_angles = self.parameters.num_angles_winds
        angle_variation = self.parameters.angle_variation_winds

        for i in range(num_angles):
            # Calcul d'un nouvel angle avec une petite variation
            variation = np.random.uniform(-angle_variation, angle_variation)
            new_angle_rad = base_angle_rad + np.radians(variation)
            wind_direction = np.array([np.cos(new_angle_rad), np.sin(new_angle_rad)])

            weight = 1.0 / (1.0 + np.abs(variation))
            self.voronoi_rainfall += self.__assignRainfall(wind_direction)*weight
            weight_sum += weight

        self.voronoi_rainfall = self.voronoi_rainfall / weight_sum

    def build_rivers_from_rainfall(self):
        """
        Construit les rivières en suivant les flux d'eau à travers le diagramme Voronoi.

        Arguments :
        - flux_threshold : seuil de flux pour considérer une cellule comme une source de rivière.
        """

        flux_array = self.voronoi_rainfall.copy()  # Initialiser le flux avec les précipitations

        # Tri des cellules par ordre décroissant d'altitude
        sorted_cells = np.argsort(-self.voronoi_altitude)

        for cell in sorted_cells:
            neighbors = self.voronoi_next_cells[cell]
            min_neighbor = neighbors[np.argmin(self.voronoi_altitude[neighbors])]
            flux_array[min_neighbor] += flux_array[cell]

        self.voronoi_drainage_flow = (flux_array - flux_array.min()) / (flux_array.max() - flux_array.min())
        self.__build_river_paths()

    def __build_river_paths(self):
        """
        Suivre les trajectoires des rivières depuis leurs sources jusqu'à un point terminal
        (océan, lac, ou autre rivière).
        """
        flux_threshold = self.parameters.water_flow_threshold
        river_paths = []  # Liste pour stocker les chemins des rivières
        visited = np.zeros(self.voronoi_obj.npoints, dtype=bool)  # Marqueur des cellules déjà visitées

        # Identifier les sources de rivières
        sources = np.where(self.voronoi_drainage_flow > flux_threshold)[0]

        for source in sources:
            if visited[source]:
                continue  # Si la source a déjà été visitée, on passe
            if self.voronoi_altitude[source] <= self.parameters.water_sea_level :
                continue

            path = []  # Chemin de la rivière actuelle
            current_cell = source

            while True:
                path.append(current_cell)
                visited[current_cell] = True

                # Identifier les voisins plus bas
                neighbors = self.voronoi_next_cells[current_cell]
                lower_neighbors = [
                    neighbor for neighbor in neighbors
                    if (self.voronoi_altitude[neighbor] < self.voronoi_altitude[current_cell])
                ]

                # Si aucun voisin plus bas, arrêter la rivière
                if not lower_neighbors:
                    break

                # Trouver le voisin avec la plus basse altitude
                next_cell = min(lower_neighbors, key=lambda n: self.voronoi_altitude[n])

                # Vérifier si le voisin est de l'eau ou déjà une rivière
                if self.voronoi_altitude[next_cell] < self.parameters.water_sea_level or \
                        visited[next_cell] or next_cell in self.frontier_cells:
                    path.append(next_cell)
                    break

                current_cell = next_cell

            # Ajouter la rivière si elle a un chemin valide
            if len(path) > 1:
                river_paths.append(path)

        self.river_paths = river_paths
        rivers_edges = []
        rivers_edges_flows = []
        for path in river_paths :
            rivers_edges += [(path[i-1],path[i]) for i in range(1,len(path))]
            rivers_edges_flows += [self.voronoi_drainage_flow[path[i - 1]] for i in range(1, len(path))]
        rivers_edges = np.sort(np.array(rivers_edges), axis=1).tolist()
        self.rivers_edges = []
        self.rivers_edges_flows = []
        for i,e in enumerate(rivers_edges):
            if e in self.rivers_edges :
                j = self.rivers_edges.index(e)
                self.rivers_edges_flows[j] += rivers_edges_flows[i]
            else :
                self.rivers_edges.append(e)
                self.rivers_edges_flows.append(rivers_edges_flows[i])
        self.rivers_edges = np.array(self.rivers_edges,dtype=int)
        self.rivers_edges_flows = np.array(self.rivers_edges_flows)
        self.rivers_edges_flows = (self.rivers_edges_flows-self.rivers_edges_flows.min())/\
                                  (self.rivers_edges_flows.max()-self.rivers_edges_flows.min())


        self.water_cells = np.unique(
            np.append(np.unique(rivers_edges),
            np.where(self.voronoi_altitude<self.parameters.water_sea_level)[0]))




    def build_rivers_from_elevation(self,keep_retention=False):

        mountaigns = np.where(self.voronoi_altitude>=self.parameters.mountaigns_level)[0]
        lakes = np.where(self.voronoi_altitude<=self.parameters.water_deepsea_level)[0]
        if self.parameters.nrivers is None :
            nrivers_m = len(mountaigns)
            nrivers_l = len(lakes)
            limited = True
        else :
            nrivers_m = min(self.parameters.nrivers, len(mountaigns))
            nrivers_l = min(self.parameters.nrivers, len(lakes))
            limited = False
        start_rivers = np.random.choice(
                                mountaigns,
                                size=nrivers_m,
                                replace=False )
        end_rivers = np.random.choice(lakes,
                                      size=nrivers_l,
                                      replace=False)
        cell_voronoi = np.array(self.voronoi_obj.ridge_points)



        rivers_edges =  []

        for cp in start_rivers :
            explored = []
            stop_condition = False
            new_rivers_edges = []
            retention = False
            while not stop_condition :
                connected = [ci for ci in cell_voronoi[cell_voronoi[:, 0] == cp, 1] if ci not in explored] + \
                            [ci for ci in cell_voronoi[cell_voronoi[:, 1] == cp, 0] if ci not in explored]

                # connected = [ci for ci in connected if ci < len(self.voronoi_altitude)]
                explored += [cp]

                if len(connected) >0 :
                    alt_k = self.voronoi_altitude[connected]
                    min_alt_k = alt_k.min()
                    ck = connected[np.argmin(alt_k)]


                    new_rivers_edges.append([cp,ck])
                    cp = ck
                    if ck in lakes or ck in self.frontier_cells:
                        stop_condition = True


                else :
                    retention = True
                    stop_condition = True
            if ( retention and keep_retention ) or not(retention) :
                rivers_edges += new_rivers_edges
        for cp in end_rivers :
            explored = []
            stop_condition = False
            while not stop_condition:
                connected = [ci for ci in cell_voronoi[cell_voronoi[:, 0] == cp, 1] if ci not in explored] + \
                            [ci for ci in cell_voronoi[cell_voronoi[:, 1] == cp, 0] if ci not in explored]

                # connected = [ci for ci in connected if ci < len(self.voronoi_altitude)]
                explored += [cp]

                if len(connected) > 0:
                    alt_k = self.voronoi_altitude[connected]
                    ck = connected[np.argmax(alt_k)]

                    rivers_edges.append([cp, ck])
                    cp = ck
                    if ck in mountaigns or ck in self.frontier_cells:
                        stop_condition = True
                else:
                    stop_condition = True
        rivers_edges = np.sort(np.array(rivers_edges), axis=1)

        if limited :

            edges,count = np.unique(rivers_edges,return_counts=True, axis=0)
            cm = count.mean()*0.5
            count_filter = count>=cm
            edges = edges[count_filter]
            count = count[count_filter]
            rivers_edges = []
            for e,c in zip(edges,count) :
                rivers_edges += [e]*( int(c - cm) +1)
            rivers_edges = np.array(rivers_edges)
        self.rivers_edges,self.rivers_edges_flows = np.unique(rivers_edges,return_counts=True,axis=0)
        self.rivers_edges_flows = (self.rivers_edges_flows - self.rivers_edges_flows.min()) / \
                                  (self.rivers_edges_flows.max() - self.rivers_edges_flows.min())

        self.__build_water_cells()


    def __build_water_cells(self):
        rivers_edges = self.rivers_edges
        lakes = self.voronoi_altitude <= self.parameters.water_sea_level
        cell_voronoi = self.voronoi_next_cells
        self.water_cells = np.unique(rivers_edges).tolist()

        lake_water_cells = [c for c in self.water_cells if c in lakes]
        cell_to_check = lake_water_cells.copy()
        while len(cell_to_check) > 1:
            ck = cell_to_check.pop()

            connected = [ci for ci in cell_voronoi[ck] if ci in lakes]

            connected = np.unique([ci for ci in connected if (ci not in cell_to_check) and \
                                   (ci not in lake_water_cells) and \
                                   (ci not in self.water_cells)]).tolist()

            lake_water_cells.append(ck)
            cell_to_check += connected
        self.water_cells = np.unique(self.water_cells + lake_water_cells)


    def build_moisture_from_rain(self):
        river_moisture = self.__build_moisture_water_cell(0.85)
        self.voronoi_moisture = np.clip(self.voronoi_rainfall + river_moisture,
                                        0,
                                        1)


    def __build_moisture_water_cell(self,krate=0.95):
        voronoi_moisture = np.zeros_like(self.voronoi_altitude, dtype=float)
        points = np.array(self.voronoi_obj.points)
        water_points = points[self.water_cells]
        range_iter = tqdm(enumerate(points), desc="Calcul de l'humidité",
                          unit="ligne") if self.parameters.show_progress_bar else enumerate(points)
        for i, pi in range_iter:
            dist = np.linalg.norm(water_points - pi, axis=1).min()
            mi = krate ** dist
            voronoi_moisture[i] = mi
        return voronoi_moisture

    def build_moisture_from_rivers(self):

        voronoi_moisture = self.__build_moisture_water_cell(0.95)

        rdn_filter = voronoi_moisture<1
        voronoi_moisture[rdn_filter] += self.voronoi_moisture_var[rdn_filter]
        self.voronoi_moisture = np.clip(voronoi_moisture,0,1)


    def build_biomes(self):
        voronoi_biomes = np.full(self.voronoi_obj.npoints,fill_value="",dtype=np.object_)

        moisture = self.voronoi_moisture
        altitude = self.voronoi_altitude
        voronoi_biomes[moisture>=0] = "Eau"

        voronoi_biomes[(altitude<1.01)&(altitude>=0.8)&(moisture<=1)&(moisture>=0.8)] = 'Neige'
        voronoi_biomes[(altitude<0.8)&(altitude>=0.6)&(moisture<1)&(moisture>=0.8)] = 'Forêt alpine'
        voronoi_biomes[(altitude<0.6)&(altitude>=0.4)&(moisture<1)&(moisture>=0.8)] = 'Forêt dense'
        voronoi_biomes[(altitude<0.4)&(altitude>=0)&(moisture<1)&(moisture>=0.8)] = 'Zone humide'
        voronoi_biomes[(altitude<1.01)&(altitude>=0.8)&(moisture<0.8)&(moisture>=0.6)] = 'Neige'
        voronoi_biomes[(altitude<0.8)&(altitude>=0.6)&(moisture<0.8)&(moisture>=0.6)] = 'Forêt alpine'
        voronoi_biomes[(altitude<0.6)&(altitude>=0.4)&(moisture<0.8)&(moisture>=0.6)] = 'Forêt dense'
        voronoi_biomes[(altitude<0.4)&(altitude>=0)&(moisture<0.8)&(moisture>=0.6)] = 'Marais boisé'
        voronoi_biomes[(altitude<1.01)&(altitude>=0.8)&(moisture<0.6)&(moisture>=0.4)] = 'Neige'
        voronoi_biomes[(altitude<0.8)&(altitude>=0.6)&(moisture<0.6)&(moisture>=0.4)] = 'Forêt clairsemée'
        voronoi_biomes[(altitude<0.6)&(altitude>=0.4)&(moisture<0.6)&(moisture>=0.4)] = 'Forêt clairsemée'
        voronoi_biomes[(altitude<0.4)&(altitude>=0)&(moisture<0.6)&(moisture>=0.4)] = 'Forêt clairsemée'
        voronoi_biomes[(altitude<1.01)&(altitude>=0.8)&(moisture<0.2)&(moisture>=0.1)] = 'Rocaille'
        voronoi_biomes[(altitude<0.8)&(altitude>=0.6)&(moisture<0.2)&(moisture>=0.1)] = 'Brousaille'
        voronoi_biomes[(altitude<0.6)&(altitude>=0.4)&(moisture<0.2)&(moisture>=0.1)] = 'Prairie fertile'
        voronoi_biomes[(altitude<0.4)&(altitude>=0)&(moisture<0.2)&(moisture>=0.1)] = 'Prairie fertile'
        voronoi_biomes[(altitude<1.01)&(altitude>=0.8)&(moisture<0.1)&(moisture>=0)] = 'Sommets arides'
        voronoi_biomes[(altitude<0.8)&(altitude>=0.6)&(moisture<0.1)&(moisture>=0)] = 'Rocaille'
        voronoi_biomes[(altitude<0.6)&(altitude>=0.4)&(moisture<0.1)&(moisture>=0)] = 'Brousaille'
        voronoi_biomes[(altitude<0.4)&(altitude>=0)&(moisture<0.1)&(moisture>=0)] = 'Plaines sèches'
        voronoi_biomes[(altitude<1.01)&(altitude>=0.8)&(moisture<0.4)&(moisture>=0.2)] = 'Thoundra'
        voronoi_biomes[(altitude<0.8)&(altitude>=0.6)&(moisture<0.4)&(moisture>=0.2)] = 'Forêt clairsemée'
        voronoi_biomes[(altitude<0.6)&(altitude>=0.4)&(moisture<0.4)&(moisture>=0.2)] = 'Prairie fertile'
        voronoi_biomes[(altitude<0.4)&(altitude>=0)&(moisture<0.4)&(moisture>=0.2)] = 'Forêt clairsemée'



        self.voronoi_biomes = voronoi_biomes
        self.biomes_color = {
            'Eau': '#004080',          # Bleu marine profond, évoquant une grande profondeur
            'Neige': '#FFFFFF',        # Blanc pur et brillant pour un effet lumineux
            'Thoundra': '#D8D8BF',     # Beige clair avec une teinte froide pour un terrain aride et gelé
            'Rocaille': '#8B8B8B',     # Gris roche plus saturé pour un aspect minéral
            'Sommets arides': '#6E6E6E', # Gris sombre, presque anthracite, pour les reliefs arides
            'Forêt alpine': '#98C195', # Vert frais et légèrement désaturé, reflétant l'altitude
            'Forêt clairsemée': '#B5D19C', # Vert tendre, lumineux, pour une forêt clairsemée
            'Brousaille': '#C8D48B',   # Jaune-vert doux pour un terrain avec une végétation éparse
            'Forêt dense': '#6FA36E',  # Vert intense et profond, pour une forêt riche
            'Prairie fertile': '#89B87F', # Vert naturel et vibrant, légèrement chaud
            'Zone humide': '#36A8D6',     # Vert foncé et bleuté, évoquant l'humidité stagnante
            'Marais boisé': '#7BB391', # Vert vif avec une touche de fraîcheur pour les marais boisés
            'Plaines sèches': '#E8D3B0', # Beige chaud et doux, reflétant des terrains semi-arides
        }





    def show_voronoi(self,ax=None,**kwargs):
        voronoi_plot_2d(self.voronoi_obj,ax=ax,**kwargs)
        ax.set(xlim=[0,self.parameters.width],
               ylim=[0,self.parameters.height])


    def show_map(self,ax=None):
        """Affiche la carte d'altitude générée."""
        if self.altitude is None:
            raise ValueError("La carte d'altitude n'a pas été générée. Appelez `generate_altitude_map()` d'abord.")

        if ax is None :
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.imshow(self.altitude, cmap='terrain')

        return ax

    def show_voronoi_altitude(self, ax=None):
        """Affiche la carte Voronoï avec les altitudes moyennes."""
        if self.voronoi_grid is None or self.altitude is None:
            raise ValueError("Les cartes Voronoï ou d'altitude doivent être générées avant l'affichage.")

        # Calcul des altitudes moyennes
        unique_cells =list(range(self.voronoi_obj.npoints))
        mean_altitudes = self.voronoi_altitude

        # Créez une carte couleur pour afficher les altitudes moyennes
        voronoi_altitude_map = np.zeros_like(self.voronoi_grid, dtype=float)

        for i, cell_id in enumerate(unique_cells):
            voronoi_altitude_map[self.voronoi_grid == cell_id] = mean_altitudes[i]

        # Affichage
        if ax is None:
            fig, ax = plt.subplots()
        cax = ax.imshow(voronoi_altitude_map, cmap='terrain')
        return ax

    def show_river(self, ax=None, color="b"):
        if ax is None:
            fig, ax = plt.subplots()
        pts = np.array(self.voronoi_obj.points)
        lw_min = 0.5
        lw_max = 3.0

        for (edge,flow) in zip(self.rivers_edges,self.rivers_edges_flows ) :
            lw = flow*(lw_max-lw_min) + lw_min
            ax.plot(pts[edge,0],pts[edge,1],color=color,lw = lw)


    def show_moisture_map(self, ax=None):
        """Affiche la carte Voronoï avec les altitudes moyennes."""
        if self.voronoi_grid is None or self.voronoi_moisture is None:
            raise ValueError("Les cartes Voronoï ou d'altitude doivent être générées avant l'affichage.")
        # Affichage
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(self.voronoi_obj.points[:,0],self.voronoi_obj.points[:,1],c=1-self.voronoi_moisture)
        return ax

    def show_rainfall(self, ax=None):
        """Affiche la carte Voronoï avec les altitudes moyennes."""
        if self.voronoi_grid is None or self.voronoi_rainfall is None:
            raise ValueError("Les cartes Voronoï de pluie ne sont pas générées.")
        # Affichage
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(self.voronoi_obj.points[:,0],self.voronoi_obj.points[:,1], c=1-self.voronoi_rainfall)
        return ax


    def show_drainage(self, ax=None):
        """Affiche la carte Voronoï avec les altitudes moyennes."""
        if self.voronoi_grid is None or self.voronoi_rainfall is None:
            raise ValueError("Les cartes Voronoï de drainage ne sont pas générées.")
        # Affichage
        if ax is None:
            fig, ax = plt.subplots()
        sc = ax.scatter(self.voronoi_obj.points[:,0],self.voronoi_obj.points[:,1], c=self.voronoi_drainage_flow)
        plt.colorbar(sc)
        self.show_voronoi_altitude(ax)
        return ax

    def show_biomes(self,ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        for i, region_index in enumerate(self.voronoi_obj.point_region):  # Associe chaque point à sa région
            # if region_index not in self.frontier_cells :
            region = self.voronoi_obj.regions[region_index]

            # Vérifie si la région est valide (ne doit pas contenir -1)
            if not -1 in region and region:
                polygon = [self.voronoi_obj.vertices[v] for v in region]
                biome = self.voronoi_biomes[i]  # Biome associé à la cellule
                color = self.biomes_color[biome]  # Couleur associée au biome

                # Trace le polygone avec la couleur du biome
                ax.fill(*zip(*polygon), color=color)
        ax.set(xlim=[1, self.parameters.width-1],
               ylim=[1, self.parameters.height-1])

# Exemple d'utilisation
if __name__ == "__main__":
    # Définir les paramètres
    params = MapParameters(width=320,
                           height=640,
                           scale_octave=5.0,
                           amplitude_func="Pow",
                           octaves=8,
                           voronoi_scale=2,
                           altitude_transform_mixing=0.5,
                           altitude_transform_exponent=0.8,
                           distribution_exponent=2.2,
                           show_progress_bar=True,
                           raininess=0.5,
                           rain_shadow=0.2,
                           evaporation=0.2,
                           angle_variation_winds=30.0,
                           num_angles_winds=15,
                           water_flow_threshold=0.01,
                           )

    # Générer la carte
    generator = MapGenerator(parameters=params)
    print("Génération de la carte ...")
    generator.generate_altitude_map(generator="Simplex")
    print("Création du diagramme Voronoi ...")
    generator.build_voronoi_grid()
    print("Calcul de l'altitude sur les cellules Voronoi ...")
    generator.build_voronoi_mapping()
    print("Transformation altitude ...")
    generator.transform_elevation("Coast")
    print("Correction des dépressions ...")
    generator.fillup_depression()

    print("Calcul des précipitations ...")
    generator.build_rainfall()
    print("Calcul des drainages ...")
    generator.build_rivers_from_rainfall()
    print("Calcul de l'humidité ... ")
    generator.build_moisture_from_rain()
    print("Calcul des biomes ...")
    generator.build_biomes()



    print("AFFICHAGE ...")

    figsize = (max(5,min(9,params.width*5/250)),
               max(5,min(9,params.width*5/250)))
    figure = plt.figure(figsize=figsize)
    ax = figure.add_subplot(111)
    figure.tight_layout()
    generator.show_voronoi_altitude(ax)
    generator.show_voronoi(ax,
                           show_points=False,
                           show_vertices=False,
                           line_colors="grey",
                           line_alpha=0.5,
                           line_width=0.5,)
    generator.show_river(ax)

    figure = plt.figure(figsize=figsize)
    ax = figure.add_subplot(111)
    figure.tight_layout()
    generator.show_moisture_map(ax)
    generator.show_voronoi(ax,
                           show_points=False,
                           show_vertices=False,
                           line_colors="grey",
                           line_alpha=0.5,
                           line_width=0.5, )

    figure = plt.figure(figsize=figsize)
    ax = figure.add_subplot(111)
    figure.tight_layout()
    generator.show_biomes(ax)
    plt.show()

