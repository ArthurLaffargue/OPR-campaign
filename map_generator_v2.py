import numpy as np
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
from opensimplex import noise2 as SimplexNoise
from opensimplex import random_seed as SimplexNoiseSeed
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
from tqdm import tqdm  # Import de tqdm pour la barre de progression

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




# ============================================================================
# 1. TERRAIN GENERATOR - Responsabilité : Générer la topographie de base
# ============================================================================

class TerrainGenerator:
    """
    Responsabilité unique : Génération du terrain de base
    - Bruit de Perlin/Simplex
    - Diagrammes de Voronoï
    - Transformation d'élévation
    - Correction des dépressions
    """

    def __init__(self, parameters: MapParameters):
        self.parameters = parameters
        self.voronoi_obj = None
        self.voronoi_grid = None
        self.altitude = None
        self.voronoi_altitude = None

    def generate_base_terrain(self, generator="Simplex"):
        """Génère le terrain de base avec bruit"""
        # Votre code actuel de generate_altitude_map

        """Génère une carte d'altitude en utilisant le bruit Perlin."""
        params = self.parameters
        altitude = np.zeros((params.height, params.width))
        moisture_var = np.zeros((params.height, params.width))

        noise_func = PerlinNoise(octaves=params.scale_octave)
        noise_func_moisture = PerlinNoise(octaves=params.scale_octave)
        if generator == "Simplex":
            SimplexNoiseSeed()
            noise_func = lambda XY: SimplexNoise(XY[0] * params.scale_octave, XY[1] * params.scale_octave)
            SimplexNoiseSeed()
            noise_func_moisture = lambda XY: SimplexNoise(XY[0] * params.scale_octave, XY[1] * params.scale_octave)

        range_iter = tqdm(range(params.height), desc="Génération de la carte d'altitude",
                          unit="ligne") if params.show_progress_bar else range(params.height)
        for y in range_iter:
            for x in range(params.width):
                alt_xy = 0
                m_xy = 0
                W = 0
                for i in range(params.octaves):
                    k = 2 ** i
                    ks = 2 ** i
                    if params.amplitude_func == "Linear":
                        ks = i + 1

                    e = 1 / ks * noise_func([x / params.width * k,
                                             y / params.height * k])

                    m = 1 / ks * noise_func_moisture([x / params.width * k,
                                                      y / params.height * k])
                    W += 1 / ks
                    alt_xy += e
                    m_xy += m

                altitude[y][x] = alt_xy / W
                moisture_var[y][x] = m_xy / W

        # Normalisation
        self.altitude = (altitude - np.min(altitude)) / (np.max(altitude) - np.min(altitude))
        self.moisture_var = (moisture_var - np.min(moisture_var)) / (np.max(moisture_var) - np.min(moisture_var))



    def __build_voronoi_grid(self):
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





    def __build_voronoi_mapping(self):
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

    def build_voronoi_structure(self):
        """Construit la structure Voronoï"""
        # Votre code actuel de build_voronoi_grid + build_voronoi_mapping
        self.__build_voronoi_grid()
        self.__build_voronoi_mapping()

    def apply_elevation_transform(self, shaping="Coast"):
        """Applique les transformations d'élévation"""
        # Votre code actuel de transform_elevation
        funct = {"SquareBump": lambda x, y: 1 - (1 - x ** 2) * (1 - y ** 2),
                 "Euclidean2": lambda x, y: np.minimum(1, (x ** 2 + y ** 2) / np.sqrt(2)),
                 "Coast": self.__angled_coast,
                 "Estuary": self.__estuary,
                 }

        if shaping not in funct:
            raise ValueError(f"La fonction {shaping} n'est pas disponible dans la liste {list(funct.keys())}")

        params = self.parameters
        X = 2 * self.voronoi_obj.points[:, 0] / params.width - 1
        Y = 2 * self.voronoi_obj.points[:, 1] / params.height - 1
        altitude_transform = (1 - funct[shaping](X, Y)) ** params.altitude_transform_exponent
        self.voronoi_altitude = self.__voronoi_altitude * (1 - params.altitude_transform_mixing) + \
                                altitude_transform * params.altitude_transform_mixing
        self.voronoi_altitude = (self.voronoi_altitude - self.voronoi_altitude.min()) / \
                                (self.voronoi_altitude.max() - self.voronoi_altitude.min())
        self.voronoi_altitude = self.voronoi_altitude ** self.parameters.distribution_exponent

        # Nouvelle fonction pour générer une côte avec un angle aléatoire

    def __angled_coast(self, x, y):
        angle = np.random.uniform(low=0, high=2 * np.pi)  # Angle aléatoire en radians
        a = np.cos(angle)  # Coefficient pour x
        b = np.sin(angle)  # Coefficient pour y
        return np.clip(a * x + b * y, 0, 1) ** 1.2  # Distance à la ligne droite définie par l'angle

        # Nouvelle fonction pour simuler un estuaire

    def __estuary(self, x, y):
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

    def fix_depressions(self,printfull=False):
        """Corrige les dépressions topographiques"""
        # Votre code actuel de fillup_depression
        voronoi_alt = self.voronoi_altitude
        voronoi_next_cells = self.voronoi_next_cells

        stop_condition = False
        exclude_index = set(np.where(voronoi_alt <= self.parameters.water_deepsea_level)[0])
        exclude_index.update(self.frontier_cells)
        points_to_inspect = np.array(
            list(set(range(self.voronoi_obj.npoints)). \
                 difference(exclude_index))
        )
        iter = 0

        while not stop_condition:
            iter += 1

            min_alt_next_cell = np.array(
                [min(voronoi_alt[voronoi_next_cells[i]]) for i in points_to_inspect])

            to_correct = min_alt_next_cell > voronoi_alt[points_to_inspect]

            corrected_index = points_to_inspect[to_correct]
            if printfull:
                print(f"     >>> iter {iter} - points {len(points_to_inspect)} - corrections {len(corrected_index)}")
            if len(corrected_index) > 0:
                voronoi_alt[corrected_index] = min_alt_next_cell[to_correct] + 0.01

                new_points_to_inspect = set()
                for ci in corrected_index:
                    new_points_to_inspect.update(set(voronoi_next_cells[ci]).difference(exclude_index))
                    new_points_to_inspect.add(ci)

                points_to_inspect = np.array(list(new_points_to_inspect))

            else:
                stop_condition = True
        # Normalisation finale
        self.voronoi_altitude = (voronoi_alt - voronoi_alt.min()) / (voronoi_alt.max() - voronoi_alt.min())


# ============================================================================
# 2. HYDROGRAPHY GENERATOR - Responsabilité : Tout ce qui concerne l'eau
# ============================================================================

class HydrographyGenerator:
    """
    Responsabilité unique : Systèmes hydrographiques
    - Calcul des vents et précipitations
    - Génération des rivières
    - Calcul de l'humidité
    - Drainage
    """

    def __init__(self, terrain: TerrainGenerator):
        self.terrain = terrain
        self.parameters = terrain.parameters
        self.rainfall = None
        self.rivers = None
        self.moisture = None



    def __calculate_wind_direction(self):
        # Calcul du barycentre des zones d'eau
        sea_cells = np.where(self.terrain.voronoi_altitude < self.parameters.water_sea_level)[0]
        if len(sea_cells) == 0:
            raise ValueError("No sea cells found to calculate wind direction.")

        sea_x = self.terrain.voronoi_obj.points[sea_cells][:, 0]
        sea_y = self.terrain.voronoi_obj.points[sea_cells][:, 1]

        xb, yb = np.mean(sea_x), np.mean(sea_y)

        # Centre de la carte
        xc, yc = self.parameters.width / 2, self.parameters.height / 2

        # Direction du vent (du barycentre des zones d'eau vers le centre)
        wind_vector = np.array([xc - xb, yc - yb])
        wind_vector /= np.linalg.norm(wind_vector)  # Normalisation

        return wind_vector

    def __assignRainfall(self,wind_direction):
        humidity_array = np.zeros(self.terrain.voronoi_obj.npoints)
        rainfall_array = np.zeros(self.terrain.voronoi_obj.npoints)
        # Calcul des projections des cellules dans la direction du vent
        projections = np.dot(self.terrain.voronoi_obj.points, wind_direction)
        wind_order = np.argsort(projections)

        # Parcours des cellules dans l'ordre du vent
        for cell in wind_order:
            neighbors = self.terrain.voronoi_next_cells[cell]

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
            if cell in self.terrain.frontier_cells:
                humidity_array[cell] = 1.0

            # Évaporation pour les cellules sous le niveau de la mer
            if self.terrain.voronoi_altitude[cell] < self.parameters.water_sea_level:
                evaporation = self.parameters.evaporation * (
                        self.parameters.water_sea_level - self.terrain.voronoi_altitude[cell])
                humidity_array[cell] += evaporation

            # Précipitations orographiques (effet de barrière des montagnes)
            if humidity_array[cell] > (1.0 - self.terrain.voronoi_altitude[cell]):
                orographic_rainfall = self.parameters.rain_shadow * (
                        humidity_array[cell] - (1.0 - self.terrain.voronoi_altitude[cell])
                )
                rainfall_array[cell] += self.parameters.raininess * orographic_rainfall
                humidity_array[cell] -= orographic_rainfall
        return rainfall_array

    def calculate_rainfall_patterns(self):
        """Calcule les patterns de précipitations"""
        # Votre code actuel de build_rainfall
        self.voronoi_rainfall = np.zeros(self.terrain.voronoi_obj.npoints, dtype=float)

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
            self.voronoi_rainfall += self.__assignRainfall(wind_direction) * weight
            weight_sum += weight

        self.voronoi_rainfall = self.voronoi_rainfall / weight_sum

    def generate_river_network(self):
        """Génère le réseau de rivières"""
        # Votre code actuel de build_rivers_from_rainfall
        flux_array = self.voronoi_rainfall.copy()  # Initialiser le flux avec les précipitations

        # Tri des cellules par ordre décroissant d'altitude
        sorted_cells = np.argsort(-self.terrain.voronoi_altitude)

        for cell in sorted_cells:
            neighbors = self.terrain.voronoi_next_cells[cell]
            min_neighbor = neighbors[np.argmin(self.terrain.voronoi_altitude[neighbors])]
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
        visited = np.zeros(self.terrain.voronoi_obj.npoints, dtype=bool)  # Marqueur des cellules déjà visitées

        # Identifier les sources de rivières
        sources = np.where(self.voronoi_drainage_flow > flux_threshold)[0]

        for source in sources:
            if visited[source]:
                continue  # Si la source a déjà été visitée, on passe
            if self.terrain.voronoi_altitude[source] <= self.parameters.water_sea_level :
                continue

            path = []  # Chemin de la rivière actuelle
            current_cell = source

            while True:
                path.append(current_cell)
                visited[current_cell] = True

                # Identifier les voisins plus bas
                neighbors = self.terrain.voronoi_next_cells[current_cell]
                lower_neighbors = [
                    neighbor for neighbor in neighbors
                    if (self.terrain.voronoi_altitude[neighbor] < self.terrain.voronoi_altitude[current_cell])
                ]

                # Si aucun voisin plus bas, arrêter la rivière
                if not lower_neighbors:
                    break

                # Trouver le voisin avec la plus basse altitude
                next_cell = min(lower_neighbors, key=lambda n: self.terrain.voronoi_altitude[n])

                # Vérifier si le voisin est de l'eau ou déjà une rivière
                if self.terrain.voronoi_altitude[next_cell] < self.parameters.water_sea_level or \
                        visited[next_cell] or next_cell in self.terrain.frontier_cells:
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
            np.where(self.terrain.voronoi_altitude<self.parameters.water_sea_level)[0]))

    def calculate_moisture_distribution(self):
        """Calcule la distribution d'humidité"""
        # Votre code actuel de build_moisture_from_rain
        river_moisture = self.__build_moisture_water_cell(0.85)
        self.voronoi_moisture = np.clip(self.voronoi_rainfall + river_moisture,
                                        0,
                                        1)

    def __build_moisture_water_cell(self,krate=0.95):
        voronoi_moisture = np.zeros_like(self.terrain.voronoi_altitude, dtype=float)
        points = np.array(self.terrain.voronoi_obj.points)
        water_points = points[self.water_cells]
        range_iter = tqdm(enumerate(points), desc="Calcul de l'humidité",
                          unit="ligne") if self.parameters.show_progress_bar else enumerate(points)
        for i, pi in range_iter:
            dist = np.linalg.norm(water_points - pi, axis=1).min()
            mi = krate ** dist
            voronoi_moisture[i] = mi
        return voronoi_moisture

# ============================================================================
# 3. BIOME CLASSIFIER - Responsabilité : Classification des biomes
# ============================================================================

class BiomeClassifier:
    """
    Responsabilité unique : Classification des biomes
    - Règles de classification altitude/humidité
    - Attribution des couleurs
    - Gestion des types de biomes
    """

    def __init__(self, terrain: TerrainGenerator, hydro: HydrographyGenerator):
        self.terrain = terrain
        self.hydro = hydro
        self.biomes = None
        self.biomes_color = {
            'Eau': '#004080',  # Bleu marine profond, évoquant une grande profondeur
            'Neige': '#FFFFFF',  # Blanc pur et brillant pour un effet lumineux
            'Thoundra': '#D8D8BF',  # Beige clair avec une teinte froide pour un terrain aride et gelé
            'Rocaille': '#8B8B8B',  # Gris roche plus saturé pour un aspect minéral
            'Sommets arides': '#6E6E6E',  # Gris sombre, presque anthracite, pour les reliefs arides
            'Forêt alpine': '#98C195',  # Vert frais et légèrement désaturé, reflétant l'altitude
            'Forêt clairsemée': '#B5D19C',  # Vert tendre, lumineux, pour une forêt clairsemée
            'Brousaille': '#C8D48B',  # Jaune-vert doux pour un terrain avec une végétation éparse
            'Forêt dense': '#6FA36E',  # Vert intense et profond, pour une forêt riche
            'Prairie fertile': '#89B87F',  # Vert naturel et vibrant, légèrement chaud
            'Zone humide': '#36A8D6',  # Vert foncé et bleuté, évoquant l'humidité stagnante
            'Marais boisé': '#7BB391',  # Vert vif avec une touche de fraîcheur pour les marais boisés
            'Plaines sèches': '#E8D3B0',  # Beige chaud et doux, reflétant des terrains semi-arides
        }

    def classify_biomes(self):
        """Classifie les biomes selon altitude et humidité"""
        # Votre code actuel de build_biomes
        voronoi_biomes = np.full(self.terrain.voronoi_obj.npoints, fill_value="", dtype=np.object_)

        moisture = self.hydro.voronoi_moisture
        altitude = self.terrain.voronoi_altitude
        voronoi_biomes[moisture >= 0] = "Eau"

        voronoi_biomes[(altitude < 1.01) & (altitude >= 0.8) & (moisture <= 1) & (moisture >= 0.8)] = 'Neige'
        voronoi_biomes[(altitude < 0.8) & (altitude >= 0.6) & (moisture < 1) & (moisture >= 0.8)] = 'Forêt alpine'
        voronoi_biomes[(altitude < 0.6) & (altitude >= 0.4) & (moisture < 1) & (moisture >= 0.8)] = 'Forêt dense'
        voronoi_biomes[(altitude < 0.4) & (altitude >= 0) & (moisture < 1) & (moisture >= 0.8)] = 'Zone humide'
        voronoi_biomes[(altitude < 1.01) & (altitude >= 0.8) & (moisture < 0.8) & (moisture >= 0.6)] = 'Neige'
        voronoi_biomes[(altitude < 0.8) & (altitude >= 0.6) & (moisture < 0.8) & (moisture >= 0.6)] = 'Forêt alpine'
        voronoi_biomes[(altitude < 0.6) & (altitude >= 0.4) & (moisture < 0.8) & (moisture >= 0.6)] = 'Forêt dense'
        voronoi_biomes[(altitude < 0.4) & (altitude >= 0) & (moisture < 0.8) & (moisture >= 0.6)] = 'Marais boisé'
        voronoi_biomes[(altitude < 1.01) & (altitude >= 0.8) & (moisture < 0.6) & (moisture >= 0.4)] = 'Neige'
        voronoi_biomes[(altitude < 0.8) & (altitude >= 0.6) & (moisture < 0.6) & (moisture >= 0.4)] = 'Forêt clairsemée'
        voronoi_biomes[(altitude < 0.6) & (altitude >= 0.4) & (moisture < 0.6) & (moisture >= 0.4)] = 'Forêt clairsemée'
        voronoi_biomes[(altitude < 0.4) & (altitude >= 0) & (moisture < 0.6) & (moisture >= 0.4)] = 'Forêt clairsemée'
        voronoi_biomes[(altitude < 1.01) & (altitude >= 0.8) & (moisture < 0.2) & (moisture >= 0.1)] = 'Rocaille'
        voronoi_biomes[(altitude < 0.8) & (altitude >= 0.6) & (moisture < 0.2) & (moisture >= 0.1)] = 'Brousaille'
        voronoi_biomes[(altitude < 0.6) & (altitude >= 0.4) & (moisture < 0.2) & (moisture >= 0.1)] = 'Prairie fertile'
        voronoi_biomes[(altitude < 0.4) & (altitude >= 0) & (moisture < 0.2) & (moisture >= 0.1)] = 'Prairie fertile'
        voronoi_biomes[(altitude < 1.01) & (altitude >= 0.8) & (moisture < 0.1) & (moisture >= 0)] = 'Sommets arides'
        voronoi_biomes[(altitude < 0.8) & (altitude >= 0.6) & (moisture < 0.1) & (moisture >= 0)] = 'Rocaille'
        voronoi_biomes[(altitude < 0.6) & (altitude >= 0.4) & (moisture < 0.1) & (moisture >= 0)] = 'Brousaille'
        voronoi_biomes[(altitude < 0.4) & (altitude >= 0) & (moisture < 0.1) & (moisture >= 0)] = 'Plaines sèches'
        voronoi_biomes[(altitude < 1.01) & (altitude >= 0.8) & (moisture < 0.4) & (moisture >= 0.2)] = 'Thoundra'
        voronoi_biomes[(altitude < 0.8) & (altitude >= 0.6) & (moisture < 0.4) & (moisture >= 0.2)] = 'Forêt clairsemée'
        voronoi_biomes[(altitude < 0.6) & (altitude >= 0.4) & (moisture < 0.4) & (moisture >= 0.2)] = 'Prairie fertile'
        voronoi_biomes[(altitude < 0.4) & (altitude >= 0) & (moisture < 0.4) & (moisture >= 0.2)] = 'Forêt clairsemée'

        self.voronoi_biomes = voronoi_biomes


# ============================================================================
# 4. MAP VISUALIZER - Responsabilité : Affichage et visualisation
# ============================================================================

class MapVisualizer:
    """
    Responsabilité unique : Visualisation
    - Tous les types d'affichage
    - Gestion des couleurs et styles
    - Export des cartes
    """

    def __init__(self, terrain: TerrainGenerator,
                 hydro: HydrographyGenerator = None,
                 biomes: BiomeClassifier = None):
        self.terrain = terrain
        self.hydro = hydro
        self.biomes = biomes

    def show_elevation_map(self, ax=None):
        """Affiche la carte d'élévation"""
        # Votre code actuel de show_voronoi_altitude
        if self.terrain.voronoi_grid is None or self.terrain.altitude is None:
            raise ValueError("Les cartes Voronoï ou d'altitude doivent être générées avant l'affichage.")

        # Calcul des altitudes moyennes
        unique_cells = list(range(self.terrain.voronoi_obj.npoints))
        mean_altitudes = self.terrain.voronoi_altitude

        # Créez une carte couleur pour afficher les altitudes moyennes
        voronoi_altitude_map = np.zeros_like(self.terrain.voronoi_grid, dtype=float)

        for i, cell_id in enumerate(unique_cells):
            voronoi_altitude_map[self.terrain.voronoi_grid == cell_id] = mean_altitudes[i]

        # Affichage
        if ax is None:
            fig, ax = plt.subplots()
        cax = ax.imshow(voronoi_altitude_map, cmap='terrain')
        return ax

    def show_river_network(self, ax=None, color="b"):
        """Affiche le réseau de rivières"""
        # Votre code actuel de show_river
        if ax is None:
            fig, ax = plt.subplots()
        pts = np.array(self.terrain.voronoi_obj.points)
        lw_min = 0.5
        lw_max = 3.0

        for (edge, flow) in zip(self.hydro.rivers_edges, self.hydro.rivers_edges_flows):
            lw = flow * (lw_max - lw_min) + lw_min
            ax.plot(pts[edge, 0], pts[edge, 1], color=color, lw=lw)

    def show_biome_map(self, ax=None):
        """Affiche la carte des biomes"""
        # Votre code actuel de show_biomes
        if ax is None:
            fig, ax = plt.subplots()

        for i, region_index in enumerate(self.terrain.voronoi_obj.point_region):  # Associe chaque point à sa région
            # if region_index not in self.frontier_cells :
            region = self.terrain.voronoi_obj.regions[region_index]

            # Vérifie si la région est valide (ne doit pas contenir -1)
            if not -1 in region and region:
                polygon = [self.terrain.voronoi_obj.vertices[v] for v in region]
                biome = self.biomes.voronoi_biomes[i]  # Biome associé à la cellule
                color = self.biomes.biomes_color[biome]  # Couleur associée au biome

                # Trace le polygone avec la couleur du biome
                ax.fill(*zip(*polygon), color=color)
        ax.set(xlim=[1, self.terrain.parameters.width - 1],
               ylim=[1, self.terrain.parameters.height - 1])



# ============================================================================
# 5. ORCHESTRATEUR PRINCIPAL - Coordonne tout
# ============================================================================

class WorldGenerator:
    """
    Orchestrateur principal qui coordonne tous les générateurs
    """

    def __init__(self, parameters: MapParameters):
        self.parameters = parameters
        self.terrain = TerrainGenerator(parameters)
        self.hydro = None
        self.biomes = None
        self.visualizer: MapVisualizer = None

    def generate_complete_world(self):
        """Génère un monde complet étape par étape"""
        print("Génération du terrain...")
        self.terrain.generate_base_terrain()
        self.terrain.build_voronoi_structure()
        self.terrain.apply_elevation_transform("Coast")
        self.terrain.fix_depressions()

        print("Génération de l'hydrographie...")
        self.hydro = HydrographyGenerator(self.terrain)
        self.hydro.calculate_rainfall_patterns()
        self.hydro.generate_river_network()
        self.hydro.calculate_moisture_distribution()

        print("Classification des biomes...")
        self.biomes = BiomeClassifier(self.terrain, self.hydro)
        self.biomes.classify_biomes()

        print("Initialisation du visualiseur...")
        self.visualizer = MapVisualizer(self.terrain, self.hydro, self.biomes)


# ============================================================================
# 6. UTILISATION - Plus simple et claire
# ============================================================================

if __name__ == "__main__":
    # Configuration
    params = MapParameters(
        width=800, height=800,
        scale_octave=5.0,
        # ... autres paramètres
    )
    figsize = (max(5, min(9, params.width * 5 / 250)),
               max(5, min(9, params.width * 5 / 250)))

    # Génération complète
    world = WorldGenerator(params)
    world.generate_complete_world()

    print("Affichage ...")
    figure = plt.figure(figsize=figsize)
    ax = figure.add_subplot(111)
    figure.tight_layout()
    world.visualizer.show_elevation_map(ax)

    # figure = plt.figure(figsize=figsize)
    # ax = figure.add_subplot(111)
    # figure.tight_layout()
    world.visualizer.show_river_network(ax)

    figure = plt.figure(figsize=figsize)
    ax = figure.add_subplot(111)
    figure.tight_layout()
    world.visualizer.show_biome_map(ax)

    plt.show()
