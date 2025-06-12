from map_generator import MapParameters,MapGenerator
import numpy as np
import matplotlib.pyplot as plt
import shutil, os



folder = 'cartes'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))




nmap = 10
params = MapParameters(width=350,
                       height=350,
                       scale_octave=5.0,
                       amplitude_func="Pow",
                       octaves=8,
                       voronoi_scale=2,
                       nrivers=25,
                       altitude_transform_mixing=0.4,
                       distribution_exponent=1.8,
                       show_progress_bar=False)

transform_list = ["Coast","SquareBump","Euclidean2"]
distribution_ratio = np.linspace(1.5,3.0,16)
altitude_transform_mixing = np.linspace(0.2,0.6,9)

for ni in range(nmap) :
    # Générer la carte
    generator = MapGenerator(parameters=params)
    print(f"Génération de la carte {ni+1}...")
    generator.generate_altitude_map(generator="Simplex")
    # print("Création du diagramme Voronoi ...")
    generator.build_voronoi_grid()
    # print("Calcul de l'altitude sur les cellules Voronoi ...")
    generator.build_voronoi_mapping()


    for trans_type in transform_list :
        for ratio in distribution_ratio :
            for mixing in altitude_transform_mixing :
                generator.parameters.distribution_exponent = ratio
                generator.parameters.altitude_transform_mixing = mixing

                # print("Transformation altitude ...")
                generator.transform_elevation(trans_type)
                # print("Correction des dépressions ...")
                generator.fillup_depression()
                # print("Calcul des rivières ...")
                generator.build_rivers_from_rainfall()
                # print("Calcul de l'humidité ...")
                generator.build_moisture()
                # print("Calcul des biomes ...")
                generator.build_biomes()

                figure = plt.figure(figsize=(11, 5))
                ax = figure.add_subplot(121)
                figure.tight_layout()
                generator.show_voronoi_altitude(ax)
                generator.show_voronoi(ax,
                                       show_points=False,
                                       show_vertices=False,
                                       line_colors="grey",
                                       line_alpha=0.5,
                                       line_width=0.5, )

                generator.show_river(ax)


                ax = figure.add_subplot(122)
                figure.tight_layout()
                generator.show_biomes(ax)
                generator.show_voronoi(ax,
                                       show_points=False,
                                       show_vertices=False,
                                       line_colors="grey",
                                       line_alpha=0.5,
                                       line_width=0.5, )


                figure.savefig(f"cartes/Carte {ni+1} - {trans_type} - R{ratio} - M{mixing}.png")
                plt.close(figure)