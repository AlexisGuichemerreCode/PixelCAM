import numpy as np

def process_file(input_file, output_file):
    # Lire le fichier et stocker les lignes
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Stocker les résultats dans un dictionnaire
    image_means = {}

    # Stocker la classe correspondante
    image_classes = {}

    # Parcourir les lignes uniques par image
    unique_images = sorted(set(line.split(',')[0] for line in lines))
    for image_name in unique_images:
        # Récupérer toutes les lignes pour une image spécifique
        matching_lines = [line for line in lines if line.startswith(image_name)]
        
        # Filtrer les lignes contenant "PixelCAM"
        pixelcam_lines = [line for line in matching_lines if 'PixelCAM' in line.split(',')[1]]
        
        # Si aucune ligne ne correspond à "PixelCAM", passer à l'image suivante
        if not pixelcam_lines:
            continue
        
        # Extraire le 4ème élément (3ème index) pour ces lignes et convertir en float
        fourth_elements = [float(line.split(',')[3]) for line in pixelcam_lines]
        
        # Extraire la classe (dernier élément de la ligne) pour la première ligne correspondante
        image_class = pixelcam_lines[0].strip().split(',')[-1]
        
        # Calculer la moyenne
        mean_value = np.mean(fourth_elements)
        
        # Stocker la moyenne et la classe pour l'image
        image_means[image_name] = mean_value
        image_classes[image_name] = image_class
    
    # Trier les images par moyenne décroissante
    sorted_means = sorted(image_means.items(), key=lambda x: x[1], reverse=True)
    
    # Écrire les résultats dans un nouveau fichier
    with open(output_file, 'w') as f:
        for image_name, mean in sorted_means:
            image_class = image_classes[image_name]
            f.write(f"{image_name},{mean},{image_class}\n")


input_file = "/export/livia/home/vision/Aguichemerre/Pixel-Adaptation/visualization/tsne/CAMELYON512/target_cam_performance_log.txt"  
output_file = "/export/livia/home/vision/Aguichemerre/Pixel-Adaptation/visualization/tsne/CAMELYON512/target_class_separability_log.txt"
  
process_file(input_file, output_file)
