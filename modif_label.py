# Lire le fichier, traiter chaque ligne et écrire les résultats dans un nouveau fichier

# Nom du fichier d'entrée et de sortie
input_file = '/export/livia/home/vision/Aguichemerre/Pixel-Adaptation/folds/wsol-done-right-splits/CAMELYON512/fold-6/train/class_labels.txt'  # Remplacez par le chemin de votre fichier d'entrée
output_file = 'output_file.txt'

# Lecture et traitement
def process_lines(input_file, output_file):
    try:
        with open(input_file, "r") as infile:
            lines = infile.readlines()

        processed_lines = []

        for line in lines:
            # Condition pour les lignes contenant "metastatic"
            if "w-512xh-512/metastatic-patches/" in line:
                line = line.rsplit(",", 1)[0] + ",1\n"
            # Condition pour les lignes contenant "w-512xh-512/normal-patches"
            elif "w-512xh-512/normal-patches/" in line:
                line = line.rsplit(",", 1)[0] + ",0\n"

            processed_lines.append(line)

        # Écriture des lignes traitées dans un nouveau fichier
        with open(output_file, "w") as outfile:
            outfile.writelines(processed_lines)

        print(f"Traitement terminé. Les résultats ont été écrits dans {output_file}")

    except FileNotFoundError:
        print(f"Le fichier {input_file} est introuvable.")

# Appel de la fonction
process_lines(input_file, output_file)