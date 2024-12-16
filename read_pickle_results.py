import pickle
import matplotlib.pyplot as plt


# Chemin du fichier pickle
#pickle_path = '/export/livia/home/vision/Aguichemerre/Pixel-Adaptation/exps/GLAS/resnet50/STD_CL/EnergyCAM/id_Energy_Source_GLAS_Inference_CAMELYON512_target-tsk_STD_CL-ds_GLAS-fold_0-mag_None-runmode_search-mode-mth_EnergyCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50/results_source_target_data.pickle'

pickle_path = '/export/livia/home/vision/Aguichemerre/Pixel-Adaptation/exps/CAMELYON512/resnet50/STD_CL/EnergyCAM/id_Energy_Source_CAMELYON512_Inference_GLAS_target_cl_correct-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_EnergyCAM-spooling_WGAP-arch_STDClassifier-ecd_resnet50/results_source_target_data.pickle'
# Charger les données
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

# Afficher ou utiliser les données
print(data['source_test_pxap'])


# Groups
groups = [
    ['target_test_acc_cl', 'source_test_acc_cl', 'source_train_acc_cl'],
    ['target_test_image_entropy', 'source_test_image_entropy', 'source_train_image_entropy'],
    ['target_test_pixel_entropy', 'source_test_pixel_entropy', 'source_train_pixel_entropy'],
    ['target_test_pxap', 'source_test_pxap', 'source_train_pxap'],
    ['target_test_dice_bg', 'source_test_dice_bg', 'source_train_dice_bg'],
    ['target_test_dice_fg', 'source_test_dice_fg', 'source_train_dice_fg'],
    ['target_test_miou', 'source_test_miou', 'source_train_miou']
]

# Generate graphics




# for i, group in enumerate(groups, start=1):
#     plt.figure(figsize=(8, 5))
#     for key in group:
#         plt.plot(data[key], label=key)
#     plt.title(f"Metric : {', '.join(group)}")
#     plt.xlabel("batch")
#     plt.ylabel("Value")
#     plt.legend()
#     plt.grid(True)
#     # Save
#     filename = f"graphique_{i}.png"
#     plt.savefig(filename, dpi=300)
#     plt.close()  
#     print(f"Graphic: {filename}")


# Generate graphics
# for i, group in enumerate(groups, start=1):
#     plt.figure(figsize=(8, 5))
#     for key in group:
#         # Prendre 1 valeur sur 2 pour le deuxième groupe
#         if i == 2:  # Spécifique au deuxième groupe
#             if key == 'source_train_image_entropy':
#                 filtered_data = data[key][::2]
#             else:
#                 filtered_data = data[key]
#             plt.plot(filtered_data, label=key)
#         else:
#             plt.plot(data[key], label=key)
#     plt.title(f"Metric : {', '.join(group)}")
#     plt.xlabel("epoch")
#     plt.ylabel("Value")
#     plt.legend()
#     plt.grid(True)
#     # Save
#     filename = f"graphique_{i}.png"
#     plt.savefig(filename, dpi=300)
#     plt.close()
#     print(f"Graphic: {filename}")
