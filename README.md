# Medical Image Problems Backbone

### Requirements

Pytoch >= 1.6 


### Guidelines
Following guidelines should be followed to correct performance:

  - Datasets have to return dictionaries with:
    - Training dataloader at least 1.'image' and 2.'label' and 3.'original_mask' entries and 'num_classes' and 'class_to_cat' attributes
    - num_classes should be 1 (no background) for single class segmentation or the number of classes + 1 (background)
    - when multiclass and average metrics and a class at class_to_cat named 'Mean' or as you prefere
    - Validation dataloader at least 1.'image', 2.'original_img', 3.'original_mask', 4.'img_id' entries

  - In segmentation background class is equals to label 0
  
### ToDo

- Valores de distancias infinitos (Hausdorff, ASSD)?
- Ejemplos de uso: classification.sh y segmentation.sh
- Redes: classification y segmentation
- Datasets juguete: classification y segmentation
- Método report para guardar metricas en csv de: 
  - Por epochs historial
  - Por pacientes par analisis de errores
  
- Probar problemas segmentacion multiclase y una clase
- Probar problemas classification multiclase y una clase  
