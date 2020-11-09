# Medical Image Problems Backbone

### Requirements

Pytoch >= 1.6 


### Guidelines
Following guidelines should be followed to correct performance:

  - Datasets have to return dictionaries with:
    - Training dataloader at least 1.'image' and 2.'label' and 3.'original_mask' entries and 'num_classes' and 'class_to_cat' attributes
    - num_classes should be 1 (no background) for single class segmentation or the number of classes + 1 (background)
    - when multiclass and average metrics and a class at class_to_cat named 'Mean' or as you prefer
    - Validation dataloader at least 1.'image', 2.'original_img', 3.'original_mask', 4.'img_id' entries
  - If you want to load checkpoint unfreezed set defrost_epoch param to 0 
  - In segmentation background class is equals to label 0
  - You can use --notify to send you a slack message to 'experiments' channel. Set envionment variable with slack token. How can create slack token [here](https://github.com/MarioProjects/Python-Slack-Logging):
  ```shell script
export SLACK_TOKEN='you_slack_token'
```

  
### ToDo

- Valores de distancias infinitos (Hausdorff, ASSD)?
- Ejemplos de uso: classification.sh
- Redes: classification y segmentation
- Método report para guardar metricas en csv de: 
  - Por pacientes par analisis de errores
  
- Probar problemas classification multiclase y una clase  
