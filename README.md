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

*MMs dataset naming*:
  - `_full` Get all volumes (not only segmented 'ED' and 'ES' phases volumes).
  - `_unlabeled` Get only unlabeled volumes (for 'ED' and 'ES' phases)
  - `_centre*xyz*` Get volumes (for 'ED' and 'ES' phases) for selected centres. Example `_centre1`, `_centre13`. Last one picks centres 1 and 3. Available Centres from 1 to 5.
  - `_vendor*jkl*` Get volumes (for 'ED' and 'ES' phases) for selected vendors. Example `_centreC`, `_vendorAB`. Last one picks vendors A and B. Available Vendors 'A', 'B', 'C', 'D'.

  
### ToDo

- Valores de distancias infinitos (Hausdorff, ASSD)?
- Ejemplos de uso: classification.sh
- Redes: classification y segmentation
- Método report para guardar metricas en csv de: 
  - Por pacientes par analisis de errores
  
- Probar problemas classification multiclase y una clase  
