## Performance of `TransferModelCNN.py` with resnet18

### Classification Report

```
                          precision    recall  f1-score   support

   bacterial_leaf_blight       0.89      0.89      0.89        80
   bacterial_leaf_streak       0.92      0.95      0.93        80
bacterial_panicle_blight       0.79      0.96      0.87        80
        black_stem_borer       0.91      0.89      0.90        80
                   blast       0.86      0.75      0.80        80
              brown_spot       0.79      0.84      0.81        80
            downy_mildew       0.85      0.89      0.87        80
                   hispa       0.79      0.84      0.81        80
             leaf_roller       0.93      0.79      0.85        80
                  normal       0.75      0.96      0.85        80
                  tungro       0.85      0.78      0.81        80
        white_stem_borer       1.00      0.68      0.81        80
       yellow_stem_borer       0.94      0.94      0.94        80

                accuracy                           0.86      1040
               macro avg       0.87      0.86      0.86      1040
            weighted avg       0.87      0.86      0.86      1040
```

### Confusion Matrix

<img src='images/TransferModelCNN_ConfusionMatrix.png'/>
