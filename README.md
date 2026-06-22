# Deep Learning — Projet de fin de module (Refonte individuelle)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Transformers-yellow)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)

**Auteur :** BADR EDDINE EL ABBADY
**Établissement :** FST Settat — Université Hassan 1er

Refonte individuelle d'un projet de Deep Learning regroupant quatre travaux pratiques.
Pour chaque TP, une **approche volontairement différente** de la version précédente a été
adoptée, afin d'explorer d'autres architectures et techniques.

---

## Description des TP

### TP1 — Classification FashionMNIST avec EfficientNet-B0
`tp1_fashionmnist/tp1_fashionmnist_efficientnet.ipynb`

- **Approche :** transfer learning avec **EfficientNet-B0** préentraîné ImageNet, images
  redimensionnées en 64×64 et converties en RGB, augmentation forte (`RandAugment` +
  `RandomErasing`), `label_smoothing`, scheduler cosinus et early stopping.
- **Pourquoi c'est différent :** la version précédente reposait sur des architectures
  classiques type **ResNet/VGG**. EfficientNet-B0 offre un meilleur compromis
  précision/coût grâce au *compound scaling*, et l'augmentation agressive renforce la
  généralisation sur un dataset simple.

### TP2 — Prévision de la consommation électrique
`tp2_consommation_electrique/`

- **Approche :** *(à compléter — notebook non encore rédigé)*.
- **Pourquoi c'est différent :** *(à documenter lors de la rédaction du TP2)*.

### TP3 — Régression des émissions de CO2 (CO2 Emissions Canada)
`tp3_regression_vehicules/tp3_regression_co2_mlp_rf.ipynb`

- **Approche :** comparaison de **trois modèles** (MLP PyTorch, Random Forest, LSTM léger),
  suppression de la variable CO2 des features d'entrée, export **TFLite** (TinyML) du MLP.
- **Pourquoi c'est différent :** la version précédente utilisait le CO2 comme prédicteur
  (circularité faussant les performances). Ici on l'exclut, on compare trois familles de
  modèles et on ajoute une dimension **déploiement embarqué** via TFLite.

### TP4 — Classification d'émotions avec RoBERTa-base
`tp4_emotions_transformers/tp4_roberta_emotions.ipynb`

- **Approche :** **RoBERTa-base** fine-tuné sur le dataset `emotion` (6 classes), gestion
  du déséquilibre via `WeightedRandomSampler`, early stopping sur le **F1 macro**.
- **Pourquoi c'est différent :** la version précédente utilisait **DistilBERT** sans
  traitement du déséquilibre. RoBERTa-base est plus performant, et la pondération de
  l'échantillonnage améliore le rappel sur les classes rares (`love`, `surprise`).

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Lancement de chaque TP

```bash
# TP1 — FashionMNIST (téléchargement auto du dataset via torchvision)
jupyter notebook tp1_fashionmnist/tp1_fashionmnist_efficientnet.ipynb

# TP2 — Consommation électrique
jupyter notebook tp2_consommation_electrique/

# TP3 — CO2 (placer "CO2 Emissions_Canada.csv" dans le dossier du TP)
jupyter notebook tp3_regression_vehicules/tp3_regression_co2_mlp_rf.ipynb

# TP4 — Émotions (téléchargement auto du dataset via 🤗 datasets)
jupyter notebook tp4_emotions_transformers/tp4_roberta_emotions.ipynb
```

---

## Tableau récapitulatif des métriques

> À compléter après exécution complète des notebooks (les notebooks ne sont pas encore
> entraînés dans ce dépôt).

| TP  | Tâche | Modèle | Métrique principale | Résultat |
|-----|-------|--------|---------------------|----------|
| TP1 | Classification images | EfficientNet-B0 | Test accuracy / F1 macro | _à compléter_ |
| TP2 | Prévision séries temporelles | _à définir_ | RMSE / MAE | _à compléter_ |
| TP3 | Régression CO2 | MLP / RandomForest / LSTM | MAE / RMSE / R² | _à compléter_ |
| TP4 | Classification texte | RoBERTa-base | Accuracy / F1 macro | _à compléter_ |
