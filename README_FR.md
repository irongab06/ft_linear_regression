# ft_linear_regression

Projet 42 : implémenter une simple régression linéaire avec descente de gradient

## Objectif

Prédire le prix d’une voiture à partir du kilométrage grâce à une régression linéaire.

## Installation

L’algorithme lui-même n’utilise que la bibliothèque standard de Python (`csv`, `math`, `sys`) : `predict.py` tourne avec n’importe quel Python 3 et n’a besoin de rien.  
`matplotlib` est la seule dépendance externe, utilisée par `train.py` pour tracer les graphes bonus. La façon recommandée de l’installer est un environnement virtuel :

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Si `matplotlib` n’est pas installé, `train.py` entraîne quand même le modèle et sauvegarde `theta.txt`, il saute seulement les graphes.

## Utilisation

```bash
python3 train.py     # lit data.csv, entraîne le modèle, sauvegarde theta.txt et les graphes
python3 predict.py   # demande un kilométrage et affiche le prix estimé
```

Si `predict.py` est lancé avant `train.py`, le fichier `theta.txt` n’existe pas encore : θ₀ et θ₁ valent 0 et la prédiction est 0.

## Introduction

La régression linéaire est un algorithme très simple et largement utilisé en machine learning.
Son objectif est de modéliser la relation entre une variable indépendante **x** (ici, le kilométrage des voitures) et une variable dépendante **y** (le prix de la voiture) à l’aide d’une fonction linéaire.

## Modèle

F(x) = ax + b

<u>**x**</u> → variable indépendante (kilométrage de la voiture)  
<u>**F(x)**</u> → prédiction (prix approximatif)  
<u>**a (ou θ₁)**</u> → pente de la droite (impact du kilométrage sur le prix)  
<u>**b (ou θ₀)**</u> → ordonnée à l’origine (prix approximatif lorsque **x = 0**)  

### Modèle en Python

```python
def estimate_price(mileage, theta0, theta1):
	return theta0 + (theta1 * mileage)
```

C’est exactement la formule du sujet : **estimatePrice(mileage) = θ₀ + (θ₁ * mileage)**

<u>**mileage**</u> → variable indépendante **x** (kilométrage de la voiture)  
<u>**theta0**</u> → ordonnée à l’origine **θ₀** (prix lorsque le kilométrage vaut 0)  
<u>**theta1**</u> → pente **θ₁** (variation du prix pour une unité de kilométrage en plus)  
<u>**theta0 + (theta1 * mileage)**</u> → le modèle linéaire **ax + b**  

## Normalisation

Les kilométrages sont de grands nombres (jusqu’à 250 000 km). Avec les valeurs brutes, le gradient de θ₁ est énorme et la descente de gradient diverge.  
`train.py` divise donc chaque kilométrage par `SCALE = 10000` avant l’entraînement :

```python
SCALE = 10000
x = [k / SCALE for k in km]
```

À la fin de l’entraînement, θ₁ est divisé par `SCALE` avant d’être sauvegardé, pour que `predict.py` travaille directement sur le kilométrage brut :

```python
with open("theta.txt", "w") as f:
	f.write(f"{theta0}\n{theta1 / SCALE}\n")
```

`theta.txt` contient **θ₀** sur la première ligne et **θ₁** sur la seconde.

## Fonction de coût

La fonction de coût sert à calculer la somme des erreurs au carré (norme euclidienne) en régression linéaire.

J(θ) = 1 / 2m * ∑_{i=1}^{m} (f(x(i)) - y(i))²

<u>**J(θ)**</u> → valeur de la fonction de coût (somme des erreurs au carré)  
<u>**m**</u> → nombre total d’échantillons  
<u>**∑**</u> → somme du premier échantillon au dernier  
<u>**f(x(i))**</u> → prédiction du modèle pour l’échantillon **i (f(x) = ax + b)**  
<u>**y(i)**</u> → valeur réelle de l’échantillon **i**  
<u>**1 / 2**</u> → facteur de normalisation (on divise par le nombre d’échantillons, et le 2 simplifie ensuite les dérivées)  

### Fonction de coût en Python

Ce nom de fonction est **Mean Squared Error** (MSE).

```python
def cost_function(x, y, theta0, theta1):
	m = len(x)
	return sum((estimate_price(x[i], theta0, theta1) - y[i]) ** 2 for i in range(m)) / (2 * m)
```

<u>**m**</u> → nombre total d’échantillons (longueur de la liste x)  
<u>**x**</u> → liste contenant tous les kilométrages (normalisés)  
<u>**y**</u> → liste contenant toutes les valeurs réelles (prix réels des échantillons)  
<u>**estimate_price(x[i], theta0, theta1) - y[i]**</u> → erreur pour l’échantillon **i** (différence entre le prix prédit et le prix réel)  
<u>**(...)²**</u> → erreur au carré, qui pénalise les grands écarts  
<u>**sum(... for i in range(m))**</u> → somme de toutes les erreurs au carré sur l’ensemble des données  
<u>**/ (2 * m)**</u> → facteur de normalisation (division par le nombre d’échantillons, et le 2 simplifie le calcul de la dérivée pour la descente)  

## Algorithmes de minimisation

Cette partie sert à calculer le gradient et à effectuer la descente de gradient.  
Elle est très importante pour compléter le processus de régression linéaire en machine learning.

### Gradient

Le gradient est la dérivée de la fonction de coût par rapport à chaque paramètre. Il donne la direction dans laquelle **θ₀** et **θ₁** doivent bouger pour faire baisser le coût.  
Pour un modèle linéaire, les deux dérivées partielles sont les deux formules données dans le sujet :

tmpθ₀ = learningRate * 1/m * ∑_{i=0}^{m-1} ( estimatePrice(mileage[i]) − price[i] )  
tmpθ₁ = learningRate * 1/m * ∑_{i=0}^{m-1} ( estimatePrice(mileage[i]) − price[i] ) * mileage[i]  

<u>**estimatePrice(mileage[i]) − price[i]**</u> → erreur du modèle pour l’échantillon **i**  
<u>**1/m * ∑**</u> → moyenne des erreurs (pour **θ₀**), ou moyenne des erreurs pondérées par le kilométrage (pour **θ₁**)  
<u>**learningRate**</u> → taille du pas de mise à jour  

### Descente de gradient

Son objectif est de mettre à jour les paramètres dans la direction opposée au gradient afin de minimiser la valeur de **J(θ)**.

```python
def gradient_descent(x, y, learning_rate, n_iterations):
	m = len(x)
	theta0, theta1 = 0.0, 0.0
	cost_history = []
	for _ in range(n_iterations):
		errors = [estimate_price(x[i], theta0, theta1) - y[i] for i in range(m)]
		tmp_theta0 = learning_rate * sum(errors) / m
		tmp_theta1 = learning_rate * sum(errors[i] * x[i] for i in range(m)) / m
		theta0 = theta0 - tmp_theta0
		theta1 = theta1 - tmp_theta1
		cost_history.append(cost_function(x, y, theta0, theta1))
	return theta0, theta1, cost_history
```

<u>**theta0, theta1 = 0.0, 0.0**</u> → les paramètres partent de zéro  
<u>**errors**</u> → liste des erreurs (prix prédit − prix réel), calculée une fois par itération avec les **θ₀** et **θ₁** courants  
<u>**tmp_theta0 / tmp_theta1**</u> → les deux pas de mise à jour du sujet, stockés dans des variables temporaires  
<u>**theta0 = theta0 - tmp_theta0**</u> et <u>**theta1 = theta1 - tmp_theta1**</u> → assignation simultanée : les deux temporaires sont calculées à partir des mêmes anciennes valeurs avant que l’un ou l’autre paramètre ne change. La soustraction déplace les paramètres dans la direction opposée au gradient (car l’objectif est de minimiser la fonction de coût, pas de la maximiser)  
<u>**learning_rate**</u> → hyperparamètre qui contrôle la taille du pas de mise à jour (trop grand → divergence, trop petit → apprentissage lent). Ici **0.01**  
<u>**n_iterations**</u> → nombre de passages sur les données. Ici **2000**  
<u>**cost_history**</u> → coût après chaque itération, utilisé pour tracer la courbe de perte  

## Coefficient de détermination

Le coefficient de détermination **(R²)** est une valeur comprise entre 0 et 1 qui mesure la qualité des prédictions du modèle.

```python
def coef_determination(y, pred):
	mean_y = sum(y) / len(y)
	u = sum((y[i] - pred[i]) ** 2 for i in range(len(y)))
	v = sum((yi - mean_y) ** 2 for yi in y)
	return 1 - u / v
```

<u>**y**</u> → liste des valeurs réelles (prix réels).  
<u>**pred**</u> → liste des valeurs prédites par le modèle.  
<u>**mean_y**</u> → moyenne des valeurs réelles (prix moyen).  
<u>**(y[i] - pred[i])**</u> → résidu (erreur entre la valeur réelle et la valeur prédite).  
<u>**u**</u> → **SSR** (Sum of Squared Residuals), somme des erreurs au carré du modèle.  
<u>**v**</u> → **SST** (Total Sum of Squares), variance des données par rapport à la moyenne.  
<u>**1 - u/v**</u> → formule du coefficient de détermination **(R²)**. Elle mesure la proportion de variance de y expliquée par le modèle (entre **0** et **1**).  

## Root Mean Squared Error (RMSE)

C’est une métrique d’évaluation qui mesure l’écart moyen typique entre les valeurs prédites par le modèle et les vraies valeurs.

```python
def rmse(y, pred):
	return math.sqrt(sum((y[i] - pred[i]) ** 2 for i in range(len(y))) / len(y))
```

<u>**y**</u> → liste des valeurs réelles (prix réels).  
<u>**pred**</u> → liste des valeurs prédites par le modèle.  
<u>**(y[i] - pred[i])²**</u> → résidu au carré (les grandes erreurs sont pénalisées plus fortement).  
<u>**sum(...) / len(y)**</u> → **MSE** (Mean Squared Error), erreur moyenne au carré des prédictions.  
<u>**math.sqrt(...)**</u> → racine carrée du MSE, ce qui ramène l’erreur à la même unité que y (par exemple, en euros).  

![linear Regression](Linear_Regression.png)

## Glossaire des abréviations

<u>**MSE (Mean Squared Error)**</u> → moyenne des différences au carré entre les valeurs prédites et les vraies valeurs.  
<u>**RMSE (Root Mean Squared Error)**</u> → racine carrée du **MSE**, ce qui ramène l’erreur à la même unité que y.  
<u>**R² (Coefficient de détermination)**</u> → proportion de variance de **y** expliquée par le modèle (entre **0** et **1**).  
<u>**SSR (Sum of Squared Residuals)**</u> → somme des différences au carré entre les valeurs prédites et les valeurs réelles (erreurs du modèle).  
<u>**SST (Total Sum of Squares)**</u> → variance totale des données par rapport à la moyenne de **y**.  
<u>**θ₀ (theta0)**</u> → ordonnée à l’origine ou biais (valeur prédite lorsque **x = 0**).  
<u>**θ₁ (theta1)**</u> → pente ou poids (impact de la variable **x** sur **y**).  
<u>**J(θ) (Fonction de coût)**</u> → fonction mesurant l’erreur du modèle (somme des erreurs au carré).  
<u>**Learning rate (α)**</u> → hyperparamètre qui contrôle la taille du pas dans les mises à jour de la descente de gradient.  
<u>**Gradient**</u> → vecteur de dérivées partielles de la fonction de coût par rapport aux paramètres (direction de la plus forte augmentation).  
<u>**Epoch / Iteration**</u> → une étape complète de mise à jour des paramètres pendant la descente de gradient.  
<u>**Assignation simultanée**</u> → **θ₀** et **θ₁** sont tous deux calculés à partir des mêmes anciennes valeurs (variables temporaires) avant d’être assignés.  
<u>**Normalisation (SCALE)**</u> → les kilométrages sont divisés par 10000 pendant l’entraînement pour que la descente de gradient converge ; **θ₁** est redivisé avant la sauvegarde.  
<u>**Over-fitting (sur-apprentissage)**</u> → quand un modèle colle trop aux données d’entraînement (y compris leur bruit) et prédit mal de nouvelles données. Une prédiction qui tombe exactement sur une valeur d’entraînement est suspecte.  

## Ensemble de formules mathématiques et de programmation

![linear Regression](Formule-math.jpg)![linear Regression](Formule-for-code.jpg)

## Auteur
Gabriel Cavalier  
[Mon GitHub](https://github.com/irongab06)
