# Traitement de données avec Python

Ce répertoire contient les carnets (Jupyter Notebook) présentés lors du séminaire du 8 décembre 2017.

## Visualiser les carnets

Vous devez avoir préalablement installé Python sur votre poste. Il est recommandé d’installer Anaconda, qui contient tous les packages nécessaires pour bien commencer dans l’environnement Python.

Ensuite, clonez ce répertoire en exécutant `git clone https://github.com/samuelduchesne/SeminairePython.git` depuis le terminal ou bien en utilisant l’application [GitHub Desktop](https://desktop.github.com). Si vous utilisez le terminal, vous devez être situé dans le dossier de votre choix avant d'exécuter la commande. Exécutez `cd c:/<chemin d'accès de votre choix>/`. Ensuite, exécutez la commande git decrite plus haut.

Dans le terminal, lancez Jupyter avec la commande `jupyter notebook`. Une fenêtre de votre navigateur internet s’ouvrira. Naviguez jusqu’au répertoire du séminaire tout juste téléchargé. Cliquez sur l’un des carnets (extension `.ipynb`).

### Dépendances

Le code inclu dans le répertoire `pysurfrad` dépend de la librairie [pvlib](http://pvlib-python.readthedocs.io/en/latest/index.html). Installez-la avec la commande `conda install -c pvlib pvlib`.
