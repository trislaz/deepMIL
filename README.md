# Multiple Instance Learning on WSI

## Installer le package
Il suffit de `cd pkg` puis d'installer avec la commande `pip install -e ./`.

## Lancer des apprentissage de multiple instance learning

Les input nécessaires aux processus de ce package sont : 
* Le dataset de WSI, déja tuilé et encodé (une WSI est alors un array numpy "name_slide.npy")
* Le "table_data", un fichier csv (séparateur = ",") qui contient forcément les colonnes :
    * ID : contient les noms des images, doit correspondre au nom du fichier numpy de la WSI encodée.
    * target_name: un colonne dont le nom match le paramètre "target_name" qui doit etre renseigné au lancement d'un processus
    * test : colonne contenant des entiers, répartissant les slides au sein des sets de tests.

Si le dataset n'est pas déja foldé en test-sets, il est possible de le faire à l'aide de `./scripts/split_dataset.py`.

Exemple d'utilisation de `./scripts/split_dataset.py`:
```
python ./scripts/split_dataset.py --table_in /path/table_data.csv 
                                  --table_out /path/table_data_folded.csv 
                                  --target_name  name_of_target # pour stratification, doit matcher une colonne du table_data.csv
                                  -k 5 # Nombre de folds
```

Ensuite, plusieurs options sont possibles.

1. Lancer une recherche d'hyperparamètres + cross_validation:

Modifier le préambule de `./cross_val/run.nf`pour spécifier:
* `model_name` : Les noms des modèles que l'on souhaite tester. dispos = attentionmil | conan | 1s | transformermil | sa .
* `resolution` : int, les résolutions auxquelles on souhaite appliquer le MIL
* `dataset`: nom du dataset sur lequel on fait tourner les algos, sert en fait juste de nom pour identifier l'expérience (on peut mettre ce qu'on veut)
* `table_data`: path vers le fichier csv mentionné avant.
* `target_name`: nom du label. Doit matcher une colonne du table_data.
* `nb_para`: nombre d'ensembles d'hyperparametres à comparer. Ils sont échantillonnés aléatoirement.
* `test_fold`: int, renseigne sur le nombre de fold de tests sur lesquels on va faire la nested_cross_val.
    Peut au maximum être égal au nombre total de folds créés par split_dataset.
* `repetition`: nombre de répétitions d'apprentissage pour chaque test, chaque set d'hyperparamètres. 
    A chaque répétition, le training set est scindé en train et val de façon aléatoire (stratifié).
* `epochs`: nombre d'epochs d'apprentissage.

Une fois tout celà précisé, il suffit de lancer le process nextflow :
```
nextflow ./cross_val/run.nf -c /path/to/the/nf_config.nf
```

L'apprentissage va automatiquement créer un dossier dans lequel on trouvera les résultats.
Ce dossier a pour path `./cross_val/outputs/dataset/model_name/resolution/`.
On y trouvera différentes choses : 
* les fichiers de résultats: 
    * final_results.csv, contient les performances du meilleur modèle (meilleure des répétitions) sur les jeux de tests.
    * all_results.csv, contient les performances de validation de tous les modèles entrainés
    * mean_over_repeats.csv, contient les **moyennes** des performances de validation sur les répétitions (contient donc test_fold * nb_para lignes)
    * mean_over_tests.csv, contient les **moyennes** des performances de validation sur les répétitions et les tests (contient donc nb_para lignes)
* la meilleure configuration `best_config_*.yaml`est le fichier de configuration contenant les hyperparamètres ayant apporté les meilleures perfs.
* les meilleurs modèles `model_best_test_n.pt.tar`sont les poids et paramètres des meilleures répétitions, parmis celles correspondant à la *best_config*.
* ./configs/ contient tous les fichiers de configuration
* ./config_n sont les dossiers dans lesquels contiennent les modèles, les sorties tensroboard etc.. de tous les modèles de la *config_n*

**Au total, `nb_para * test_fold * repetition` seront lancés !**

2. Lancer une simple cross_val.

Permet de lancer simplement une *nested cross validation* avec un set fixe d'hyperparamètre (avec objectif de sélection de modèle et d'évaluation de performances).

De même remplir le préambule de `./cross_val/simple_cross_val.nf` avec les paramètres voulus.
Deux paramètre sont en sus : 
* `config_path` renseigne du chemin vers le fichier de config à utiliser (fichier en `.yaml`)
* `expe` précise le nom de l'experience, juste pour le stockage des outputs
On peut le lancer de la même manière que `run.nf`.

Les sorties se trouvent dans `./cross_val/outputs/dataset/expe/model_name/resolution/` On y trouve les mêmes fichiers que pour le processus de recerche `1.`

3. Lancer une prédiction -ne peut se faire que sur le dataset initial pour l'instant-

```
python ./scripts/predict.py --config /path/to/config --model_path /path/to/model.pt.tar
```

4. Lancer un seul entrainement, avec des paramètres donnés

```
python ./scripts/train.py --config /path/to/config.yaml 
```

On peut aussi rajouter des paramètres à la main, plutot que de les rajouter dans config : 

```
python ./scripts/train.py --conig /path/to/config.yaml` --patience 15
```

Des infos sur les paramètres disponibles peuvent être obtenues en demain l'aide : 

```
python ./scripts/train.py --help
```

## Visualiser les entrainements

Il est possible de visualiser en temps réel un ou plusieurs apprentissage.
2 cas sont envisageables :
1. Les calculs tournent sur votre machine. Il faut installer tensorboardx (`conda install tensorboardx` si vous ne l'avez pas) puis le lancer : 
```
tensorboard --logdir /path/of/the/experiment_outputs --port=8008
```
Ouvrez ensuite votre navigateur et entrez dans la barre d'adresse `localhost:8008/`. J'utilise ici le port 8008 mais vous pouvez utiliser celui de votre choix.
2. Les calculs tournent sur une machine distante. Il faut alors ouvrir un tunnel ssh entre un des ports de votre machine et cette machine distante : 
```
ssh user@machine_distance -L 8008:localhost:8008
```
vous pouvez ensuite, sur votre machine distante, lancer tensorboard comme dit dans le point un sur le port 8008 (qui est maintenant retransmis sur le port 8008 de votre machine locale), et ouvrir votre navigateur à l'adresse `localhost:8008/`
