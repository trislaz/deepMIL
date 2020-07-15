Implémentation du deepMIL - WSI

Contient différents modèles de multiple instance learning permettant de traiter 
les Whole Slide Images avec différents modes :
* Possibilité de charger les WSI et d'apprendre les classifieurs MIL à partir : 
	* Des WSI encodés; chaque WSI est alors une matrice (NxM) de N tuiles et M features, 
	avec pour nom $table['ID'].npy. La structure du dossier à donner à args.wsi est alors :
  ```
	path_wsi/
		./nom_1.npy
		./nom_2.npy
		...
		./nom_n.npy
  ```

	* Des WSI "brutes", à une échelle donnée. Chaque WSI est alors un dossier contenant 
	les tuiles au format jpeg. La structure du dossier de données est : 
  ```
	path_wsi/
		./nom_1/
			./tile_0.jpg
			./tile_1.jpg
			...
		./nom_2/
			./tile_0.jpg
			./tile_1.jpg
			...
		...
  ```

* Utilisation de différets modèles de MIL:
	* AttentionDeepMIL [Ilse 2018]
	* CONAN
	* CHOWDER - 1S
	* Self-attention mechanism [Li & Eliceiri 2020]

* Lancement des apprentissages :
	* En nested cross-val avec random parameter search; se trouve dans cross_val/run.nf
	* En cross val simple avec un lot d'hyperparamètre donné dans un fichier /cross_val/handcrafted_config/config.yaml,
	lancement à partir du nf simple_cross_val.nf
	* Un seul entrainement possible grace à train/train.py, qui peut etre feed avec soit un config.yaml, soit à la ligne de commande


