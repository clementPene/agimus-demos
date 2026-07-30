# Test rapide MegaPose sur le pylône (caméra de tête)

Objectif : vérifier que MegaPose estime correctement la pose 6D du pylône
depuis une image de la caméra de tête, avant de monter le pipeline complet
(détecteur + happypose_ros + MPC).

## Prérequis (PC GPU, une seule fois)

```bash
pip install "happypose[render]" trimesh
export HAPPYPOSE_DATA_DIR=$HOME/happypose_data   # à mettre dans le .bashrc
python -m happypose.toolbox.utils.download --megapose_models   # ~1 GB
```

## Procédure

1. **Capturer** une image + intrinsèques (machine qui voit les topics ROS,
   pylône à ~1-1.5 m, bien dans le champ) :

   ```bash
   python3 capture_image.py --output ./captures/capture_01
   ```

   Topics par défaut : `/head_front_camera/color/image_raw` et
   `.../camera_info` (`--image-topic` / `--info-topic` sinon).

2. **Convertir le mesh** (une seule fois) — STL mètres → PLY millimètres,
   gris très foncé (pylône noir, mais le noir pur écrase le shading du
   rendu) :

   ```bash
   python3 convert_pylone_mesh.py --output ./pylone.ply
   ```

3. **Construire l'exemple happypose** — la bbox se trace à la souris :

   ```bash
   python3 make_megapose_example.py --capture ./captures/capture_01 --mesh ./pylone.ply
   ```

4. **Inférence + visualisation** (PC GPU) — lancée depuis `results/` pour
   que les PNG de secours (`render_raw.png`, `contour_overlay.png`, voir
   Notes) y atterrissent directement plutôt que dans `vision/` :

   ```bash
   cd results/
   python -m happypose.pose_estimators.megapose.scripts.run_inference_on_example \
       pylone --run-inference --vis-poses
   cd ..
   ```

   Résultats dans `$HAPPYPOSE_DATA_DIR/examples/pylone/` :
   - `outputs/object_data.json` — pose estimée `T_caméra→pylône`
     (quaternion TWO + translation, en m)
   - `visualizations/` — rendu du mesh superposé à l'image : si le contour
     colle au pylône réel, le test est concluant.

Refaire les étapes 1 et 3 pour 3-4 points de vue différents pour juger la
stabilité (`--output ./captures/capture_02`, `--label` reste `pylone`,
l'exemple est écrasé à chaque fois).

## Structure du dossier

- `capture_image.py`, `convert_pylone_mesh.py`, `make_megapose_example.py`,
  `pylone_pose_estimator_node.py` — scripts (suivis par git).
- `pylone.ply` — mesh converti, réutilisé à chaque run (pas un résultat
  jetable, généré une fois par `convert_pylone_mesh.py`).
- `captures/` — images + intrinsèques capturées (`capture_NN/`), non suivi
  par git.
- `results/` — sorties de `run_inference_on_example` (PNG/HTML de
  visualisation), non suivi par git.

## Étape 5 : nœud persistant + intégration orchestrator

Une fois l'exemple `pylone` calibré (étapes 1-3 ci-dessus), un nœud ROS2
persistant peut servir des estimations à la demande au lieu de relancer le
script CLI à chaque fois :

```bash
python3 pylone_pose_estimator_node.py   # dans vision_cuda
```

Expose :
- Service `/vision_pylone/estimate` (`std_srvs/Trigger`) — déclenche une
  capture + inférence (~20-30s), publie le résultat.
- Topic `/vision_pylone/pose` (`geometry_msgs/PoseStamped`, latché) — dernière
  pose estimée, en frame `base_link`.
- TF `base_link -> pylone_vision_estimate`, rediffusée à chaque estimation.

Ce nœud est joignable depuis le container `control` sans configuration
réseau supplémentaire (les deux devcontainers tournent en
`--network host`, donc DDS les voit comme deux terminaux sur la même
machine). Côté orchestrator HPP (`hpp/orchestrator.py`), méthodes miroir de
celles de la mocap :

```python
o.connect_vision()
o.compare_vision()              # affiche le delta vision ↔ pose courante (q_init)
o.localize_pylone_from_vision() # commit la pose vision dans q_init + config/pylone_pose_vision.yaml
```

`config/pylone_pose_vision.yaml` est un fichier séparé de
`config/pylone_pose.yaml` (qui reste la source de confiance
mocap/pointage manuel) — les deux peuvent être comparés sans que l'un
n'écrase l'autre.

**Limite connue** : la bbox 2D n'est PAS re-détectée à chaque appel du
service (MegaPose n'est pas un détecteur, cf. Notes ci-dessous) — le nœud
recharge celle du dernier `make_megapose_example.py` exécuté à la main.
Recalibrer si la caméra ou le pylône bougent significativement.

## Notes

- MegaPose n'est pas un détecteur : la bbox 2D est une entrée. En prod elle
  viendra d'un détecteur (YOLO ou celui de happypose) ; ici on la trace à la
  main.
- Unités mesh : `run_inference_on_example` suppose des meshes en **mm**
  (`mesh_units="mm"` codé en dur dans `make_object_dataset`). Si la version
  installée diffère, vérifier ce point — un facteur 1000 sur la profondeur
  estimée est le symptôme.
- Si l'estimation est mauvaise : vérifier l'éclairage (le pylône noir sur
  fond sombre est le cas difficile), tenter une bbox plus serrée, ou capturer
  sous un angle qui montre plus de structure 3D du pylône.
