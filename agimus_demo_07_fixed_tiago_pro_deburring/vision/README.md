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
   python3 capture_image.py --output ./capture_01
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
   python3 make_megapose_example.py --capture ./capture_01 --mesh ./pylone.ply
   ```

4. **Inférence + visualisation** (PC GPU) :

   ```bash
   python -m happypose.pose_estimators.megapose.scripts.run_inference_on_example \
       pylone --run-inference --vis-outputs
   ```

   Résultats dans `$HAPPYPOSE_DATA_DIR/examples/pylone/` :
   - `outputs/object_data.json` — pose estimée `T_caméra→pylône`
     (quaternion TWO + translation, en m)
   - `visualizations/` — rendu du mesh superposé à l'image : si le contour
     colle au pylône réel, le test est concluant.

Refaire les étapes 1 et 3 pour 3-4 points de vue différents pour juger la
stabilité (`--output ./capture_02`, `--label` reste `pylone`, l'exemple est
écrasé à chaque fois).

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
