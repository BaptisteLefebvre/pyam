# Références

## Articles
- [Eulerian video magnification for revealing subtle changes in the world](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
- [Phase-based video motion processing](http://delivery.acm.org/10.1145/2470000/2461966/a80-wadhwa.pdf?ip=134.157.180.205&id=2461966&acc=OA&key=7EBF6E77E86B478F%2EA72B4D473219EA0C%2E4D4702B0C3E38B35%2E8A13F40887C8CA9E&__acm__=1527881263_0adb3bbf2bbe7783b80ab66fa966b11a)
- [Mechanical surface waves accompany action potential propagation](https://www.nature.com/articles/ncomms7697.pdf)
- [Motion microscopy for visualizing and quantifying small motions](http://www.pnas.org/content/pnas/early/2017/10/11/1703715114.full.pdf)
- [Imaging Action Potential in Single Mammalian Neurons by Tracking the Accompanying Sub-Nanometer Mechanical Motion](https://pubs.acs.org/doi/abs/10.1021/acsnano.8b00867)


## Liens
- [Video Magnification](https://people.csail.mit.edu/mrub/vidmag/#code)
- [Eulerian Video Magnification for Revealing Subtle Changes in the World](http://people.csail.mit.edu/mrub/evm/#code)


# Data
`testmovie kept stack aligned.tif`: original file (i.e. data)


# TODO :
 - [?] Générer une vidéo sans les basses fréquences.
 - [?] Générer une vidéo sans les hautes fréquences.
 - [?] Séparer hautes, moyennes et basses fréquences.
 - [ ] Générer deux vidéos (amplitude & phase) pour chacun des filtres de la
       pyramide.


# Questions :
 - Combien de temps à durée l'enregistrement de 'testmovie kept stack aligned.tif' ?
 - Quelle est l'activité des neurones pendant l'enregistrement ?
   Quelle sont les événements neuronaux qui ont eu lieu pendant l'enregistrement ?
   (activité spontanée, nombres de potentiels d'actions, nombre de "frames" par potentiel d'action)
   Peut-on utiliser un marqueur pour avoir plus d'informations sur l'activité des neurones ?
 
 - Comment sont "flashés" les neurones (puissance, laser, ...) ?
 - Quel est la dye calcique utilisée (i.e. constante de temps de 30 s) ?
 - Pourquoi imager à 20 Hz et non à 60 Hz (constante de temps des variations mécaniques) ?
 - Expériences avec activité spontanée (i.e. sans flash) ?


# Réponses :
 - Les tâches noires qui se déplacent le long des neurones sont probablement des vésicules.
 - La vidéo a été enregistrée à 50 frame par seconde.
 - Pour voir la propagation d'un potentiel d'action (1 mètre par seconde) il faudrait augmenter la fréquence d'échantillonage (i.e. utiliser une caméra à grande vitesse).
 - Pour pouvoir quantifier les modifications de volumes potentiellement détectées, adapter l'algorithme de magnification à la présence de débris + vérifier que la résolution de la vidéo permet de détecter ces modifications.
 - Pour pouvoir attester des événements neuronaux qui ont eu lieu, des mesures optogénétiques vont être effectuées.
 - La but ultime (graâl), observer la propagation des potentiels d'actions au sein d'un réseau de neurones.

# Idées :
 - Utiliser des algorithmes de biologie pour enlever les débris (probablement classique).
 - Utiliser des algorithmes de super-résolution en pré-processing.
 - Adapter l'algorithme de magnification eulérienne pour son application en présence de débris.

# Notes
 - 1 m/s pour la propagation électrique dans l'axone.
