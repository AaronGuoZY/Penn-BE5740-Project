# Penn-BE5740-Project

- This is a class project modelled after the following publication:

<!--bibtex-->
@article{XIA2021102169,
title = {Learning to synthesise the ageing brain without longitudinal data},
journal = {Medical Image Analysis},
volume = {73},
pages = {102169},
year = {2021},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2021.102169},
url = {https://www.sciencedirect.com/science/article/pii/S1361841521002152},
author = {Tian Xia and Agisilaos Chartsias and Chengjia Wang and Sotirios A. Tsaftaris},
keywords = {Brain ageing, Generative adversarial network, Neurodegenerative disease, Magnetic resonance imaging (MRI)},
abstract = {How will my face look when I get older? Or, for a more challenging question: How will my brain look when I get older? To answer this question one must devise (and learn from data) a multivariate auto-regressive function which given an image and a desired target age generates an output image. While collecting data for faces may be easier, collecting longitudinal brain data is not trivial. We propose a deep learning-based method that learns to simulate subject-specific brain ageing trajectories without relying on longitudinal data. Our method synthesises images conditioned on two factors: age (a continuous variable), and status of Alzheimer’s Disease (AD, an ordinal variable). With an adversarial formulation we learn the joint distribution of brain appearance, age and AD status, and define reconstruction losses to address the challenging problem of preserving subject identity. We compare with several benchmarks using two widely used datasets. We evaluate the quality and realism of synthesised images using ground-truth longitudinal data and a pre-trained age predictor. We show that, despite the use of cross-sectional data, our model learns patterns of gray matter atrophy in the middle temporal gyrus in patients with AD. To demonstrate generalisation ability, we train on one dataset and evaluate predictions on the other. In conclusion, our model shows an ability to separate age, disease influence and anatomy using only 2D cross-sectional data that should be useful in large studies into neurodegenerative disease, that aim to combine several data sources. To facilitate such future studies by the community at large our code is made available at https://github.com/xiat0616/BrainAgeing.}
}
