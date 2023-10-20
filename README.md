# Yottixel - A search engine for Digital Histopathology Archives
### Paper Abstract
With the emergence of digital pathology, searching for similar images in large archives has gained considerable attention. Image retrieval can provide pathologists with unprecedented access to the evidence embodied in already diagnosed and treated cases from the past. This paper proposes a search engine specialized for digital pathology, called Yottixel, a portmanteau for “one yotta pixel,” alluding to the big-data nature of histopathology images. The most impressive characteristic of Yottixel is its ability to represent whole slide images (WSIs) in a compact manner. Yottixel can perform millions of searches in real-time with a high search accuracy and low storage profile. Yottixel uses an intelligent indexing algorithm capable of representing WSIs with a mosaic of patches which are then converted into barcodes, called “Bunch of Barcodes” (BoB), the most prominent performance enabler of Yottixel. The performance of the prototype platform is qualitatively tested using 300 WSIs from the University of Pittsburgh Medical Center (UPMC) and 2,020 WSIs from The Cancer Genome Atlas Program (TCGA) provided by the National Cancer Institute. Both datasets amount to more than 4,000,000 patches of 1000 × 1000 pixels. We report three sets of experiments that show that Yottixel can accurately retrieve organs and malignancies, and its semantic ordering shows good agreement with the subjective evaluation of human observers.
### Graphical abstract
![yottixel](image.png)
### Versions
This repository includes two versions of the Yottixel code:
- yottixel_original includes the version that uses only DenseNet as a backbone.
- yottixel_modified includes the version that gives the option to use KimiaNet as a backbone.
### Useful Links
- [Yottixel Paper](https://www.sciencedirect.com/science/article/pii/S1361841520301213)
- [Learn more on Kimia Lab](https://kimialab.uwaterloo.ca/kimia/index.php/data-and-code-2/kimia-net/)
### Disclaimer
Rhazes Lab does not own the code in this repository. This code and data were produced in Kimia Lab at the University of Waterloo. The code is provided as-is without any guarantees, and is stored here as part of Kimia Lab's history. We welcome questions and comments.

Before using or cloning this repository, please read the [End User Agreement](agreement.pdf).