# AI-Based Smart Image File Searching Algorithm

## Overview

This project implements an **AI-based smart image file searching algorithm** designed to enhance the efficiency and accuracy of image retrieval processes. It leverages cutting-edge machine learning techniques, integrating state-of-the-art computer vision models with advanced similarity search algorithms. The system is particularly beneficial in domains like healthcare, e-commerce, media management, and law enforcement.

## Features

* **AI-Powered Search:** Utilises advanced AI models like **CLIP** (Contrastive Language–Image Pretraining) and **BLIP** (Bootstrapped Language–Image Pretraining).
* **Feature Extraction:** Employs pre-trained deep learning models (CNNs) to extract meaningful features from images.
* **Dynamic Indexing:** Uses techniques like KD-Trees, Ball Trees, or Approximate Nearest Neighbour (ANN) algorithms for quick and scalable retrieval from large datasets.
* **Versatile Applications:** Adaptable to different search queries, considering visual content, metadata, and other attributes.
* **User-Friendly Interface:** Designed with an intuitive GUI using Tkinter for easy query input, search refinement, and result browsing.
* **Image Editing Capabilities:** Incorporates tools for cropping, annotating, and adjusting images directly within the platform, powered by the Pillow library.
* **Automated Description Generation:** Leverages the BLIP model to automatically generate detailed, domain-specific captions for images.

## System Architecture

The system architecture integrates several key components to provide a seamless user experience:

* **CLIP Model:** Enables natural language queries to locate relevant images by embedding both text and images into a unified space.
* **BLIP Model:** Automates the generation of detailed and domain-specific captions for images, ensuring accurate and consistent metadata.
* **User Interface (UI):** An intuitive GUI built with Tkinter, allowing users to input natural language queries, refine search parameters, and browse results with ease.
* **Image Editing Features:** Advanced image editing tools, enabling users to crop, annotate, and adjust images directly within the platform, powered by the Pillow library.

## Implementation Details

* **Programming Language:** Python.
* **GUI Library:** Tkinter.
* **AI Models:**
  * CLIP (Contrastive Language–Image Pretraining).
  * BLIP (Bootstrapped Language–Image Pretraining).
* **Image Processing Library:** Pillow.
* **Key Techniques:**
  * Feature Extraction using CNNs.
  * Dynamic Indexing (KD-Trees, Ball Trees, ANN).
  * Cosine Similarity for comparing feature vectors.

## Requirements

* **Hardware:**
  * **Minimum:** Multi-core processor (Intel i5 or equivalent), 8 GB RAM, 256 GB disk space.
  * **Recommended:** High-performance CPU (Intel i7 or AMD Ryzen 7), 16 GB+ RAM, 512 GB SSD, Dedicated GPU (NVIDIA RTX series).
* **Software:**
  * Python
  * Tkinter
  * Hugging Face Transformers
  * Pillow

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Yogeshknaik/Image-searching-using-AI
    ```
2. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    (Create a `requirements.txt` file listing dependencies like `transformers`, `pillow`, etc.)
3. **Run the application:**
    ```bash
    python main.py
    ```

## Usage

1. **Launch the application** using the command `python main.py`.
2. **Input a query:** You can either upload an image or enter a text query to search for relevant images.
3. **Browse results:** The system will display retrieved images based on the query.
4. **Edit images:** Use the built-in image editing tools to crop, annotate, or adjust the displayed images.
5. **Generate descriptions:** The system can automatically generate detailed captions for the images.

## Testing

* **Unit Tests:** Verify individual components like image pre-processing and feature extraction.
* **Integration Tests:** Ensure different components work together, such as CLIP and BLIP integration with the GUI.
* **System Tests:** Test the entire system under real-world conditions, ensuring accurate image retrieval and meaningful descriptions.
* **Black Box Testing:** Testing the functionality of the algorithm without knowing or inspecting its internal workings or source code.
* **White Box Testing:** Verifying and validating the internal logic of each component of the image analysis algorithm.

## Performance

* **Diagnostic Accuracy:** 94% when analysing medical images.
* **Image Retrieval:** Precision of 92% and recall of 89%.
* **Average Response Time:** 2.3 seconds for querying large datasets and less than 1 second for real-time image analysis.

## Future Enhancements

* **Support for Additional Image Formats:** Expanding support to include DICOM, ultrasound, and nuclear medicine scans.
* **Improved Search Speed:** Optimising the algorithm to reduce search and retrieval time.
* **Incorporating Advanced Deep Learning Models:** Integrating state-of-the-art deep learning models, such as transformer-based architectures.
* **Enhanced Image Editing Features:** Adding advanced editing tools, such as region segmentation and 3D reconstruction.
* **More Detailed Image Descriptions:** Enhancing the system’s description generation capabilities to produce detailed, context-aware summaries.

## Team

* Rushikesh B Kattimani (1OX21CS119)
* Sukshitha S (1OX21CS147)
* Yogesh K N (1OX21CS170)
* Uday Kiran G (1OX22CS423)

## Acknowledgements

We would like to thank:

* Prof. M Ramya Sri, Dept of CSE, for their guidance.
* Dr. S.N.V.L Narasimha Raju, Chairman, for providing infrastructure.
* Dr. H N Ramesh, Principal, for their support.
* Dr. E. Saravana Kumar, Prof and Head of the Department, for their encouragement.
* Department of Computer Science Engineering, The Oxford College of Engineering.

## References

* VISVESVARAYA TECHNOLOGICAL UNIVERSITY BELAGAVI -590018 A Project Report On “AI-Based Smart Image File Searching Algorithm”
* Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., Devin, M., Ghemawat, S., Irving, G., Isard, M., et al. Tensorflow: Asystemforlarge-scale machine learning. In 12th USENIX symposium on operating systems design and implementation (OSDI 16), pp. 265–283, 2016
* Michael Brown, Emily Davis, Robert Lee: Efficient Image Retrieval Using CNN (2022)
* John Doe, Jane Smith, Alice Johnson: Deep Learning-Based Image Retrieval (2023)
