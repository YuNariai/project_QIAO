## Project_QIAO
## Project Name:
**Apply Random Quantum Neural Network on Noisy Medical Image Recognition**

## Team Name:
**QIAO**


## Team Members

* Xiaoran Li, x832li@uwaterloo.ca 
* Giuliana Siddi Moreau,  julie@crs4.it
* FUTe Wong, zuxfoucault@gmail.com
* David Liu, davidliulearn@gmail.com


## Summary
This project modifies Random Quantum Neural Networks (RQNN) (published March 2022) with a new quantum encoder and incorporates NVIDIA cuQuantum for running at run.ai for better performance. We apply our new QML model to PneumoniaMNIST dataset with added either Gaussian or Salt and Pepper noise model. For both noise models, our model achieved around 75% accuracy which is better than classical random neural networks.


## References
- [1] "Random Quantum Neural Networks (RQNN) for Noisy Image Recognition," by Debanjan Konar, et al March 3, 2022, https://deepai.org/publication/random-quantum-neural-networks-rqnn-for-noisy-image-recognition
- [2] "MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification," Jiancheng Yang. et al, Sept 25, 2022, https://arxiv.org/pdf/2110.14795.pdf
- [3] "PennyLane by Xanadu", https://pennylane.ai/qml/

## Main Contributions
This project makes the following contributions
- add quantum encoder to RQNN to reduce the number of qubits required
- incorporate cuQuantum SDK for running in run.AI
- Apply the model to larger and more complex dataset PneumoniaMNIST
- Apply both Gaussian, Salt and Pepper models noise models
- Achieve 75% accuracy on test dataset.



## Program Files
- RQNN Model with GPU: data_encoder_train_gpu.ipynb
- RQNN Model without GPN: data_encoder_train_nogpu.ipynb
- Running Envionment: requirements.txt
- Results: gpu_result_with_Qencoder.ipynb
- 




## Future Plan
 
