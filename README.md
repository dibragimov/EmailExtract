# Email Extraction Project

This project is used for extracting the last email from the chain of emails. Uses 2 methods - Neural Network (NN) and Heuristics. 
The heuristics part is taken from mailgun's talon project (https://github.com/mailgun/talon) and adapted to take into account some Nordic languages. 
NN model uses LASER embedding and MLP architecture and is trained on some impersonated email data. 
The training for the model is done separately. Then REST API service is deployed that serves the requests using either NN model or the heuristics - it will be 
listening for incoming sentences. Given a sentence, it will embed this
using the endpoint in the embedding service and return the content (and a signature) of the last email.

## Changes 
Signature extraction needs to be improved.


**NEWS**
* 

**CURRENT VERSION:**


## Dependencies
* Python 3.6
* [NumPy](http://www.numpy.org/), tested with 1.15.4
* [Cython](https://pypi.org/project/Cython/), needed by Python wrapper of FastBPE, tested with 0.29.6
* [Faiss](https://github.com/facebookresearch/faiss), for fast similarity search and bitext mining

## Installation
* using port 7010 in Docker. The native port is 5000.

## Applications

## License
Copyright InoviaGroup

## Supported languages

## References
