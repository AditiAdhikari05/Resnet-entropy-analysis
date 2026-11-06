**Understanding ResNet-18 Through Activation Entropy

How do a neural network’s internal representations evolve as it learns to recognize images? This project investigates that question by training ResNet-18 from scratch and tracking the distribution of activations in its final convolutional layer.

Developed as part of NYU’s Computer Vision coursework, the project combines image classification with an information-theoretic analysis of learning dynamics, examining changes within the network alongside conventional accuracy and loss metrics.

##Experimental approach

- Trained ResNet-18 for 50 epochs, achieving **78.68% validation top-1 accuracy** and **93.28% top-5 accuracy**.
- Tracked joint and conditional activation entropy on a **fixed set of 2,000 validation images**, allowing consistent comparisons across training epochs.
- Examined how activation mass was distributed across **512 feature channels** and across spatial positions within each feature map.
- Used the entropy chain rule to distinguish changes across channels from changes within feature maps.
- Visualized entropy trajectories, variation across images, classification accuracy, and loss to study how representation changes relate to predictive performance.

##Key findings

Joint activation entropy decreased from **13.443 to 12.912 bits**, while conditional entropy decreased from **5.135 to 4.741 bits**. Approximately **74% of the net joint-entropy reduction** came from the conditional component, suggesting that training changed the spatial concentration of activations more strongly than their distribution across channels.

The relationship between entropy and accuracy was not simply inverse. Both entropy measures reached their lowest values early in training and then partially recovered, while classification accuracy continued improving. This demonstrates that, in this experiment, better predictive performance did not require continuously decreasing activation entropy.

Variation in entropy across images also increased relative to initialization, indicating that the trained network produced more varied activation distributions across inputs.

##Implementation and efficiency

The code is organized into separate modules for configuration, data handling, model construction, training, entropy measurement, and visualization. Training used automatic mixed precision with gradient scaling and cuDNN benchmarking. Entropy calculations were performed outside autocast to preserve numerical accuracy.

This project demonstrates practical experience with deep learning experimentation, GPU training, quantitative analysis of neural representations, and careful interpretation of model behavior. The entropy measurements characterize activation distributions; they do not, by themselves, establish what individual channels detect or whether those channels causally contribute to predictions.
