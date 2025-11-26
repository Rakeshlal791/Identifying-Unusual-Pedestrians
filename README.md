# Identifying-Unusual-Pedestrians
An anomaly detection Variational Autoencoder (VAE) to identify unusual pedestrians in the scene.

# Description
Pedestrians may adopt unusual positions due to various reasons, such as medical emergencies, accidents, or disorientation. These individuals face a heightened risk of accidents because of their limited mobility and reduced situational awareness. Detecting such anomalous pedestrians is therefore critical for autonomous driving systems, robotic navigation, and pedestrian safety.

In this work, we define anomalous pedestrians as those exhibiting atypical body poses—such as fallen or lying-down positions—that deviate from normal upright walking or standing postures in traffic environments.

# Dataset
We trained the VAE model on the [European City Person (ECP)](https://https://eurocity-dataset.tudelft.nl/) dataset to learn the latent distribution of normal pedestrians.

## Pedestrian Samples
Our dataset class returns a per-instance bounding box and normalized pose keypoints for training.

The following are some samples from the dataset.

<img src="https://github.com/Rakeshlal791/The-Fallen-Pedestrians/blob/master/images/ped_samples.png?raw=true" width="520">

# Model and Training
A Variational Autoencoder (VAE) is employed to model pedestrian pose and bounding box features, capturing structural information such as bone angles and limb lengths.

We train our model with batch size 2, and for 20 epochs on train and val ECP dataset.

![alt text](https://github.com/Rakeshlal791/The-Fallen-Pedestrians/blob/master/images/vae_model.png?raw=true)

The input feature vector **x** encodes:

$$
\begin{aligned}
&\text{• 14 body bones} \times 2~(\cos\theta, \sin\theta) \Rightarrow 28~\text{values} \\
&\text{• 14 bone lengths (normalized, log-scaled)} \\
&\text{• Bounding box width and height (log-scaled and normalized)}
\end{aligned}
$$

Thus, the total feature dimension is:

$$
\mathbf{x} \in \mathbb{R}^{28 + 14 + 2 = 44}
$$

The encoder network models the latent distribution as

$$
q_{\phi}(z|x),
$$

which approximates a posterior Gaussian for each input sample.  
A standard normal prior,

$$
p(z) = \mathcal{N}(0, I),
$$

is assumed over the latent space, encouraging all encoded representations to cluster around a single global Gaussian distribution.

The decoder network,

$$
p_{\theta}(x|z),
$$

learns to reconstruct the original input feature vector from the latent code, thereby capturing the underlying structure of normal pedestrian poses.  
During training, the VAE minimizes the combined reconstruction and Kullback–Leibler (KL) divergence losses:

$$
\mathcal{L} = \mathbb{E}_{q_{\phi}(z|x)}[\lVert x - \hat{x}\rVert_1] + \beta D_{KL}(q_{\phi}(z|x)\lVert p(z)).
$$

After training on normal pedestrian poses, anomalous poses (e.g., fallen or lying pedestrians) are identified as samples with **high reconstruction error** or **low likelihood** under the learned latent distribution.

# Test Data Generation
We utilized a small subset of the validation split, excluded from training, to obtain representative samples of normal pedestrian behavior. To evaluate anomaly detection performance, we generated synthetic images representing five atypical pedestrian postures: sitting, lying, crawling, bending/leaning, and imbalanced positions.

## Synthetic Data
We generated synthetic anomalous scenarios using [Gemini 2.5 Flash Image (Nano Banana)](https://aistudio.google.com/models/gemini-2-5-flash-image) by performing inpainting to introduce fallen pedestrians into selected images from the test dataset. Any visual artifacts produced during synthesis were manually corrected using the GNU Image Manipulation Program (GIMP). Finally, we annotated the pose keypoints and bounding boxes for each anomalous pedestrian.

We generated approximately 50 synthetic test images per category, annotated, and applied small Gaussian perturbations to pose keypoints, introducing subtle variations that better capture real-world posture diversity.

Below are some examples of our synthetic pedestrians for each anomalous category.

| Category | Examples |
|-----------|-----------|
| **Sitting** | <img src="https://github.com/Rakeshlal791/The-Fallen-Pedestrians/blob/master/images/sitting_row.png?raw=true" width="520"> |
| **Bending** | <img src="https://github.com/Rakeshlal791/The-Fallen-Pedestrians/blob/master/images/bending_row.png?raw=true" width="520"> |
| **Crawling** | <img src="https://github.com/Rakeshlal791/The-Fallen-Pedestrians/blob/master/images/crawling_row.png?raw=true" width="520"> |
| **Lying** | <img src="https://github.com/Rakeshlal791/The-Fallen-Pedestrians/blob/master/images/lying_row.png?raw=true" width="520"> |
| **Imbalanced** | <img src="https://github.com/Rakeshlal791/The-Fallen-Pedestrians/blob/master/images/imbalanced_row.png?raw=true" width="520"> |

---

## Results
To evaluate our model, we used approximately **1.2k non-anomalous** and **1.2k anomalous** pedestrian samples.  
We visualize the learned latent representations ($$\mu$$) using **Principal Component Analysis (PCA)**, comparing the distributions of normal and anomalous data in the latent space.

### Anomaly Score
The anomaly score for each sample is computed as the **weighted mean reconstruction error** of each feature (head, left limb, right limb, torso, left leg, right leg, bounding box width, and height):

$$\text{score}_i = \frac{1}{D} \sum_{d=1}^{D} w_d \big| x_{i,d} - \hat{x}_{i,d} \big|$$

where:
- $$D$$ — total number of feature dimensions (44)  
- $$w_d$$ — region-wise weight for the $$d^{th}$$ feature  
- $$x_{i,d}$$ — original feature value  
- $$\hat{x}_{i,d}$$ — reconstructed feature value  

Higher scores indicate poses that deviate more strongly from the learned “normal” distribution.

<img src="https://github.com/Rakeshlal791/The-Fallen-Pedestrians/blob/master/images/latent_mu.png?raw=true" width="520">

We compute the appropriate threshold for anomaly score from the AUC and report the following metrics on model performance.

| Metric | Value |
|-----------|-----------|
| **Precision** | 0.994 |
| **Recall** | 0.992 |  
| **F1** | 0.993 |  
| **AUC** | 0.998 |

---
| Confusion Matrix | Anomaly Score distribution |
|-----------|-----------|
| <img src="https://github.com/Rakeshlal791/The-Fallen-Pedestrians/blob/master/images/confusion_matrix.png?raw=true" height="500"> | <img src="https://github.com/Rakeshlal791/The-Fallen-Pedestrians/blob/master/images/vae_score.png?raw=true" height="500"> |

---



