# TexStFusion

**TexStFusion: A Controllable Diffusion Model using Textural, Structural, and Textual Feature Fusion**

📄 *Paper*: [TexStFusion](https://link.springer.com/article/10.1007/s11760-025-04367-2)
🧠 *Area*: Controllable Text-to-Image Diffusion
🏗️ *Backbones*: Stable Diffusion v1.5 / v2.1 / SDXL

---

## 🔍 Overview

Text-to-Image (T2I) diffusion models generate highly realistic images from text prompts, but often lack **fine-grained controllability** when prompts become complex. Existing controllable diffusion methods typically rely **only on structural visual cues** (e.g., edges or keypoints), which limits texture fidelity and content composability.

**TexStFusion** introduces a **feature-level fusion framework** that jointly conditions generation on:

* **Textural details** from a *content image*
* **Structural + textual details** from a *condition image + prompt*
* A **frozen, pre-trained T2I diffusion model**

Crucially, this is achieved **without fine-tuning the base diffusion model**, preserving its generative prior while enabling precise control.

This figure shows the sample inputs and generated images from the model.
![alt text](https://github.com/SuhasHegde25/SubjectNet/blob/main/R_Cover_New%20(1).jpeg)
---

## ✨ Key Contributions

* **Tri-modal Conditioning**
  Simultaneous integration of **textural, structural, and textual features** for controllable image generation.

* **Texture–Structure Feature Fusion**
  A novel **Fuser module** that combines texture and structure-text feature maps with tunable guidance scales.

* **Unit Convolution Layer**
  A new convolution initialization strategy designed to preserve and emphasize **high-frequency texture details**.

* **Content-Composable Generation**
  Enables fine-grained control over *what* is generated and *how* it appears.

* **Small-Dataset Robustness**
  Achieves strong performance with as few as **30k training images**.

---

## 🧠 Method Overview

TexStFusion decomposes image information into two complementary components:

1. **TextureNet**

   * Encodes *textural details* from a **content image**
   * Uses **Unit Convolution** to allow all features to pass initially and gradually suppress irrelevant ones

2. **StructureNet**

   * Encodes *structural + textual information* from a **condition image + text prompt**
   * Uses **Zero Convolution** (as in ControlNet-style adapters)

3. **Fuser Module**

   * Linearly combines feature maps from TextureNet and StructureNet
   * Controlled by **guidance scales**:
     [
     E' = G_s \cdot F_{\text{texture}} + G_c \cdot F_{\text{structure-text}}
     ]
   * The fused map is injected into the **decoder blocks of a frozen diffusion U-Net**

📌 *Result*: controllable generation without sacrificing realism or diversity.

---

## 🏗 Architecture

![TexStFusion Architecture](https://github.com/SuhasHegde25/TexStFusion/blob/main/TexStFusion_Architecture.png)

**Pipeline**:

1. Content Image → TextureNet → Texture Map
2. Condition Image + Prompt → StructureNet → Structure-Text Map
3. Feature Fusion via Fuser
4. Injection into frozen Stable Diffusion decoder

> See **Figure 2** in the paper for the complete architecture diagram .

---

## 📊 Experimental Results

TexStFusion is evaluated on **15 derived datasets** using:

* **Canny edges**
* **HED maps**
* **Facial keypoints**

### Metrics

* **FID** – image realism
* **SSIM** – structural adherence
* **CLIP-T** – prompt fidelity

### Highlights

* ✅ Best or near-best **FID on 10/15 datasets**
* ✅ Best **SSIM on 13/15 datasets**
* ✅ Strong **CLIP-T × SSIM trade-off**, indicating superior content composability

This figure shows the comparison of our model with the state-of-the-art methods.
![alt text](https://github.com/SuhasHegde25/TexStFusion/blob/main/TexStFusion_qual_comparison.jpeg)

Compared against:

* ControlNet
* T2I-Adapter
* UniControlNet
* Composer

📈 Detailed quantitative tables are provided in Section 5 of the paper .

---

## ⚙️ Training Strategy

* **Two-stage modular training**:

  * Train **TextureNet** and **StructureNet** separately
* Allows:

  * Flexible guidance scales at inference
  * Easy swapping of condition types (e.g., edges → keypoints)

**Key Hyperparameters**

* Optimizer: AdamW
* Learning rate: `1e-5`
* Epochs: `3`
* Batch size: `4`
* Sampler: DDIM (20 steps)
* CFG scale: `7.5`

---

## 🧪 Ablation Studies

The paper includes extensive ablations on:

* **Unit vs Zero Convolution**
  → Unit convolution captures high-frequency texture significantly better.

* **Architecture Variants**
  → Texture-only or structure-only models fail to balance fidelity and control.

* **Fuser Guidance Scales**
  → Optimal composability observed for:
  [
  G_s, G_c in [0.3, 0.7]
  ]
The users can control the percentage of features to be incorporated in the generated image using guidance scales
![alt text](https://github.com/SuhasHegde25/SubjectNet/blob/main/CelebGuidanceAblation%20(1).png)
Text prompt = A woman with blonde hair
---

## ⚠️ Limitations

* Slight computational overhead due to dual encoders
* Sensitive to **semantic mismatch** between content image and prompt
* Limited ability to reason about **quantities** (e.g., “two dogs”)

These are discussed in Section 5.4 of the paper .

---

## 🎯 Applications

* Precise image data augmentation
* Controlled image generation
* Face editing and styling
* Creative content generation

---

## 📚 Citation

```bibtex
@article{Hegde2025TexStFusion,
  title   = {TexStFusion: A controllable diffusion model using textural, structural, and textual feature fusion},
  author  = {Hegde, Suhas and Tiwari, Aruna},
  journal = {Signal, Image and Video Processing},
  year    = {2025},
  volume  = {19},
  pages   = {741},
  doi     = {10.1007/s11760-025-04367-2},
  publisher = {Springer}
}

```




