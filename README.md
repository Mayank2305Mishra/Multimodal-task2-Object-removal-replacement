Lightweight Object-Level Image Editing on Edge Devices

> **Segmentation, Object Removal, Inpainting, and Exploratory Object
> Replacement**

This project studies lightweight, CPU-oriented image editing pipelines
for text-guided object removal on edge devices. The system
decomposes object removal into three practical stages:

1. Target localization / grounding
2. Instance segmentation and mask generation
3. Image reconstruction / inpainting

The project also contains a limited exploration of object
replacement and lightweight generative editing. Replacement was not
benchmarked as extensively as object removal because of development-time
constraints.

────────

1. Project Overview

The central goal is to build an object-editing system that can operate
under constrained compute while maintaining a useful balance between:

• Mask quality
• Image reconstruction quality
• CPU latency
• Execution reliability
• Model size and deployment complexity
• Compatibility between model interfaces

The experimental pipeline is modular:

                  Natural-language prompt
                           │
                           ▼
                 ┌─────────────────────┐
                 │ Target localization │
                 │ / optional grounding│
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Segmentation     │
                 │ MobileSAM /         │
                 │ EfficientSAM-S /    │
                 │ YOLO26n-seg         │
                 └──────────┬──────────┘
                            │
                         Binary mask
                            │
                            ▼
                 ┌─────────────────────┐
                 │     Inpainting      │
                 │        LaMa         │
                 │     / MI-GAN        │
                 └──────────┬──────────┘
                            │
                            ▼
                     Edited image

Grounding DINO-Tiny can be inserted before segmentation when explicit
text grounding is required.

────────

2. Architectures Evaluated

Eight executable architecture variants were evaluated on a ten-image
COCO subset.

────────

Architecture              Success       Mean IoU      Mean Dice     Mean Total
Latency

────────

YOLO26n-seg + LaMa            60%          0.847          0.916        23.25 s

MobileSAM + LaMa              90%          0.810          0.892    16.63 s

EfficientSAM-S +              90%          0.827          0.903        25.74 s
LaMa

DINO +                        60%          0.847          0.916        46.06 s
YOLO26n-seg + LaMa

DINO + MobileSAM +           100%          0.805          0.885        50.32 s
LaMa

DINO +                       100%          0.814          0.891        62.66 s
EfficientSAM-S +
LaMa

DINO +                       100%          0.814          0.891        61.69 s
EfficientSAM-S +
adaptive LaMa

DINO +                         0%            —            —            — EfficientSAM-S + MI-GAN

Important: latency values are means over successful executions.
Success rate is calculated over all attempted images. Therefore, quality
and latency should always be interpreted together with execution
reliability.

Main observations

• MobileSAM + LaMa had the lowest mean successful end-to-end
latency among the reliable tested pipelines.
• EfficientSAM-S + LaMa improved mean mask quality over MobileSAM,
but increased segmentation cost substantially.
• YOLO26n-seg + LaMa produced the highest successful-run IoU/Dice
values, but only 60% of attempts completed.
• Grounding DINO-Tiny increased execution cost considerably. It
was useful as an explicit text-grounding stage, but it did not
produce a corresponding improvement in measured mask quality for the
tested configurations.
• MI-GAN could not be fairly benchmarked because all ten runs
failed at the ONNX interface due to a mask-rank mismatch.

────────

3. Benchmark Dataset

The main benchmark uses a fixed subset of 10 COCO validation images.

For quantitative evaluation, the benchmark compares the predicted
segmentation mask against the corresponding COCO instance mask.

The benchmark records:

• Segmentation / grounding latency
• Inpainting latency
• Total latency
• Mask area
• Mask IoU
• Dice coefficient
• Change score
• Success / failure
• Error message for failed runs

The complete benchmark contains 80 result records:

8 architectures × 10 images = 80 attempted runs

────────

4. Evaluation Metrics

Intersection over Union

For predicted mask P and ground-truth mask G:

IoU = |P ∩ G| / |P ∪ G|

Higher IoU indicates stronger overlap between the predicted and
reference masks.

Dice coefficient

Dice = 2|P ∩ G| / (|P| + |G|)

Dice is particularly useful for measuring segmentation overlap when
foreground regions are relatively small.

Success rate

Success Rate = Successful Runs / Total Attempts

A high segmentation score with a low success rate should not be treated
as a reliable end-to-end system.

Latency

Latency is measured separately for:

• Grounding / segmentation
• Inpainting
• Total pipeline execution

This separation makes it possible to identify the actual deployment
bottleneck.

────────

5. Repository / Output Structure

A typical project layout is:

.
├── README.md
├── notebooks/
│   └── benchmark_notebook.ipynb
│
├── models/
│   ├── mobilesam/
│   ├── efficientsam/
│   ├── yolo/
│   ├── grounding_dino/
│   └── lama/
│
├── pipeline/
│   ├── segmentation/
│   ├── grounding/
│   ├── inpainting/
│   └── object_removal/
│
├── outputs/
│   ├── masks/
│   ├── removed/
│   ├── visualizations/
│   └── benchmark/
│
├── figures/
│   ├── 01_mean_total_latency.png
│   ├── 02_stage_latency.png
│   ├── 03_mask_quality.png
│   ├── 04_quality_vs_latency.png
│   ├── 05_success_rate.png
│   └── 06_per_image_latency.png
│
├── benchmark/
│   ├── architecture_summary.csv
│   └── all_benchmark_results.csv
│
└── report/
    ├── Anonymous_Edge_Object_Editing_IEEE_Report.tex
    └── Anonymous_Edge_Object_Editing_IEEE_Report.pdf

Exact filenames can vary depending on the notebook or deployment
environment.

────────

6. Installation

The project is intended for CPU-oriented experimentation.

Create a clean virtual environment:

python -m venv .venv
source .venv/bin/activate

On Windows:

python -m venv .venv
.venv\Scripts\activate

Install the project dependencies from the provided requirements file if
available:

pip install -r requirements.txt

For notebook-based execution:

pip install jupyter
jupyter notebook

Dependency isolation

Because this project combines PyTorch, ONNX Runtime, PIL/Pillow, NumPy,
OpenCV, scikit-image, and model-specific packages, a clean environment
is strongly recommended.

In particular, avoid mixing incompatible system-wide installations of:

• NumPy
• Pillow
• pandas
• scikit-image
• ONNX Runtime
• PyTorch

If binary compatibility errors occur, recreate the virtual environment
rather than repeatedly upgrading individual packages.

────────

7. Model Preparation

The project uses lightweight model variants where possible.

Segmentation

Supported experimental segmentation routes include:

• MobileSAM
• EfficientSAM-S
• YOLO26n-seg

Text grounding

Optional:

• Grounding DINO-Tiny

Grounding DINO is useful when the prompt must first be converted into an
explicit target region.

Inpainting

The main reconstruction model is:

• LaMa

MI-GAN was also investigated as a lightweight alternative, but its
benchmark path failed because of an ONNX mask input-rank mismatch.

────────

8. Basic Object Removal Pipeline

Conceptually, a removal request follows:

image = load_image(...)
prompt = "remove the person"

target = localize_target(image, prompt)

mask = segment_target(
    image,
    target
)

result = lama_inpaint(
    image,
    mask
)

save_image(result, ...)

The important engineering requirement is that the image and mask passed
to the inpainting model have compatible spatial dimensions and tensor
ranks.

Before inference, validate:

image dimensions == mask dimensions
mask rank matches ONNX contract
mask contains a valid foreground region
bounding box is inside image bounds
target detection is not empty

────────

9. LaMa Input Contract

The benchmark uses a LaMa ONNX interface with a fixed reconstruction
input size.

The expected conceptual tensor layout is:

Image:
[1, 3, 512, 512]

Mask:
[1, 1, 512, 512]

The output is converted back into an RGB image.

The mask must therefore be resized and reshaped correctly before ONNX
inference.

A common failure mode is accidentally providing:

[H, W, 1]

instead of:

[1, 1, H, W]

or passing an image and mask with different spatial dimensions.

────────

10. EfficientSAM Integration

EfficientSAM uses a different decoder interface from the naive
point-input names that may appear in some implementations.

The tested ONNX decoder interface uses inputs conceptually corresponding
to:

image_embeddings
batched_point_coords
batched_point_labels
orig_im_size

When integrating the model, inspect the ONNX graph first:

for inp in session.get_inputs():
    print(inp.name, inp.shape, inp.type)

Do not assume that inputs are named point_coords or point_labels.

────────

11. Benchmarking

Run the benchmark notebook and record each architecture independently.

The benchmark should measure:

1. preprocessing
2. grounding
3. segmentation
4. inpainting
5. total execution

For each image, store:

architecture
image id
success
segmentation_ms
grounding_score
inpainting_ms
total_ms
mask_area_px
mask_area_pct
change_score
iou
dice
error

This makes it possible to distinguish model-quality problems from
integration failures.

────────

12. Failure Analysis

Several failures were observed during benchmarking.

YOLO26n-seg

Some runs failed because the generated mask and original image had
incompatible shapes, for example:

(427, 640, 3) vs (448, 640, 1)

and:

(408, 640, 3) vs (416, 640, 1)

There was also a no-target case.

These failures indicate that post-processing must restore masks to the
original image coordinate system before compositing or metric
calculation.

MobileSAM / EfficientSAM-S

Each standalone architecture had one no-target failure in the ten-image
benchmark.

MI-GAN

All ten MI-GAN runs failed with an ONNX interface error equivalent to:

Invalid rank for input: mask
Got: 3
Expected: 4

This was treated as an implementation/interface failure rather than a
measured image-quality result.

────────

13. Object Replacement

Object replacement was explored as an extension of the removal pipeline.

The general structure is:

Input image
     │
     ▼
Target detection
     │
     ▼
Target mask
     │
     ├───────────────┐
     ▼               ▼
Object removal   Replacement generation
     │               │
     └───────┬───────┘
             ▼
          Composite

A single-image edge-oriented test measured approximately:

LaMa reconstruction       : 6.926 s
Replacement compositing   : 0.006 s
Total                     : 6.932 s

This measurement was separate from the 80-record benchmark.

The experiment suggests that reconstruction is substantially more
expensive than simple compositing.

Scope limitation

Extensive object-replacement experiments were not completed because of
time constraints. Therefore, replacement should be considered an
exploratory extension rather than a fully benchmarked component of the
project.

────────

14. ControlNet-Lite / Generative Extensions

ControlNet-Lite was investigated as a possible intermediate solution
between deterministic inpainting and heavier diffusion-based editing.

The exact original ControlNet-Lite checkpoint/configuration interface
was not validated sufficiently for a fair end-to-end benchmark, so it is
not presented as a numerical benchmark result.

The intended future direction is:

Prompt
  │
  ▼
Target mask / structural condition
  │
  ▼
Lightweight controlled generation
  │
  ▼
Object replacement

Generative replacement should remain a selective path because
diffusion-style reconstruction is substantially more expensive than
deterministic LaMa-based removal.

────────

15. Edge Optimization Strategy

The benchmark indicates that the major optimization targets are:

1. Reduce segmentation overhead

MobileSAM provides a useful low-latency baseline.

EfficientSAM-S provides stronger mask quality but at higher segmentation
cost.

2. Reduce LaMa cost

LaMa remains a major component of total inference time.

Useful future optimizations include:

• Cropped/local inpainting
• Smaller reconstruction regions
• Quantization
• ONNX graph optimization
• Hardware-specific acceleration
• Reduced preprocessing overhead

3. Avoid unnecessary grounding

Grounding DINO can add tens of seconds to CPU execution.

A practical system can therefore route requests:

Unambiguous target
      │
      ▼
Lightweight segmentation
      │
      ▼
LaMa

Ambiguous text target
      │
      ▼
Grounding DINO
      │
      ▼
Segmentation
      │
      ▼
LaMa

4. Use generative models selectively

Diffusion or controlled generation should be activated only when the
user explicitly requests synthesis or replacement rather than simple
removal.

────────

16. Recommended Deployment Architecture

A modular edge-oriented deployment can use:

                    User prompt
                        │
                        ▼
                 Prompt classifier
                        │
             ┌──────────┴──────────┐
             │                     │
       Simple removal        Ambiguous target
             │                     │
             ▼                     ▼
         MobileSAM          Grounding DINO
             │                     │
             │              MobileSAM /
             │              EfficientSAM
             │                     │
             └──────────┬──────────┘
                        ▼
                       Mask
                        │
                        ▼
                      LaMa
                        │
                        ▼
                 Removed image
                        │
             Optional replacement
                        │
                        ▼
              Generative branch

This avoids paying the computational cost of the most expensive models
for every request.

────────

17. Benchmark Outputs

The project produces the following main visual analyses:

Mean total latency

01_mean_total_latency.png

Compares mean successful end-to-end latency across architectures.

Stage latency

02_stage_latency.png

Separates segmentation/grounding cost from inpainting cost.

Mask quality

03_mask_quality.png

Compares IoU and Dice scores.

Quality vs latency

04_quality_vs_latency.png

Shows the practical trade-off between segmentation quality and execution
cost.

Success rate

05_success_rate.png

Shows how often each architecture completed successfully.

Per-image latency

06_per_image_latency.png

Shows latency variation across individual benchmark images.

────────

18. Reproducibility

For reproducible experiments:

1. Use a fixed COCO image subset.
2. Keep the same target-selection protocol.
3. Record all successful and failed runs.
4. Report success rate separately from quality.
5. Measure stages independently.
6. Keep model versions and ONNX files fixed.
7. Run on the same CPU environment when comparing latency.
8. Avoid comparing absolute latency across different hardware.
9. Save per-image benchmark records.
10. Preserve failure messages rather than silently skipping failed runs.

The benchmark artifact contains:

architecture_summary.csv
all_benchmark_results.csv

These files should be treated as the source of truth for the reported
numerical results.

────────

19. Results Summary

The experiments demonstrate that an edge object-removal system is
constrained by more than segmentation accuracy.

The main engineering findings are:

• Lightweight segmentation can provide useful mask quality without
requiring large segmentation models.
• LaMa is a major reconstruction-time bottleneck.
• Grounding DINO adds substantial CPU latency.
• Higher mask quality does not automatically translate into a more
reliable end-to-end pipeline.
• Tensor shape and ONNX interface compatibility are critical
deployment concerns.
• Failure handling is part of model selection, not merely an
implementation detail.
• Cropped inpainting is a promising direction for reducing
reconstruction latency.
• Object replacement is feasible as an extension, but the current
project does not provide a comprehensive replacement benchmark.

────────

20. Report

The project includes an anonymous IEEE-style report:

Lightweight Object-Level Image Editing on Edge Devices: Segmentation,
Removal, and Replacement

The report contains:

• System architecture
• Experimental protocol
• Architecture comparison
• Latency analysis
• Mask-quality analysis
• Success-rate analysis
• Failure analysis
• Visual results
• LaMa profiling
• Object-replacement discussion
• ControlNet-Lite discussion
• Deployment strategy
• Reproducibility notes

────────

21. Citation / References

The project builds on the following research directions:

• Segment Anything (SAM)
• MobileSAM
• EfficientSAM
• MobileCLIP
• Grounding DINO
• LaMa image inpainting
• Stable Diffusion
• Latent Consistency Models
• LoRA
• MiDaS

The accompanying IEEE report contains the detailed bibliography and
attribution.

────────

22. Limitations

This project is an experimental edge-oriented benchmark rather than a
production-ready image editor.

Important limitations include:

• Only ten COCO validation images were used for the main benchmark.
• CPU latency is hardware-dependent.
• Some architectures had integration failures.
• MI-GAN could not be fairly evaluated because of its ONNX mask
interface mismatch.
• YOLO26n-seg had mask-shape and no-detection failures.
• Object replacement was only partially explored.
• ControlNet-Lite was not included in the numerical benchmark because
its exact pretrained interface could not be validated sufficiently.
• The benchmark focuses primarily on object removal rather than full
generative photo editing.

────────

23. Future Work

The next iteration should focus on:

1. Cropped LaMa inference around the target bounding box.
2. INT8 / FP16 optimization where supported by the target hardware.
3. Robust mask resizing and coordinate restoration for YOLO-based
segmentation.
4. Better no-detection fallback logic.
5. Quantitative perceptual metrics for the reconstructed image.
6. Larger and more diverse benchmark sets.
7. Actual edge-device profiling on ARM / Apple Silicon / mobile
hardware.
8. Efficient object replacement using controlled generative models.
9. Prompt-aware routing between fast and high-quality pipelines.
10. Memory and model-size profiling alongside latency.

────────

24. Project Takeaway

The main conclusion is that practical edge image editing should be
treated as a modular systems problem rather than simply selecting
the largest or most accurate model.

A reliable design separates:

Target understanding
        +
Mask generation
        +
Image reconstruction
        +
Failure handling
        +
Deployment optimization

The experiments show that MobileSAM + LaMa provides a useful
low-latency baseline, EfficientSAM-S + LaMa provides a
higher-mask-quality alternative at additional cost, and YOLO26n-seg +
LaMa demonstrates the importance of robust post-processing and failure
handling.

For future deployment, the most promising direction is an adaptive
edge pipeline that uses lightweight segmentation by default, invokes
explicit grounding only when necessary, applies optimized/cropped LaMa
for removal, and activates generative replacement only for requests that
actually require synthesis.
