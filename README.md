## Experiment Workflow in Five Elaborate Tasks

1. **Dataset Ingestion & Preprocessing**  
   - **Unzip & Inspect**: Extract `TestDataSet.zip` into `./TestDataSet/TestDataSet/` and verify class indices 401–500 via `labels_list.json`.  
   - **Normalization Pipeline**: Compose transforms to convert to tensor and normalize with ImageNet mean/std:  
     ```python
     transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
     ])
     ```  
   - **Loader Configuration**: Wrap in `DataLoader(batch_size=32, shuffle=False)` for deterministic evaluation.

2. **Model Instantiation & Device Setup**  
   - **ResNet-34**:  
     ```python
     resnet34 = models.resnet34(weights='IMAGENET1K_V1').to(device).eval()
     ```  
   - **DenseNet-121**:  
     ```python
     densenet121 = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1).to(device).eval()
     ```  
   - **Performance Tuning**: Enable `torch.backends.cudnn.benchmark = True` for throughput.

3. **Baseline Accuracy Measurement**  
   - Implement `evaluate_topk(model, loader, offset, device)` to compute Top-1/Top-5.  
   - Run on original dataset:  
     - ResNet-34 → Top-1: 76.00%, Top-5: 94.20%  
     - DenseNet-121 → Top-1: 74.60%, Top-5: 93.60%

4. **Adversarial Set Generation & Timing**  
   - **FGSM (ε=0.02)** via `make_fgsm()` → generation time ~1.82 s  
   - **PGD (ε=0.02, iters=100)** via `make_pgd()` → ~122.51 s  
   - **Random Patch (ε=0.50, size=32, iters=100)** via `make_patch()` → ~120.95 s  
   - Clamp perturbed images within L∞ ball and verify norm constraints.

5. **Robustness Comparison & Insights**  
   - **Accuracy Under Noise**:  
     | Attack  | ResNet-34 (%) | DenseNet-121 (%) |
     |---------|---------------|------------------|
     | **FGSM**| Top-1: 6.20   | Top-1: 63.40     |
     |         | Top-5: 36.00  | Top-5: 89.00     |
     | **PGD** | Top-1: 0.00   | Top-1: 65.80     |
     |         | Top-5: 15.00  | Top-5: 91.00     |
     | **Patch**| Top-1: 16.40  | Top-1: 70.00     |
     |         | Top-5: 58.40  | Top-5: 91.20     |
   - **Key Insight**: Across all perturbation types—FGSM, PGD, and localized patches—DenseNet-121 consistently retains substantially higher Top-1 and Top-5 accuracies than ResNet-34, demonstrating superior noise-defense capacity.  
   - **Visualization & Reporting**: Embed comparative bar charts and annotated tables in your GitHub-hosted Jupyter notebook to emphasize the resilience gap.  

