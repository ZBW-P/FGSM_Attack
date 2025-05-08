## Jailbreaking_Deep_Models Experiment details

1. **Dataset Ingestion & Preprocessing**  
   - **Unzip & Inspect**: Extract `TestDataSet.zip` into `./TestDataSet/TestDataSet/` and verify class indices 401–500 via `labels_list.json`.  
   - **Normalization Pipeline**: Compose transforms to convert to tensor and normalize with ImageNet mean/std:  
     - Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225].  
   - **Loader Configuration**: Wrap in `DataLoader(batch_size=32, shuffle=False)` for deterministic evaluation.

2. **Model Instantiation & Device Setup**  
   - **ResNet-34**: Load with `IMAGENET1K_V1` weights, move to GPU, set to eval mode.  
   - **DenseNet-121**: Load with `IMAGENET1K_V1` weights, move to GPU, set to eval mode.  
   - **Performance Tuning**: Enable `torch.backends.cudnn.benchmark = True` for optimized throughput.

3. **Baseline Accuracy Measurement**  
   - Implement `evaluate_topk(model, loader, offset, device)` to compute Top-1 and Top-5 accuracies over the held-out 100-class subset.  
   - **Results on Original Dataset**:  
     - ResNet-34 → Top-1: 76.00%, Top-5: 94.20%  
     - DenseNet-121 → Top-1: 74.60%, Top-5: 93.60%

4. **Adversarial Set Generation & Timing**  
   - **FGSM (ε=0.02)**  
     - **What happens?** A single gradient-sign step perturbs each pixel to maximally increase the loss.  
     - **Why it matters:** It’s extremely fast (~1.8 s) but often yields only a modest drop before stronger attacks are applied.  
   - **PGD (ε=0.02, iters=100)**  
     - **What happens?** Repeated gradient-sign steps project back into the ε-ball, crafting a multi-step “worst-case” perturbation.  
     - **Why it matters:** It takes longer (~122 s) but produces far more potent adversarial examples, driving ResNet-34 to 0% Top-1.  
   - **Random Patch (ε=0.50, size=32, iters=100)**  
     - **What happens?** PGD-style updates restricted to a 32×32 window, leaving the rest of the image untouched.  
     - **Why it matters:** Offers a stealthy, localized attack (~121 s) that still severely degrades performance without global noise.

5. **Robustness Comparison & Insights**  
   - **Accuracy Under Noise**:  
     | Attack    | ResNet-34 (Top-1 / Top-5) | DenseNet-121 (Top-1 / Top-5) |
     |-----------|---------------------------|------------------------------|
     | **FGSM**  | 6.20% / 36.00%            | 63.40% / 89.00%              |
     | **PGD**   | 0.00% / 15.00%            | 65.80% / 91.00%              |
     | **Patch** | 16.40% / 58.40%           | 70.00% / 91.20%              |
   - **Key Insight**: DenseNet-121 consistently outperforms ResNet-34 under all noise attacks, demonstrating markedly superior defense against both global and localized perturbations.  
   - **Next Steps**: Visualize these gaps with comparative bar charts and discuss architectural reasons (e.g., dense connectivity, feature reuse) in your GitHub notebook.  
