## Experiment Workflow in 5 Tasks

1. **Data Preparation**  
   - Unzip `TestDataSet.zip`  
   - Load images with `ImageFolder` and apply ImageNet normalization  

2. **Model Initialization**  
   - Instantiate `resnet34` and `densenet121` with ImageNet-pretrained weights  
   - Move models to GPU if available  

3. **Baseline Evaluation**  
   - Run `evaluate_topk` on original dataset  
   - Record Top-1 and Top-5 accuracies  

4. **Adversarial Set Generation**  
   - **FGSM**: ε=0.02 → `make_fgsm`  
   - **PGD**: ε=0.02, iters=100 → `make_pgd`  
   - **Random Patch**: ε=0.5, size=32, iters=100 → `make_patch`  
   - Measure generation time for each  

5. **Adversarial Evaluation & Analysis**  
   - Evaluate both models on each adversarial loader  
   - Compare accuracy drops and robustness  
   - Summarize findings in Jupyter notebook on GitHub  
