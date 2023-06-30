## Examples (using wrappers)

### Running a classifier pipeline

```python
CP = ClassifierPipeline(run_name, override, pipeline_path="ml_pipeline/test/") # all the magic is in the override argument that changes the predefined hydra config in conf/ directory

CP.build_train_pipeline()
CP.fit(force=True) # if model with this name already exists force training again

CP.build_inference_pipeline(test_dataset) # test classification, e.g. "higgs_bkg"

res = CP.infer(return_results=True) # returns classification scores for test_dataset

class_train_pipeline, class_infer_pipeline = CP.pipeline["train_pipeline"], CP.pipeline["inference_pipeline"]
```


### Running a flow pipeline
Only change is in model_name, which can be any implemented flow model and in N_gen (number of generated events).

```python
FP = FlowPipeline(run_name, model_name, override, pipeline_path=f"ml_pipeline/test/",)

FP.build_train_pipeline()
FP.fit(force=True)

FP.build_inference_pipeline(N_gen=10 ** 5) # inference == generation

res = FP.infer(return_results=True) # returns flow generated results

flow_train_pipeline, flow_infer_pipeline = FP.pipeline["train_pipeline"], FP.pipeline["inference_pipeline"]
```

## Workflow 

1. generator_pipeline
2. cut_pipeline
3. hists_pipline
4. ul_pipeline
    - pull_plots
5. cls_pipeline

### List of analysis blocks 
- utils.py
  - SigBkgBlock
- hists_pipeline.py
  - MakeHistsFromSamples
  - MakeHistsFromSamplesLumi
- ul_pipeline.py
  - UpperLimitScannerBlock
  - PullBlock
- cls_Pipeline.py