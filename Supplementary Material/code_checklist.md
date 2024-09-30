# A Deep Learning and Stochastic Process-Based Framework for Traffic States Estimation¡ªDensity, Speed and Flow Direction¡ªSolely By Traffic Flow Data

This paper elaborates on how to utilize the code and data provided in the supplementary materials. The paper is available in two versions: one in .md format and the other in .pdf format, with both versions containing identical content.

## Requirements

The code provided in the supplementary materials runs on Python and requires the following packages: torch, math and matplotlib.

## Training

To sample for training the fitting model in dataset1, run this command:

```train
python sample_for_fitting_model1.py
```

The code will output the extracted samples saved in the file <font color="OrangeRed">'data_of_dataset1/output/traindata_for_fitting_model1.pt'</font>.  <font color="OrangeRed">'preextracted_traindata_for_fitting_model1.pt'</font> is the pre-extracted samples.

To train fitting model for fitting numerical solution in dataset1, run this command:

```train
python train_fitting_model_for_dataset1.py
```

The parameters are saved in <font color="OrangeRed">'data_of_dataset1/output/fitting_model_for_dataset1.pt'</font>, while <font color="OrangeRed">'pretrained_fitting_model_for_dataset1.pt'</font> is the pre-trained parameters of the fitting model.

To train inference model in dataset1, run this command:

```train
python train_inference_model_for_dataset1.py
```

The parameters are saved in <font color="OrangeRed">'data_of_dataset1/output/inference_model_for_dataset2_gat.pt'</font>, while <font color="OrangeRed">'pretrained_inference_model_for_dataset2_gat.pt'</font> is the pre-trained parameters of the inference model.

 The usage method of numerical_solution.py is as follows. The definition of <font color="Blue">n</font>, <font color="Blue">l</font> and <font color="Blue">g</font> is detailed in Algorithms 2 and 3 in the appendix. <font color="Blue">differential_u</font> is the expectation of U at time t+1. <font color="Blue">differential_flow</font> is the expectation of F during [t,t+1]. Do not set n too high to avoid numerical overflow. We recommend setting n=4 or 5.

```python
equation=Equation(batch_size,d,n,l,g)
differential_u=equation.last_u(v*p,u,b,d)
differential_flow,differential_u=equation.last_output(v*p,u,b,d)
```

## Results

To visualize fitting results, run this command:

```train
python visualize_fitting_results.py
```

To visualize inference results of U and V, run this command:

```train
python visualize_infenence_results_of_u_v.py
```

To visualize inference results of P, run this command:

```train
python visualize_infenence_results_of_p.py
```
