# Anomaly Detection in 3D-bioprinting Technologies Using Machine Learning Methods

Authors: Zeqing Jin, Yifei Zhang, Xianlin Shao, Zilan Zhang  
University of California, Berkeley  
{zjin2017, yifei_zhang, shayd, shilan}[at]berkeley[dot]edu
  
Project summary:  
Although 3D-bioprinting additive manufacturing technologies have been actively studied and advanced in the past decade, anomalies such as discontinuity and irregularity are commonly seen in the process of bioprinting. Challenges lie in detecting these transparent defects efficiently and accurately. In this study, an anomaly detection system based on layer-wise camera images and machine learning methods is developed to distinguish various anomalies with satisfying performance. It is envisioned that this work will provide effective information on layer-wise printing conditions as well as potential applications in the autonomous correction of bioprinters without human interaction.  

Code description:  
(1) Training (optional): The training process is established on Colab platform. A 30 epoch training is about 45 minutes using the GPU provided on-line. In the 'ipynb' file, a history of the training prcess is recorded. You may omit the training step and look through on how we establish the customized dataset loader as well as the CNN models. The training log can also be visualized through the 'Tensorboard block' at the end of the file.  
(2) Baseline models are examined after the dataloading in both 'classification_final' file and 'baseline2_final' file.  
(3) Evaluation on testing dataset: With the saved model checkpoints, the evaluation metrics are calculated based on the testing dataset. A csv file is also exported with detailed prediction results.  
(4) In the file folder model_checkpoints and logs, you can find the files used in the paper as well as the sample file matching with the 'ipynb' file history.   
(5) The training log results and testing results are visualized using 'training_log_result_visualization.py' file and 'testing_result_roc_curve.py' file. The figure is shown in the Figure 3 of the paper.  
* The link to the dataset and model_checkpoints: 
