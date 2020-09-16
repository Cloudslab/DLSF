# Dynamic Scheduling for Stochastic Edge-Cloud Computing Environments using A3C learning and Residual Recurrent Neural Networks
The ubiquitous adoption of Internet-of-Things (IoT) based applications has resulted in the emergence of the Fog computing paradigm, which allows seamlessly harnessing both mobile-edge and cloud resources. Efficient scheduling of application tasks in such environments is challenging due to constrained resource capabilities, mobility factors in IoT, resource heterogeneity, network hierarchy, and stochastic behaviors. Existing heuristics and Reinforcement Learning based approaches lack generalizability and quick adaptability, thus failing to tackle this problem optimally.  They are also unable to utilize the temporal workload patterns and are suitable only for centralized setups. However, Asynchronous-Advantage-Actor-Critic (A3C) learning is known to quickly adapt to dynamic scenarios with less data and Residual Recurrent Neural Network (R2N2) to quickly update model parameters. Thus, we propose an  A3C based real-time scheduler for stochastic Edge-Cloud environments allowing decentralized learning, concurrently across multiple agents. We use the R2N2 architecture to capture a large number of host and task parameters together with temporal patterns to provide efficient scheduling decisions.  The proposed model is adaptive and able to tune different hyper-parameters based on the application requirements. We explicate our choice of hyper-parameters through sensitivity analysis. The experiments conducted on real-world data set show a significant improvement in terms of energy consumption, response time, Service-Level-Agreement and running cost by 14.4\%, 7.74%, 31.9%, and 4.64%, respectively when compared to the state-of-the-art algorithms.

**This is also the first work which integrates CloudSim (java) with Deep Learning frameworks like PyTorch (python).**

## Contributions

The key contributions of this work:
1. We design an architectural system model for the data-driven deep reinforcement learning based scheduling for Edge-Cloud environments.   
2. We outline a generic *asynchronous* learning model for scheduling in *decentralized* environments.
3. We propose a *Policy gradient* based Reinforcement learning method (A3C) for *stochastic* dynamic scheduling method.
4. We demonstrate a *Residual Recurrent Neural Network* (R2N2) based framework for exploiting temporal patterns for scheduling in a hybrid Edge-Cloud setup.
5. We show the superiority of the proposed solution through extensive simulation experiments and compare the results against several baseline policies.

## System Model
<div align="center">
<img src="https://github.com/Cloudslab/DLSF/blob/master/Images/system.PNG" width="700" align="middle">
</div>

## Reinforcement Learning Framework
<div align="center">
<img src="https://github.com/Cloudslab/DLSF/blob/master/Images/RL.PNG" width="500" align="middle">
</div>

## Neural Network Architecture
<div align="center">
<img src="https://github.com/Cloudslab/DLSF/blob/master/Images/network.PNG" width="700" align="middle">
</div>

## Experiments

### Experimental Setup
<div align="center">
<img src="https://github.com/Cloudslab/DLSF/blob/master/Images/env.PNG" width="900" align="middle">
</div>

### Baselines
* *LR-MMT:*  schedules workloads dynamically based on Local Regression (LR)  and Minimum Migration Time (MMT) heuristics for overload detection and task selection, respectively.
* *MAD-MC:* schedules workloads dynamically based on Median Absolute Deviation (MAD)  and Maximum Correlation Policy (MC) heuristics for overload detection and task selection, respectively.
* *DDQN:* standard Deep Q-Learning based RL approach, many works have used this technique in literature.  We implement the optimized Double DQN technique.
* *DRL (Reinforce):* policy gradient based REINFORCE method with fully connected neural network.

### Results
<div align="center">
<img src="https://github.com/Cloudslab/DLSF/blob/master/Images/comparison.PNG" width="900" align="middle">
</div>

## Quick setup and run tutorial
To run the experiments, clone the repo and open the *CloudSim/cloudsim-package.iml* in Idea IntelliJ IDE.
1. Open terminal and change directory to *Deep-Learning/* and run 
```
python3 DeepRL.py
```
2. On the IDE, open *DeepRL-Runner.java* and set the selection and placement algorithms:
```
String vmAllocationPolicy =  "lr"; // Local Regression (LR) VM allocation policy
String vmSelectionPolicy = "mmt"; // Minimum Migration Time (MMT) VM selection policy
```
For training choose any selection, placement algorithm pair. For execution select: *'deeprl-sel'* and *'deeprl-alloc'*. All model files are store in *Cloudsim/model2/* directory.

Note:
1. The output file *DL.txt* can be used to create a combined performance '.pickle' file using *Results/saveData.py*. To generate graphs, you can use *Results/resultGen.py* on the pickle file. However, these codes are specific to the implementation and baseline models in the paper. You can use your own parser from the pickle or log outputs for other QoS parameters and/or baselines.
2. *DeepRL.py* does not produce any output. All interaction with the java files is done using RPC.

## Developer

[Shreshth Tuli](https://www.github.com/shreshthtuli) (shreshthtuli@gmail.com)

## Cite this work
```
@article{tuli2020dynamic,
  title={{Dynamic Scheduling for Stochastic Edge-Cloud Computing Environments using A3C learning and Residual Recurrent Neural Networks}},
  author={Tuli, Shreshth and Ilager, Shashikant and Ramamohanarao, Kotagiri and Buyya, Rajkumar},
  journal={IEEE Transaction on Mobile Computing},
  year={2020},
  publisher={IEEE}
}
```

## References
* **Shreshth Tuli, Shashikant Ilager, Kotagiri Ramamohanarao, and Rajkumar Buyya, [Dynamic Scheduling for Stochastic Edge-Cloud Computing Environments using A3C Learning and Residual Recurrent Neural Networks](http://buyya.com/papers/DynSchedulingEdgeCloudNets.pdf), IEEE Transactions on Mobile Computing (TMC), ISSN: 1536-1233, IEEE Computer Society Press, USA**

[![](http://www.cloudbus.org/logo/cloudbuslogo-v5a.png)](http://cloudbus.org/)
