Abhinav Venigalla lead the NLP team at formerly MosaicML, now part of Databricks.

# Training LLMs From Scratch
In this module, we'll be talking about training LLMs from scratch and some of the infrastructure and science that goes into that with the focus on how we built the open source models **MPT-7B** and -30B.
So in this talk there'll be three main sections. 
- The first is compute and orchestration, basically how we execute these large LLM training runs on big clusters of GPUs.
- I'll be talking about the training runtime, some of the software and open source libraries that we've developed that make it easy and high performance for our customers.
- And lastly, I'll be talking about the MPT models themselves, some of the choices that our research team made in terms of the data architectures that we used, how we fine-tuned these models to produce different variants and also how we evaluated them.

# Compute + Orchestration
The first section is about compute and orchestration. The the most important thing to know about training LLMs is that they require a lot of compute and when I say compute, I mean the actual math and floating point operations required to actually train them. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/73fcd5f9-8fa9-4f5e-b3d1-14291476fac1)

To actually finish all this math in a human-friendly time scale, AKA weeks not years, we have to parallelize the work across hundreds to thousands of GPUs and that's what's unique about this. We need special tools for launching these very large runs on huge clusters of GPUs, for managing the compute and sharing with your teammates, for automatically resuming the runs in cases of failure, and and all sorts of things like that. 

# MosaicML Cloud
So in **MosaicML**, we basically decided to build a product that specifcally addresses for ML engineers and we called it the **MosaicML Cloud**. 
- It's an orchestration and sketching layer that sits on top of any GPU compute cluster, whether it's yours, you know whether you rent compute from us, you can switch up seamlessly to any one of them.
- And what's special about it is that it basically addresses all these different problems that ML engineers face when they're actually trying to train models at scale. So that means multi-node training; that means resuming runs when they fail; it means supporting object stores both for dataset streaming and for checkpointing and also experiment trackers such as MLflow and Weights and Biases.
- And the best part of course is that's high performance that we tune it for all the different cloud providers out there. So when you come to us, you don't have to redo all of that work.

# Multi-Node Orchestration (2:14)
