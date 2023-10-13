#  Efficient Fine Tuning: Doing more with less
A large language model is called large because of both the large amount of data it's trained on and also because of a large number of trainable parameters. Naturally, that prohibits many individuals and organizations from training an LLM from scratch. As such, we turn to doing fine-tuning, which is a way of updating only a subset of model parameters. We do fine-tuning in a parameter-efficient way and the limitations that come with that.

# Fine tuning vs. transfer learning
Transfer learning refers to when we apply a general pre-trained model to a new but related task. A common analogy that you might have heard about is playing sports. If you know how to play tennis, you can probably pick up volleyball more easily because both require power. 

Fine-tuning falls under the umbrella of transfer learning. It simply means that we are training that model even more. You may hear some differentiate both by saying
fine-tuning applies when we change or modify the architecture of the base model, you know, where we unfreeze a few top layers of the base model and train those layers,
in addition to the new layers that you have just added. But there's really not much need to differentiate them such so finely. And in fact, most people reference them interchangeably including Andrew Ng and also Andrei Karparthy as well. You can think of fine-tuning simply as training a base model even more and/or training a base model on more data or different data.

# How to leverage a pre-trained foundation model?
The pre-trained model that we talked about is also often called as a foundation model or a base model, especially in the context of LLMs. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/cf73f6a4-20e2-4ba1-870d-0dcc897de183)

Generally speaking, there are three ways of leveraging these foundation models. The first way is simply to use them as they are and the examples of these pre-trained models are everywhere, you know, including T5, GPT-4. BloombergGPT.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/78b5f233-cf34-4ada-9b8d-7aa1a0086121)

The second way is to use the output of these foundation models as a feature. In this instance, we can use the output embeddings from the base model and feed that into another machine learning model to generate predictions. So this is a very common use case of BERT embeddings.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/9fcb7961-ce35-45b1-ac53-a1e1cfe7eabe)

The third way, which is what we'll focus on in this module, is fine-tuning, where we may update only a few top layers of the base model or update all layers or add some layers before we use the model to generate our desired predictions.

We'll look at one of the newest LM examples in this category, which is a model called **GOAT**, unsurprisingly another farm animal. It is a fine-tuned version of **Llama** to perform arithmetic operations. So we'll return to GOAT in a few slides. 

# Why fine-tuning? 
We want to have better performance in a downstream task we can fine-tune a specific pre-trained language model on task-specific data, adapting to a specific style and vocabulary and this can also help us to achieve regulatory compliance as well.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/154a3911-6614-4c02-93f2-3a5b6f95fb78)

But this idea is not new at all. In fact, in 2018, Jeremy Howard and Sebastian Ruder published a [paper](https://arxiv.org/pdf/1801.06146.pdf) on fine-tuning techniques that we can use for any NLP task and one of them is to fine-tune a classifier layer on a target task by gradually unfreezing layers. 

To put it simply, fine-tuning means we are updating model weights or model parameters.

# Fine tune = update foundation model weights
Usually, when we update more layers, we get better model performance. And typically when we do full fine-tuning, which means when we are updating all the model weights, we produce one model per task.
