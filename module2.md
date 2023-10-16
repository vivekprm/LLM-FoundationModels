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

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/86921c5b-6ef0-4d4f-a6fd-34b3d95f8999)

On the above image over here, you will see that we can fine-tune BERT on different data sets: on **SQUAD**, which is a Q/A data set, a **named entity recognition data set** a multi-genre natural language inference data set. Each of these fine-tuning processes will give us a new model, so we serve one model per task at deployment time.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/ce5cca71-7b05-45a9-a706-38b01b8c7afb)

But this means that we need to deal with many copies of the same foundation model and this is especially undesirable when our models these days are much bigger than this 110 million BERT model. Disk space is cheap these days but some models like **LaMDA**, which is a family of models specialized for dialogues, takes almost **500 GB** of disk space. So another question is that we need to consider at deployment time: **do we have that many input requests for specific tasks**?

Next, for full fine-tuning, it may also produce an **undesirable consequence called catastrophic forgetting**, which is when the foundation model that we trained or we leveraged before already forgot how to perform other pre-trained tasks. So full fine-tuning is expensive. How do we avoid doing it? There are two methods:
- One is few-shot learning;
- The other is parameter-efficient fine-tuning.

Before we go to parameter-efficient fine-tuning, which is what we often abbreviate as PEFT, we'll briefly look at what few-shot learning is. 

# X-shot learning
Few-shot learning is when we provide several examples of new tasks in a text prompt. We typically use this when we don't want to do fine-tuning or when we lack labeled training data, but we can easily write out a few examples for the LLM to learn from. So this process of writing but we can easily write out a few examples for the LLM to learn from. So this process of writing out instructions in the form of text is called a prompt and we iteratively write different prompts to find out the best prompt to pass to our LLMs. And this is a process called **prompt engineering**.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/a91f89a6-e547-4fd0-bd3c-8f7b1f82d3b2)

So this process of developing prompts can also be called as **prompt design**. And especially important for this particular module, we will be referring to prompt engineering as **hard prompt tuning** or **discrete prompt tuning**, so just know that whenever I refer to hard prompt tuning or discrete prompt tuning, I'm really talking about few-shot learning. And a main distinction that we should make between few-shot learning and fine-tuning is **few-shot learning doesn't update any model weights whereas fine-tuning updates model weights**.

## Pros and cons of X-shot learning (Also known as in-context learning)
![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/4bb32fcb-a0b1-4890-b3a9-ac7857990cb3)

Few-shot learning is also commonly known as **in-context learning** because we provide our LLM some context to learn from or to leverage during its output generation process. The pro for such few-shot learning is that **there is no need for labeled training data** and we don't have to create any copy of model for each new task. So this can greatly simplify our model deployment process.

Probably the nicest advantage of all is that **the text prompts that we pass in can be very easily understandable because those are the inputs that we, humans, craft as text to pass to the LLM**.
But the con is that, even though we get the advantage of being able to interpret the text prompts, everything is manual and labor-intensive. **The prompts can also be highly specific to models, which means when you change to a different model, you may need to develop a new prompt**. altogether. 

And we also often encounter the limitation of context length. If we were to add more examples then that would give us less space for instructions. But if we were to use a model with higher context window or larger context window, then that will also give us higher latency at serving time as well.

There is also a recent [research paper](https://arxiv.org/abs/2307.03172) that shows that **longer context windows may not be the solution**, you know, for future LLMs because LLMs tend to also forget the middle portion of the context that we provide. So lastly, there this is really the reason why we turn to fine-tuning, which is that even with few-shot learning, the model performance might still be lackluster.

# Fine-tuning outperforms X-shot learning (Example: Good at Arithmetic Tasks (Goat-7B))
So here is an example of fine-tuning that outperforms few-shot learning, which is our **GOAT-7B**, 7 billion parameter model. The foundation model is Llama and it is trained on a million synthetic data examples and we find that the accuracy outperforms the few-shot learning model over here: **PALM model**, which is a 540 billion parameter model. **It also outperforms GPT-4 on arithmetic tasks as well**.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/dbe1f37a-bfca-4518-81b8-a4861f881967)

But in fact, GPT-4 almost doesn't do that well across all the arithmetic tasks. We also see that this GOAT model can achieve state-of-the-art result on an arithmetic benchmark. So this is a model that falls under supervised instruction fine-tuning realm. It's trained using a **PEFT technique called LoRA**.

## Important observations about Goat
Before we move on from GOAT, I want to call out two important things. First is that **it's an instruction fine-tuned model** and secondly, it can perform multiple tasks at serving time.

So the first task is addition; the second task is subtraction but mixing it with some natural language; the third task is multiplication; and the fourth task is division. So you can see that there is no need to produce one model for each of the mathematical tasks over here at serving time. We are using one model to rule all of the arithmetic tasks that we're providing to the LLM.

Now let's look at a couple other examples of popular instruction-tuned, multitask LLMs. The first is called **FLAN**, which stands for **Fine-tuned LAnguage Net**. The foundation model is a 137-billion parameter model and is fine-tuned on over 60 NLP data sets with different task types. An example of different FLAN-flavored models is T5: or **FLAN-T5** or **FLAN-Palm**.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/927044a9-5ae2-4ef2-b7de-c6fea92714e8)

# Instruction-tuned, multi-task LLM  (Instruction-tuned = tune general purpose LLMs to follow instructions)
Now let's look at a couple other examples of popular instruction-tuned, multitask LLMs.

The first is called **FLAN**, which stands for Fine-tuned LAnguage Net. The foundation model is a 137-billion parameter model and is fine-tuned on over 60 NLP data sets with different task types. An example of different FLAN-flavored models is T5: or FLAN-T5 or FLAN-Palm.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/d39f9c13-0494-4d7b-a547-0300e4c5ff4c)

Another model example in this category is **Dolly**, where the foundation model is **Pythia 12-billion-parameter model** and is fine-tuned on 15,000 pairs of prompts and responses. 

Now that you have seen a few examples of fine-tuned models, let's recap the goals that we want to accomplish with these fine-tuned models. We want efficient training, we want efficient serving and storage. Usually full fine-tuning is computationally prohibitive for many organizations, even though this gives them best model performance. So the compromise for those with smaller budgets is to do some fine-tuning, but not full fine-tuning.

# Quick recap
We want efficient training, serving, and storage
- Full fine-tuning can be computationally prohibitive
  - Memory usage: activation, optimizer states, gradients, parameters
  - This gives the best performance
- Compromise: Do some, but not full, fine-tuning
  - Saves cost to use low-memory GPUs
- We want multi-task serving, rather than one model per task
  - E.g. one model for Q/A, summarization, classification
