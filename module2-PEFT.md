# Parameter-efficient fine-tuning (PEFT) and Soft Prompt
Parameter-efficient fine-tuning is often abbreviated as PEFT. Parameter efficiency encompasses many aspects including storage, memory, computation and also performance.

There are three categories of PEFT, including:
- Additive,
- Selective and
- Re-parameterization.

In this course, we'll focus only on the first and the third category, which is additive and also re-parameterization because the selective category is found to be not as good as the other two. But if you are interested in the selective category methods, feel free to check out the links that I've have provided in the slide deck.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/62290194-3d49-42b2-9935-ba9052d992e9)

# High-level overview of PEFT (Active research area: >100 papers in last few years!)
PEFT has been a very actively researched area. At its core, **additive methods include adding new tunable layers to the model. And during fine-tuning process, we only update those new layer weights, while keeping the foundation model weights frozen**.

Re-parameterization involves decomposing a weight matrix into lower-rank matrices. We dive more deeply into both of these methods in the slides to come.

## Implementation
Fundamentally, **these methods act on the core Transformer block; some act specifically on the query, key, value weight matrices that are responsible for passing information**. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/12f91e4b-4a4f-474f-b9da-4ab3a19e7dcf)

# Additive: Prompt Tuning (and prefix tuning)
Without further ado, let's talk about **soft prompt**. 

## Soft prompt tuning (Concatenates trainable parameters with the input embeddings)
If you recall from the last section, **the manual act of writing text prompts is also called hard prompts or discrete prompts**. In this section, it is all about how to remove that manual aspect of prompt engineering but use soft prompts instead.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/4f6d1e0f-1da7-4f76-9485-8fcce6d63804)

Adding soft prompts means that we're adding virtual tokens to our inputs. Take a look at a graphic over here, where I have text inputs. So we often call these as text embeddings "these chips are tasty" but soft prompt means that we are now adding virtual tokens. 

Virtual tokens are task-specific So whenever I mention soft prompt, you should immediately think of soft prompt as virtual tokens and they are synonymous. So the soft prompt has the same dimension as our input embedding vectors. So you can see that it spans the same length over here from top to bottom. We concatenate this trainable virtual prompt or the virtual tokens or the soft prompt with the input embedding vectors during fine-tuning process. And we call this prompt tuning, rather than model tuning, because we are only updating the prompt weights, which is the ones in pink over here.

## What are these virtual tokens? Goal: remove manual element of engineering prompts!
So what are those virtual tokens? Remember that the challenging part with prompt engineering or writing discrete prompts was that it's very manual and very labor intensive and very error-prone. So instead of relying on humans crafting a perfect prompt, in soft prompt or virtual tokens, we let the model find the best prompt through fine-tuning.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/56283c51-8d1e-4497-b2c8-0e2f172d2593)

We first randomly initialize an embedding vector that's just made up of random numbers and this vector will have the same dimension as our input embeddings. So these randomly initialized embedding vectors, since they are completely random, so **they are not part of any vocabulary**. We don't know what text they correspond to. So on the right-hand side, the graphic over here, you can see that when we have real word input
tokens, we can visualize them in an embedding space and know exactly what word it represents, what token it represents. But when we look at virtual tokens in an embedding space, we know that it occupies somewhere in the embedding space but we don't know what text it corresponds to. 

So it's a little bit like **Bitcoin when we know that it functions like money but we cannot touch it like cash; we don't even know how it looks but it exists and it works**.

In some [research](https://arxiv.org/pdf/2104.06599.pdf), we have also seen people initialize these virtual tokens to represent discrete prompts. It means that we are providing some minimal discrete prompts for the model to start with and then the model is free to update embedding vectors during the training process. 

So for example, my discrete prompt can be as simple as a three-word prompt **"classify this sentence"** or **"translate this sentence"** and those discrete prompts will then be free to be updated as the model is fine-tuned.

So that initialization to discrete prompt is also called as **informed initialization**. But what is interesting is that this paper has also found that random initialization, which is when we set the prompts to random numbers, instead of specific text input, is nearly as good as giving an informed initialization, which is when we pass in really simple prompt input, text prompt input like "classify this text" or "translate this text".

So in the notebook later on, we'll play with both types of initialization: **random initialization** and this **informed or discrete prompt initialization**. So that will help reinforce or make the ideas of random initialization and informed initialization more concrete.

# Compare full fine-tuning vs prompt tuning
Let's now first take a look at what prompt tuning involves by comparing it with the full fine-tuning scenario. We're sticking to the sentiment classification scenario. Now, instead of having just one input, we are having multiple inputs, which is why we're defining the task batch to be 4. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/a506674e-0d0e-468f-a61e-e04d3139db56)

Because we have four sentiments to classify in total, notice that virtual tokens are really just random numbers, so they don't correspond to any specific vocabulary or specific text. So just like fine-tuning, when we do full fine-tuning, we update model weights based on loss through backprop but the entire foundation model is unlocked or unfrozen so that we can update all the model weights during the backprop process.

In contrast to prompt tuning, you can see in this diagram that the foundation model weights are completely frozen. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/ab99aecc-1b82-4746-9f1d-34babdc35169)

So when the model goes through forward pass and also backprop, **we only update the virtual token weights here**. So these virtual tokens or these soft prompts are basically learned through backprop and tuned to incorporate signals from any number of labeled prompts that we provide but. And we only apply the gradient updates to these virtual embedding vectors.

To recap, when we think about manual prompt engineering or discrete prompt writing, which is when we supply just like in a few-shot learning example, where we supply some examples and pass in as context to the LLM. That's when we are searching the text space over tokens with fixed embeddings but with prompt tuning, we are searching in the embedding space to find the best representation of the prompt that the LLMs should accept. And **the best thing out of this is the model learns the optimal representation of the prompt automatically**. So that removes the manual labor of engineering a discrete prompt.

##  Allows swapping of task prompts (Efficient for multi-task serving)
So far, our example only includes a single example of task: sentiment classification. But what if we have multiple tasks including Q/A, translation, etc.? That's not a problem because **we can treat each task as a prompt**. For each prompt, we would specify that for each task. So **for each task, there will be a different soft prompt altogether**, so that way at deployment time, we don't need to swap in and out the base model or the foundation model. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/3c18d432-dda2-4b90-a844-b9de1b3cf844)

What we need to do is to **only swap in and out the learned virtual tokens**. The other benefit is that the model can now process a single larger mixed task batch. And a task batch over here can consist of multiple requests. You can see now it not only has sentiment classification but it can also take in a Q/A requests or translation requests. And the variety of tasks that is captured in a batch is what we call as a **mixed task batch** over here.

# Matches fine tuning performance for >11B model
What researchers found is that for larger models, especially **at more than 11 billion scale parameter, prompt tuning matches the performance of full fine-tuning**. **[SuperGLUE](https://w4ngatang.github.io/static/papers/superglue.pdf)** score is a benchmark that's styled after [GLUE](https://gluebenchmark.com/) that contains a variety of tasks, including answering boolean or comprehension questions. So the higher the score, the better it is.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/0b38d7af-212f-4272-8237-2ce125229f4c)

We can see that soft prompt tuning doesn't perform as well when a model is small, which intuitively makes sense because **smaller models have smaller capacities to learn**. But when the model gets larger, then we can see that, with full fine-tuning versus prompt tuning, we can see the performance actually is very similar. 

And I also want to call out with this image that, prompt design is referring to the manual prompt engineering that we have to do. Prompt length also doesn't really make that much of a difference for large model performance.

# Prompt length affects larger models less (Prompt length of 20-100 is typical)
So what is prompt length? We can see in this example, when we initialize our virtual prompts, there are only two embedding vectors over here. So this is a prompt length of two. **Larger models do well even when the prompt length is one**. And in fact, there seems to be diminishing return, diminishing return in model performance as we increase the prompt length.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/e8a4af9c-fc66-4f36-a244-68a020f0ae7e)

A prompt length of 100, which is the green one over here, seems to be the sweet spot but also notice that the confidence bar for the scores for the SuperGLUE score is also quite wide. So, **soft prompt tuning has the problem of unstable performance**.

# Advantages of prompt tuning
Let's recap the advantages of prompt tuning. Unlike few-shot learning, where we have to do manual prompt engineering, 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/12af4317-dcb9-4d22-a605-97a6481e1df5)

- We are not limited by the number of examples that we can pass to the model context.
- We also eliminate the challenge of crafting the best prompt manually. We can use backprop to ask the model to help us find the best embedding representation of the task-specific virtual prompts.
- We don't need to have multiple copies of the same model, We can afford to have multitask serving
- And lastly, it is also resilient to domain shift.What this refers to is, since we freeze or we lock the foundation model weights, prompt tuning prevents the model from modifying its general understanding of language. Therefore, it reduces the model's ability to overfit on your fine-tuned data set.

By comparison, our learned soft prompts also have a much smaller number of parameters, so they can be more generalizable to the variations of the tasks at inference time. 

# Disadvantages of prompt tuning
Of course, as we saw earlier, prompt tuning has some disadvantages. You'll probably have been wondering the whole time or even in the beginning too, you know, how do you know what these virtual tokens are? The answer is we don't really. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/8cf15d84-95e2-4692-acd7-0158e3ebcfb8)

**The best attempt interpreting them is to use some cosine distance or some distance metric to find out nearest neighbors** so we can estimate or guess what words they might represent. So it is much less interpretable compared to discrete prompt. The second disadvantage, as we saw just now, is prompt tuning can have unstable performance.

# Prefix tuning is very similar to prompt tuning (Adding tunable layer to each transformer block, rather than just the input layer)
Under the category of soft prompt, there's another very similar method called **prefix tuning**. So prefix tuning is very similar to prompt tuning. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/c7ae63e5-d38e-4587-8c85-981265a890be)

It also allows task-specific prompts, where each prefix represents a different task. **The only difference is that these prefix layers are added to each Transformer block, rather than just the input embedding layer**.

# Re-parameterization: LoRA
In this section, we'll talk about one of the most popular techniques today in PEFT, which is LoRA. **LoRA is our re-parameterization metho**d, which means **it leverages low-rank representation to minimize the number of trainable parameters**. We'll cover a small amount of linear algebra to dissect what low-rank representations mean. 

## Low-Rank Adaptation (LoRA)
In fine-tuning or any general model training scenario, we update the model weights as we go through forward and also backward pass.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/5fa4be59-0df4-4a13-a29d-841ff0da5aa6)

The idea behind LoRA is that we can decompose the weight_delta matrix into two low-rank matrices. What difference does it make, you may wonder? Before we can answer that question, let's now briefly revisit linear algebra basics to understand what matrix rank is.

# Rank? Brief visit to linear algebra
**Rank refers to the maximum number of linearly independent columns in a matrix**. When you have a full-rank matrix, it means that the matrix doesn't have any redundant rows or columns that can be expressed based on other combinations of columns.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/ecb4f423-a645-44cf-8552-a2d7f018cd12)

So take a look at an example of above over here. Since column 2 and column 3 can be obtained by multiplying column 1 with a constant, they are not linear and they are not independent. Therefore, the column rank is one. The same applies to the second row as well: because we can get a second row by multiplying the first row by three. So essentially, we're trying to make sure that we represent information in a matrix without any redundancy.

## How does weight matrix decomposition work? (Observation: Actual rank of the attention weight matrices is low)
So how does weight matrix decomposition work? The observation that inspires LoRA is that **the rank of the attention weight matrix change is lower than the actual Weight_delta matrix**. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/cd297b71-c00b-45a8-b278-f59c41f2c5e9)

So when we do any fine-tuning, **we can freeze the pre-trained weights and only update the two lower rank weight matrices as demonstrated by W_a and also W_b over here**.

But how does this reduce the number of trainable parameters? Let's take a look at a dummy example. W_delta has dimension of 100 x 100. We can decompose that to two smaller matrices, W_a with 100 x 2 dimension and W_b with a dimension of 2 x 100. When we multiply these two matrices together, they still give us the same shape as (100x100) which is the same shape as W_delta. And this is really important because we can now then concatenate the results of these matrices with the original pre-trained weights and pass that to the subsequent layer. And this decomposition method dramatically reduces the number of parameters.

So the total number of parameters that we see over here is (100x2) + (2x100), which is 400. But if I were to get the number of parameters from this original W_delta matrix, then we have 10,000 parameters altogether. So this brings us to a 96% of reduction in number of trainable parameters.

# LoRA matches/~outperforms full fine-tuning
![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/418476d2-c99f-4649-b194-4f621f5dd86e)

Researchers found that **LoRA matches the performance of fine-tuning and sometimes even outperforms fine-tuning with just 0.02 percent of the original parameters of GPT-3 over here**. 

# LoRA performs well with very small ranks (GPT-3’s validation accuracies are similar across rank sizes)
The next natural question to ask is how do we determine the rank of these matrices? We can treat that as a hyperparameter to search over. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/c91b919a-048a-4d4e-b1ae-fa68c51e3563)

Generally, rank sizes produce roughly similar validation accuracies, at least for GPT-3. Intuitively speaking though, small r likely won't work for all tasks or data sets because in cases where the downstream tasks are much more different than the original tasks that the base model is trained on.

But the researchers have also played with updating different combinations of weight matrices for decomposition but there were no clear trends to take away from. 

# Advantages of LoRA

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/21f46ebe-b626-4d2d-8189-aa9c523b4821)

**In a nutshell, LoRA, just like prompt tuning, 
- Locks up or freezes the majority of the model weights**.
- You can share or reuse the same foundation model.
- You can improve training efficiency since you don't have to compute most gradients or optimizer states.
- It also adds no additional serving latency because we can merge W_a and also W_b literally.
- We can also combine it with other PEFT methods as well.
  - However, the existing PEFT library from Hugging Face doesn't allow a combination of PEFT methods to be concurrently used yet.


# Limitations of LoRA
Now let's talk about some of the limitations of LoRA. 
- Even though we could theoretically just swap different weight update matrices at serving time, It is not really straightforward on how to do so using when we have a single mixed task batch.
  - If we want to dynamically choose which combos of weight matrices A and B at serving time,
- Then there's additional serving latency. But there are also, of course, other open research questions as well, such as,
  - why do we only decompose W_delta? Can we decompose other matrices like the original weight matrix or can we reduce the number of parameters even more?
  - And in fact, there is already a newer PEFT technique called [IA3](https://arxiv.org/abs/2205.05638) that improves upon LoRA that can reduce even more number of trainable parameters.

# PEFT Limitations
