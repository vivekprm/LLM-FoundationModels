# Deployment Optimizations: Improving model size and speed
Let's talk about what the problem is now that we have with these large language models they're becoming extra large.

# The issue with high performance LLMs: Paying the price for quality
What this comes down to is really memory. We found that these large language models that have hundreds of billions of parameters don't tend to fit on consumer or enterprise GPUs.

We've seen from our very large language models that as they grow in size they tend to perform better, both with the accuracy of their output. They're able to be better aligned with what we need them to do, and they have much broader range of abilities to solve different types of tasks.

However, this comes at a cost, particularly of speed, which we even saw in Module 1 when GPT-2 extra large took much longer to produce the output. The memory footprint, if you're a developer of large language models then the out of memory error is something you're going to be all too familiar with. **But also updateability, if we have new data that comes in and we need to keep training our models the larger they are the harder this is to do, and so we end up with a choice**. 

Do we have to pick a small model that is at least fast, but doesn't have the quality, or do we try to work with a very large model and spend as much as we can on our compute resources to make use of that high quality. Or what if we could do both What we're going to do in this Module is we're going to take another look at some of the components inside the Transformer itself to see if we can make some improvements and help alleviate some of these issues.

# Improving Learning Efficiency
One of the key pieces of technologies that unlocked large language models to be what they are today is the **attention mechanism**. And while this is a wonderful tool that has enabled large language models to interpret text in a way that we didn't imagine possible before, it also leaves us with some issues that we've needed to come to terms with, and find ways around.

## How we interact with LLMs: The importance of context length
If we think about how we interact with a large language model, we have to talk about the context length. That's effectively the amount of information that we put into one of the prompts that the large language model will use to interpret what is being asked or the type of conversation that's going on.

Like us, we do better with contexts that are larger, so do large language models. However, increasing large language model context length isn't as simple as you might think. If you think about the attention mechanism, and how that operates. In theory, you can have those longer context length as you want the operations that go through the query, key, and value vectors span across the entire sequence length.

There's no limit of what that sequence length can be, and that doesn't affect the parameters. The parameters are based on the dimensionality that you have for each of your word embedding vectors. When we increase the size of the context length and we apply an attention mechanism over that context, the computing of input values increases somewhat linearly, performing the position-wise feed forward network calculations also increases linearly.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/08da33e0-1701-4e1c-bb50-a83bd0be2846)

However, calculating the attention scores themselves increases quadratically, and that's because we have a, N by N matrix to deal with.

However, even worse than this is the fact that if we train on a context length of say 1000 and then we perform inferences on context lengths they're twice as long as
this or three times or ten times the performance of the models are shown to get far worse.
One of the reasons that is likely for this, is the positional encoding that we use when we give the information of how one token is related to another in position to the attention mechanism.

We saw in the notebooks that we use the positional encoding using the sines and cosines to give a relative sense of how these tokens are positioned relative to another. However it appears that this doesn't allow for the neural network or the attention mechanism to understand very well how these are different when we train on one length and test on a much longer length. You can see the **perplexity scores**, which is how well it knows which token to select next, get worse and worse as you change the number of contexts.

# Training short but inference long: You’ll need a good Alibi for this one
We can try and fix this by looking at the attention mechanism itself. If you think about what is going on in the attention mechanism, we provide a query vector and look at all of the keys. Now one innovation that was introduced which is called **ALiBi**, took an approach by weighting the these particular pairs of vectors, the query times the key and just applied a linear basis so that tokens that were one step away from the current query token we're given a scale of negative one times by some factor. It turns out that a good factor, and this is 'm' in this case, is a geometric sequence.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/b7c4c4e2-7d2f-43d0-a322-4e9a0bfc792e)

You can learn more about this by looking at [the paper](https://arxiv.org/abs/2108.12409) in the link provided. But essentially this just gives a fading off of importance from the current token all the way back to however long that you might want your context to be. What this means though, is that you can train on a relatively short context length and increase it in inference to almost any size that you want.

This has enabled the context lengths to be at a maximum of something like 4000 all the way up to 32,000 64,000 and in some cases even larger than 100,000. This means that we can add a lot more information into our context lengths, including documents, code bases, and chat histories, and we get even more performance from our large language models. This is fantastic however, you might also then realize, based on what we were talking about just before, that the compute resources required to have
much longer inferences now become an issue.

This is no longer an issue of storing the model on memory, just loading the parameters in, but actually creating these attention weights themselves. They are now an issue that we'll have to deal with. Thankfully, this field is full of many smart people, and we've already come up with many solutions that can fix this problem.

# Faster calculations: Calculating attention in a flash.
But in particular the one that is catching most of the attention of the wider community is something called **FlashAttention**. This leverages something that we've seen
in linear algebra before, where we don't actually need to materialize these large matrices at all. We can actually do this in a matrix free operation.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/d8b1250f-0b71-41d6-a506-05a259f73f62)

Because we know which index one vector will need to interact with the index of another vector, we can actually just take these individual variables one by one.
The reason that this is an important approach is because we need to look at the hardware that's used when we calculate attention.

If we think about what goes on inside the GPU when we're calculating our attention scores we actually interact with something called the **SRAM**. This is akin to the cache that's on a CPU and it's basically very fast memory that's very small but very close to the compute units.

In the case of large language models, this **SRAM gets overloaded very quickly when we try to load in the full matrix of the attention**. If we never actually materialize the attention matrix, then we can keep sending the individual variables to the SRAM and line it up so that we never have to go back to the slower memory and
incur the performance costs that you would if you had to put the entire matrix onto this SRAM that just isn't large enough.

Using something like **FlashAttention**, and it's later variants, we see an order of magnitude speed up in calculating these longer attentions that are coming from
the fact that we are now able to use longer contexts.

# Many queries, fewer keys: Multi-query and Grouped-query Attention

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/fc27c1a2-f348-4340-a2f1-f23b3801a044)

Looking even more at the attention mechanism, we can think of other ways that we can try and improve this process. In the first module, we talked about attention but we didn't talk about **multi-head attention**. In essence, **multi-head attention takes the entire attention matrices of the key and value matrices and splits them into multiple heads. This means that we send a query to a number of different matrices but their sum total is the same size as if we just have a single matrix**.

The reason that we might want to split this into multi-headed attention is it allows these different versions of keys, queries, and values to focus on different parts of the speech.

By splitting these up into multiple heads, you might have one head looking at nouns, one looking at prepositions, and one looking at other parts of speech. However, while this produces more accurate results as we get a more detailed and enriched version of what we end up with at the end of our attention scores, this is slower as we have to do these calculations in multiple steps.

Some improvements to this have included **multi-query attention**, where actually we create different copies of our query vector and feed them into just one key and value vector. The issue with this however, while it is much faster than doing multi-headed attention, it tends not to capture all of the differences in the nuances that we need when we're looking at something like a multi-headed attention case.

The happy medium to this is **grouped query attention**, which is what large language models like **LLama 2** leverage. In this situation, while we have a number of
different heads for our attention mechanism, we send them a few different query vectors. Keep in mind these query vectors come from the same token, but they use slightly different projection operations so that we have different versions of those query vectors for the key vectors to look at.

This enables us to leverage the multi-headed, multi-focus, version that we get in multi-headed attention with some of the speed-ups in the multi-query attention.
Grouped query attention is one of the latest innovations that we've seen so far in improving the attention operation.

# Improving Model Footprint: Doing more with less
In the previous sections we looked at how we can improve the storage and processing of our large language models by changing the algorithms and also looking at how we actually compute some of these algorithms in practice. We can take this a step further, and look at how we're actually storing the numbers that large language models represent in a fundamental way. This process is called **quantization**.

## Storing numbers: Billions of parameters. Each a floating point number.
The way that we store numbers in modern computing is by using things like **fp16** or **32** standards.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/ccf8957d-d87f-41be-807b-28d89ee4ba55)

This allocates some part of the number being stored in memory for the exponent, so that's raising some number to some power, and also the mantissa which is how we represent the digits in the number itself.

The IEEE standards have particular ways of storing 32-bit precision and 16-bit precision, and that means that it allocates those number of bits to store that particular number. These are typically stored in base 2.

Google Brain, when they were developing more deep learning algorithms, also realized that storing full, 32-bit precision was potentially unnecessary.
They came up with the [brain float](https://cloud.google.com/tpu/docs/bfloat16) (bf) 16 format which is used in a slightly different way than fp16. While it takes up the same amount of space on the computer as fp16, it actually changes how the exponent and the mantissa are allocated in the number.

It allows for a much larger exponent, the same as that of floating point 32, but a much smaller mantissa. The way this works is they can actually store numbers that have a very large value or a very small value but for intermediate values they don't pay as much attention and they lose a bit of accuracy. This is actually fine for
things like deep learning, as typically the numbers that you want to pay most attention to are the weights, but also the training procedure. We have to look at the gradients, typically very small and very large gradients are what you want to pay attention to, when you're optimizing and having this trade-off of a much larger exponent relative to the mantissa was something that paid off and we see the bf16 standard very common now and supported by most GPU providers.

However, we can take this a step even further and go beyond the bf16 standard to try and save even more space. 

# Quantization
The process of quantization takes a floating point number and turns it into an integer. It does this by first taking a look at the numbers that we want to quantize, looking at the largest number and then finding what the ratio is of that largest number to the largest number that we could represent for the amount of space that we're allocating. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/3582391f-9db2-471a-97ca-04dc02066506)

In the case of an 8-bit integer, that would be the number of 127. And so we look at the largest number in our floating point vector and we take 127 and divide it by that largest number, that then gives us a quantization factor. Which means that we'll take all of the numbers in the floating point vector and multiply them by that quantization factor.

We'll also round that number so that we  get a complete integer. That is where the source of the error typically comes from. So we can then create a quantized vector
of integers with just storing one floating point quantization factor.

We can then convert our quantized vector back to a floating point vector and, depending on how much precision we lost, we may end up with the same floating point vector that we started with.

If we think about how a function is actually represented in any computer, it is a finite precision. There are no continuous functions that we can infinitely store in any computer, what we're doing with quantization is just taking this to an extreme level where we just distinctly take chunks of value ranges and putting numbers in those buckets.

# QLoRA: Applying quantization to fine tuning
If you recall from Module 2, we looked at LoRA. An improvement to LoRA came in the form of the quantized LoRA. Quantized LoRA, or QLoRA, takes all of the good parts of LoRA and then turns it up to 11 by quantizing the Transformer itself into a 4-bit version.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/bab0c8ed-6a9b-4f53-ab38-fd8199b74ec2)

And we'll look at how we can create quantized versions of neural networks in the notebook to follow. It then takes the adapters and turns them into an even smaller version of their 16-bit representation, and as a final step it also uses the CPU to store the state of the optimizer and then load that from system memory into the compute resource as it is required. This allows QLoRA to work on gigantic large language models but do it on small pieces of hardware.

QLoRA is one of the most popular parameter efficient fine tuning methods available to date, and more innovations are being brought to the table. We've seen 8-bit, 4-bit, and even some 2-bit optimizers produced for the QLoRA approach.

This is about as far as we can take quantization for the moment, as the errors that we get when we try to quantize a 100 billion parameter model tend to become too severe for the applications that we want.

# Multi-LLM Inferencing: Hybrid and Ensemble-based systems
Suppose you've already done all the optimizations that you can, or maybe you don't need to. Maybe you have the compute abilities to handle some of the multi hundred billion parameter large language models that are on offer, but you still have more data that you want to train on. Where can you go from here? 

Maybe you have enough access to the large language models but you have a finite amount of budget for inference. In this section we're going to talk about how we can use multiple large language models, and go through a number of different approaches and styles of using these large language models in both inference and training. 

# Mixture-of-Experts: A trillion parameters, for a fraction of the training
Let's start with mixture of experts. The idea behind mixture of experts is that we can typically make use of multiple versions of a smaller system that is trained to perform particular tasks. This is fairly common in the realm of machine learning and deep learning as it is how ensemble methods work.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/2947ab7e-fb82-46f8-993f-7b437c76cb8f)

The difference with mixture of experts is that an input is sent to a piece called a router, and the router is trained, and learns how to send one input to a different type of expert. This doesn't have to be a large language model, it can work in different types of machine learning and deep learning applications, but in this situation we're going to focus on mixture of experts in the context of large language models.

When we think about where the parameters lie in large language models, **more than two-thirds of them are present in the position-wise feed forward neural networks**. These are present in each of their Transformer blocks and they add extra enrichment to the vectors after they've come out of the attention mechanism.

**The switch Transformer**, which was presented by researchers from Google, leverages the fact that, by using different feed forward neural networks and training them during the training procedure, we come up with an approach where we can have multiple experts of these feed forward neural networks trained at the same time.

The way that this helps us with our parameter cost, is that we could have multiple, say 100 billion parameter, feed forward networks and train them one at a time.
This would mean that during the training process, through different samples in each batch, the router would learn which expert to send the signal to. It might send it to a couple of experts and then take some sort of aggregate of the outputs, or it might send it to just one expert.

This is how we could take multiple 100B parameter models and piece them together to make one large ensemble model. This is how we could easily go from multiple 100 billion parameter Transformers all the way up to trillion and multi-trillion parameter models.

More research is being conducted into how these mixture of experts approaches work, but we've seen excellent results with the switch Transformer. These still require a lot of compute resources and all of the optimizations that we've seen up till now will be useful as we dive deeper into the realm of mixture of experts.

But let's say we're not worried about training per se, we're more worried about inference. Let's say you have a fixed cost budget and you're only able to interact with large language models through some API. In this case, you might look at something like an **LLM Cascade**.

# LLM Cascades and FrugalGPT: Improving our resource allocation in LLM inferencing
In The [Frugal GPT](https://arxiv.org/abs/2305.05176) paper that was released in 2023, the researchers came up with an approach where they would take a prompt and pass it first to the lowest performing model, and then look at the results of how that model thinks it performed. We have access when we output a particular token from a model into the perplexity of that result that gives us a sense of how sure these large language models are about what they just selected.

If the model, which is a low quality model, at this stage is unsure or has a high value of perplexity, then we would skip the output for that one and move on to the next complex model.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/dbef02d8-e2b0-41cd-94b8-5532610c0e6e)

This cascading effect of complexity and self-checking meant that Frugal GPT was able to maintain its accuracy far higher but use far less cost. Approaches like LLM Cascades and Frugal GPT are just the start of a new area of exploration for research and industrial use cases where we'll take the most of what we can with the vast array of large language models that are present in the domain.

# Current Best Practices: If you want to build now, do it right
Now that we've seen a lot of the optimizations and improvements that you can make for both training and deployment, let's put everything together and discuss some of your options now if you want to get into the realm of LLMs either for training or just for inference.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/b5af008f-2fb6-4bec-8dc7-6eda099f6c99)

As a list of best practices if we're trying to train something from scratch
- We'd recommend making sure that you incorporate ALiBi, so you can have very large context lengths, much larger than what you train on.
- That you'd leverage **FlashAttention**, so that you don't have to overwhelm the SRAM of your GPU when you calculate attention and allow for much larger context lengths to be used.
- And also leverage grouped query attention, so you can save on the amount of compute resources that you have and the number of parameters that you would need for your attention mechanism.
- Depending on your application you might also be interested in pursuing a mixture of experts approach, if you want to have truly vast scales for your large language model.

If you're just focusing on fine tuning and inferencing then you're also going to want to:
- Leverage tools like LoRA or quantized LoRA.
- And Frugal GPT and LLM Cascades if you're interested in how to minimize the amount of cost for the particular budget that you have for inferencing.

There's also been some fantastic work by the community to have a look at some numbers that every LLM developer should know. This work was collected by Ray with any
scale and includes some excellent pieces of wisdom.
In particular, talking about GPU memory, the general guide that if you double the amount of parameters, that gives you a sense of your GPU memory requirements. That is to say that **if you have a 7 billion parameter model, then you're going to want something like a 14 Gigabyte capacity for the graphics RAM for your GPU**.

Note that this is for serving, for training, you'll actually need even more. If we look at the options that we have available at the moment, for the v100s, the a10g, and the a100s that gives us a sense that we could use a 5,10,20, and maybe 40 billion parameter model for each of those as we increase the size of the RAM that's available.

There are of course always going to be improvements in the technology from NVIDIA, AMD, and others so that we'll continue to get better hardware such as h100s and beyond, that's able to provide more resources for us to work with. Do keep in mind that these will obviously come with a price premium and the availability for GPUs is still quite difficult.
