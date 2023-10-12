![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/6f7c5f1c-d044-4168-a2ca-b12a019b2890)The big innovation that unlocked the power of Large Language Model was something called the "attention mechanism". 

**Attention as the word implies, allows a computer, or transformer in this case, to see exactly how one word relates to the others in a certain sequence**. It gives a score of how important each of the word is in a sequence to each other. To you and me this seems like an obvious concept, it's something that we developed early on in life but it's vitally important piece for natural language processing to unlock abilities that it wasn't able to achieve before.

# Attention is (not) all you need
Now while attention was a huge step forward in our abilities to master natural language processing, it was actually only one piece out of a few that were required to 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/767db0da-cbfe-4dbd-922f-2b93be976e4a)

build the tranformers and the models that we see these days.

# The Transformer Block
Transformers like most language models are used to try and predict the next word. The way Transformers do this is different to the way most other language models do it
but they still work on the same underlying principle.

They will take in a sequence of tokens and then do something with that sequence change the information inside that sequence so that that sequence can be used to predict whatever the next word or token is in the vocabulary.

In a Transformer we take the input tokens and convert them to word embeddings, so we have a vector of different word embedding vectors. We then go through a series of enrichment phases so that those vectors get transformed and build in more and more context and more information for each vector so that when we finally give it to our softmax classification layer or a prediction layer it has a lot of information to work with.

## Transforming a sequence
If we look at the Transformer Blocks in which the tokens are enriched and then moved into the next sequence, we can see what actually happens in the schematic.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/e2bf364d-ff4f-44f2-ba86-1b440d6ac304)

If we're considering a Transformer with just one Transformer Block, the process would look something like this: we would take an input sequence of tokens, transform them into word vectors so that we would have a series of word vectors.

We would then add extra information, like positional encoding which we'll talk about in a moment, which gives information about the relative positions of each token to each other token in that sequence. And then we pass that into the Transformer Block.

The goal of the Transformer block is to enrich every token in that sequence with as much contextual information as possible. It does this through both the attention mechanism and with the transformations using a neural network. We then process it further by adding a residual connection and normalizing the vectors in the sequence
and then those vectors are used in a linear and softmaxx combination at the output of our Transformer to then try and predict the next token or classify the sequence.

Most Transformers will have many hundreds of Transformer blocks but the process is exactly the same, at the end of the transform block the sequence now bearing little resemblance to the natural language input that we started with but still with the same size and format is passed on to the next Transformer block.

Let's look step-by-step at how these preparation, enrichment, and prediction stages work. 

## Attention Mechanism
So firstly we have the all-important attention mechanism.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/882c1841-017f-419a-9b4e-5be239fb1960)

Now the **role of attention is to measure the importance and relevance of each word compared to each other word in a sequence**. This concept gets stretched slightly
as we go from one block to another block and higher in the traditional Transformer architectures we might have dozens if not hundreds of Transformer blocks and so after the first block you're not really comparing one word to another but you are looking at the exact same sequence and positions that each token started with.

They have extra contextual information given to them by the previous blocks and we're still looking at how different blocks pass these sequences differently.
By doing this, by adding more and more layers of Transformer blocks we're able to enrich these vectors and look deeper into how the sequence interacts with each other.

Typical sequence lengths are on the order of many thousands if not tens of thousands these days and so there is a lot of information to be processed and
so a single Transformer block is unable
to capture all of the information in say
a multi-paragraph context.

## Position-wise Feed-Forward Networks
In addition, attention being a linear operation, and we'll see exactly what that looks like in just a moment, doesn't add any of the deep learning so far to this model, in fact Transformers are almost not really spoken about in terms of deep learning, however the majority of the parameters inside a large language model are taken up by the feed forward network that's used at each attention block.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/53a9f593-4754-4c2d-a080-638a222e7fc9)

Now the feed forward neural network in a Transformer operates slightly differently to what you might expect. As we said before, the tokens that are given to the Transformer are turned into word embeddings and they'll have a particular dimensionality let's say 100. And so that means we'll have a sequence of vectors each of them being 100 dimensions long.

The "position-wise feed forward neural network" as they're referred to as in the Transformer will be 100 neurons wide at the input, and so what this means is that we pass each token to the neural network one by one. The weights and the structure of that neural network is identical every time it's applied to each vector in the sequence and so this position wise, which means it moves each position, feed forward neural network is applied to each token so that it can transform them into the right
format to be then given to the next block in the Transformer or to the output block at the end of the Transformer.

This allows for nonlinear Transformations and also enables the Transformer itself to build up different levels of complexity and understanding as we go from the initial understanding of say, how a noun and a verb might be related at the lower levels of the Transformer blocks all the way up to the sentiment of the context right up the
end of the Transformer blocks.

We'll see more of this as we play with the Transformers in the notebooks to follow.

## Residual Connections & Layer Normalization
Another really important part of the **Transformer block architecture** are the residual connections and layer normalizations.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/069fb682-3c5a-4fc4-abe5-ee4b8be3c2bb)

Now the residual Connections in particular are very important because they allow for both gradients to flow freely backward during back propagation and they also make sure that the signal of the input sequence isn't lost during the processing of these vectors as they become more and more enriched, so we have a uninterrupted pathway for the original structure of the input sequence to go all the way through the Transformer.

The layer normalization is also vitally important as Transformers typically take a long time to train and so ensuring that we have stability in our training is something that layer normalization allows us to do.

## Into and out of the transformer
Now let's talk about the input and the output of the whole Transformer itself. Let's start with the input as I said before we start with a series of natural language tokens that are converted to word embeddings and then in order to make sure that we preserve the order of those tokens in the sequence we also attach to our word embeddings a type of **positional encoding**, now there are a number of different types of positional encodings and we'll explore some of these in both the notebooks for this module and in Module 3.

Once we've got our tokens enriched to word embeddings with positional encodings we then pass them into the Transformer blocks they then work on adding different types of enrichment and complexity and hopefully understanding to the vectors in the sequence and then they pass it to the output of the Transformer.

At the output of the Transformer we have our vocabulary and a linear neural network that selects, using the softmax function, which token is either the next token to be generated based on the sequence of vectors that we've been building up in our Transformer blocks, or it'll classify it using some classification scheme that we've developed for the particular application.

Now there's a number of different ways that you can use the Transformer blocks that we've been describing in this section, and in the next section we'll talk about some of those different approaches. Those will include encoder models where we don't actually do any generation of new tokens they're decoder models where we only focus on generating the next token and then there are encoder-decoder models where we take one sequence in and output a completely different sequence based on the task.
