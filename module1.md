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

# Transformer Architectures
Up until now we've been fairly generic about the type of architecture that a Transformer can take. We focused on what the Transformer blocks are, and what they're comprised of. However there's a lot of different ways we can construct a different type of Transformer by the way that we organize these Transformer blocks.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/0f9ed3ad-1da2-4dfd-bbfe-a66c81c390c1)

We're going to look at the different types of common architectures that we see with Transformers and the different use cases and innovations that they require. If we take a look at the current state of the [Transformer family tree](https://github.com/Mooler0410/LLMsPracticalGuide/blob/main/imgs/qr_version.jpg?raw=true) it's pretty big and pretty messy. But we can separate these into three distinct categories. We have on the left here, encoder only models and we'll talk about what it means to encode with a Transformer in just a moment. 

On the far right we have decoder only models and you'll see some familiar names there like Claude or GPT-4, and in the middle we have the encoder decoder models. And if you look carefully you'll see that we have a very familiar G letter representing Google for the encoded decoder models very prevalent in the middle. And there's a very good reason that Google is so prevalent in the encoder decoder model space.

## The Original Transformer
In the original Transformer paper titled "Attention is all you need", the researchers from Google presented an architecture based on an encoder decoder approach.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/5c751927-2826-4bcb-9eac-18c81e443299)

The reason for this is that they wanted to do machine translation between English and German. The goal there was to input a sequence of English tokens and output a
translated German sequence at the end.

The way that they achieve this goal is by taking an encoder series of blocks, so these would be regular Transformer blocks as we've seen so far, they would put in the English tokens, transform them and prepare them in the way that we saw in the previous section, and then at the end of the Transformer blocks the vectors that we have at the output of the different sequence vectors that we produce after they've gone through the Transformer blocks are actually used for the attention mechanism in
something called **cross-attention** for the decoder side of things that they presented in their model.

Now the way that this would work is that the model would first look at the words that it had produced as the decoder side of things, and then when we move up to the point where cross attention is needed it would compare the word that it had at the middle of its Transformer block and look at the cross attention vectors from the encoder side of things.

We'll look at how attention takes these different types of vectors and combines them together in just a moment, but you can think of it as first the encoder takes the English language and transforms it into some sort of enriched vector and then it uses those enriched vectors and learns how the German words relate to the English words to be translated.

So encoder decoder models typically take one type of language task and convert it to a different type of language. This could be translation or conversion or it
might be some kind of halfway in between such as taking input from English or natural language of some kind and outputting it as say code language, or it might be one programming language to another programming language or it might be summarization, there are a number of different use cases for encoder decoder models and they're based on the concept of cross attention we'll dive deeper into what cross attention is and how it can be used later on when we talk about the attention mechanism in detail.

But essentially, what the encoder does is it provides an extra source of signal for the decoder so that it can achieve the task that it is given and that during back propagation it learns to rely on the signals from the encoder to achieve its task. The next part of the Transformer architecture family is the encoder model.

Now Google also produced a second Transformer architecture a couple of years after the original Transformer was released and this was the bi-directional encoding representations from Transformers or BERT. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/124d7888-070e-4c73-abf7-1fcc3d8ad9fe)

There were a couple of new innovations that BERT released with, one was segment embedding so you could take one sentence
separate them with the [SEP] variable and then put in a second sequence or a second sentence and BERT would be able to compare the two sentences together The way they trained BERT was also different as they would intentionally mask different words into the sentence it would also allow you to incorporate next sentence prediction and
by that it would be able to tell whether or not the next sentence preceded was preceded by the first sentence that it saw it could give a true or false whether or not the sequence the first sequence it saw led then to the second sequence or not.

BERT was excellent for fine tuning and has been used and still dominates many of the state-of-the-art techniques for different types of natural language processing. BERT is excellent for things like question and answering, named entity recognition, and other more traditional types of natural language processing tasks.

BERT is still in use today and is much more lightweight than some of the larger models that we typically see in the news.

# Generating text with GPT
Speaking of these types of models, the third type of architecture that was produced based on the Transformer architecture are known as the **decoder only models**, the most popular and well-known version of this is GPT **GPT: generative pre-trained Transformer** is a type of Transformer that, as the name suggests, generates new words. You've probably heard of the buzz term 'Generative AI' and GPT is the reason for that buzzword.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/a9d37a27-8524-4250-ae9a-4b3def68a1c6)

**The whole aim of a decoder only model is to try and predict the next word based on the sequence that it's currently processing**. In GPT it'll take in all of the vectors that it's been working on and enriching and use the classification softmax layer at the end of the Transformer blocks to try and predict the next token or the
next word.
We've seen a huge amount of applications based on these GPT or decoder-based models and you'll be familiar with probably ChatGPT, Bard, Claude, LLama, MPT and the list goes on.

# Important variables in Transformers
![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/8939daaf-3aea-4408-8b59-b6071849a12c)

# Time to Pay Attention (The secret that unlocked the power of LLMs)
One of the goals of this module is to understand how we can build and train our own base or foundation Transformers.

However, before we get into that let's take a moment to talk about attention. It's one of the most important components of Transformers and something that can be quite complicated if you haven't seen something like it before. 

## The inner workings of attention (Learning the weights of attention.)
To start with let's think about how we can take the vector that we're working with, that's going to be the current token that we're looking, at so let's assume that we're in the first layer where we can directly correlate the input word embedding vector with the vector that we're going to talk about in attention here.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/ac431d0f-36ad-4cb0-a2dc-9e9cf471b7b1)

Now attention is built out of three vector families: the query vector the key vector and the value vector.

Now we actually have one query vector and that's going to relate to the current token that we're looking at in the sequence, we're going to have a number of key vectors, they're going to come from all of the vectors in the sequence, and we're also going to have a series of value vectors.

We're going to use a matrix multiplication with the word vector or the enriched embedding vector if you like, multiplied by this query matrix to give us our query vector. All of the matrices, the query matrix, the key matrix, and the value matrix are comprised of weights that are learned during back propagation.

The idea behind attention is that we use a single query vector and talk to all of the other key vectors that we generate in this from the sequence and we effectively ask it how much are you, the key vector, related to me, the query vector. We do this in parallel for all of the tokens in the sequence so every time we do the attention
calculation we're focusing on our query vector and we're broadcasting the query vector to all of the keys, by that I mean the key vectors.

What we're doing is we're asking how similar, how important is this key vector to the query vector. In the equation that you see here you can see we take the softmax of Q times K transpose. Now Q in this situation is the query vector and K is a matrix transposed here to make it so that we end up with another vector and we multiply that by the value vector.

## The inner workings of attention (How do we calculate attention?)
Let's take a look at what happens in this situation. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/699445c1-d9f8-4094-ae9d-a3191e28d279)

So to calculate attention step one, we take our input vector, which if we're in the first layer is the word embedding vector with positional information, and we create three new types of vectors. We create the query vector, the key vectors. and the value vector.

The query vector as I said before is just built from the current token, we then multiply that using a scaled dot product on the query vector to all of the key vectors and what this gives us are attention scores.

We're going to have an attention score for each pair of the current query vector to each of the key vectors, so we'll end up with an attention score vector which is the same length as the query vector, which is the same length as the word embedding that we get from the token. This is another reason why the dimensionality of the model that we built, so if we remember from the previous sections video we talked about important variables for Transformers the modal dimension or the model size is
very important here.

The size of these vectors is the dimensionality of the model, so the query times key vectors gives us these attention weights and they're scaled from zero to one.
And then we do a special type of multiplication so that for each position in our vector the attention weight is multiplied by the value of the value vector at each of those indices.
So from zero to the size of the embedding we multiply a simple scalar product between the attention weight score at index zero with the value vector score at index zero and we do that for each of them. This then gives us a full output vector of the attention score for that particular token across the entire sequence.
We'll take a moment to think about this one more time as attention can be somewhat complicated.

Realistically you can think of this as some kind of filing cabinet and lookup system where we have our query which comes from the current token and we're looking through the files to see how well each of the different other files these are the key vectors have the information that we need that would be the value vector.
Once we've figured out exactly how much each key vector should give to the query vector, and that's the attention weights, we then combine it all together so we
get a full picture this will be our output vector we get a full picture of how much attention to pay to each other token in the sequence. And so this is where the notion of attention comes from, is that the value in each of the parts of the output vector the value in each of the parts of the output vector tells us how much attention we should be paying to each token relative to the current token of focus.

# Building Base/Foundation Models (Training transformers, what does it take?)
We've seen now, all of the different building blocks that go into the construction of creating a Transformer large language model. You're then probably wondering how do we build the actual model to be useful, how do we train it and construct it, and what pieces do we need to actually get everything working. In this section, we're
going to go through all of the different components. Including the data, the compute, and the training procedures that you'll need to follow if you want to pursue training your own large language model from scratch.

## Foundation Model Training - Getting Started (Choosing the right options to build your model.)
One thing to keep in mind, you may hear the terms foundation or base model interchangeably throughout this course and in the wider literature. These refer to large language models that are trained from randomized weights into just predicting the next word. Now you sometimes see foundation models or base models behave in ways that you might think oh that's not actually what I want my model to do and that's just because what happens when we train a foundation model or a base model is it's understanding fundamentally the syntax and potentially the semantics of what language contains.

So you might ask a foundation model or a base model what's the capital of France and rather than answering with the actual answer that you'd expect it might then ask what's the capital of Germany and that's because it's more often than not seeing a list of questions rather than a question answer in the training set.

We'll talk about the different types of training sets in just a moment but keep in mind that for task specific performance you'll almost always need to fine tune your model. This requires a much smaller amount of training data and is usually recommended for most people, as they'll take a Large Language model that's been pre-trained
or produced a foundation version and then they want to fine-tune on top of that.

However for those of you brave enough to get started training your own foundation model, let's look at some of the different options that you'll need to go through.

You'll want to think about the model architecture, whether it's a decoder an encoder or a combination of the two, and you will also want to think about the type of tasks that you want the fine-tuned version of this foundation model to perform, as well. This will inform some of the decisions that you make with the structure of your model and also the different types of data you'll also want to make sure that you think about how big you want the model to be how rich you want its representation of language to be and needs to be embedding Dimensions the number of blocks Etc.

And then the type of data and the availability of data that you have and the wrangling of that data most importantly is going to be one of the most difficult things for you to overcome.

Finally actually getting the compute resources, both the amount of time that you have allocated to train the model and the hardware that's available, which is not something you can take for granted these days as GPUs are quite hard to come by, particularly the ones that are needed for foundation model training.

# Foundation Model Training - Architecture (Which transformer flavor is right?)
Let's think about the different types of architectures. We've seen already the encoder decoder model that Google produced in the attention is all you need paper and the
different generations that came after that such as BERT, GPT, and T5 which we haven't spoken about but we'll look at more later in the course.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/fc8495f4-8501-4afc-a67d-7cac01293cf2)

Depending on the tasks that you want whether it's classification maybe you'll go with something like BERT if it's generation you'll probably want something like GPT,
translation you'll probably want an encoded decoder like T5.
You'll also want to think about the numbers of layers that you have and the context size that you can deal with. 

# Foundation Model Training - Data (It’s all about the data)
Most importantly the data is something that you'll have to fight for. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/93a35d3b-d530-4e0d-b5d1-609588e683d5)

There are a number of publicly available data sets such as the well-known **Pile** data set which is a combination of different openly available text resources. However if you train the same model that someone else has trained on the same data although you're not going to get much of an advantage and you're better off just downloading the weights of that model. You'll want to start at least with something like the **Pile** to get a good understanding of, at least in this case the English language, but language in general.

You then may have proprietary or just curated data sets of your own that are more specific and might include things like transcriptions, digitized text, code examples, and other sources that you think is valuable for this foundation model to be trained on.

## Foundation Model Training - Training (Optimizing LLM Losses)
Once you have all of your data and your architecture and the compute ready to go then you can breathe a sigh of relief as now you're just back to training a regular deep learning model.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/6ddf809e-5cf5-4339-9217-9d14b1d725f9)

Large language models train more or less like every other deep learning model except they're massive, they take many weeks and months to train to a reasonable state and they often require hundreds of GPUs to do so.
However they often rely on fairly typical loss functions like cross entropy and optimizers like AdamW. Though new optimizers are being researched and developed by the community day by day.
Now that your model's been fully trained you might then wonder what do you do now.

## Now what? (What do you use a foundation LLM for?)
Well odds are if you start to interact with your large language model you'll find that it suffers from alignment problems. If you're unfamiliar with the alignment problem in large language models, in essence it boils down to just a few components.
Is the model accurate. Does the model behave well for what we want it to do, is it toxic, does it show negative biases or any sort of biases that would detract from the performance that we want. And does it hallucinate, does it make up situations and examples when we need it to be as factual as possible or maybe you want it to be as creative as possible and it's just not very good at doing that either way the problem of alignment is still an ongoing area where different types of tools and procedures are being investigated.

Really what you want to do after you've built your foundation model is to look at fine tuning methods.

# Generative Pretrained Transformer (A journey to discover how GPT-4 and ChatGPT were built.)
As its name suggests, it's a generative model so it's a decoder model and it's a pre-trained Transformer, so that just means that it's been built and trained like we looked at in the previous section.

In this section we're going to look at how we went from GPT-1 all the way up to GPT-4 and how ChatGPT works. 

## The journey to ChatGPT
ChatGPT was released in November of 2022 and **was a fine-tuned version of GPT-3 or GPT-3.5** depending on when you started using it.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/533a4bff-3ae9-4754-8ca7-b8c3ec44e970)

ChatGPT was one of the first chat bots using a large language model as its base and received renown acclaim and usage being, as we said at the start of this module, the most adopted technology in human history. ChatGPT is an application of a decoder-based transformer model and let's see how we actually got to this point.

## Generative Pre-trained Transformers (GPT): Decoder-based transformers
The generative pre-trained Transformers or GPTs were a family of models that were researched and released to the wider community after the release of the attention is all you need paper by Google in 2018. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/d5ff1a84-f4b6-4cce-89f9-4a0bda8cc0e6)

The first GPT model looked at the encoder decoder model of the attention is all you need paper and decided to just work on a decoder model so they were just looking at the right hand side version of the architecture.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/82490757-3697-4a7b-b38f-28d26153320f)

The different families that were built after that original GPT so this is GPT-2, -3 and -4 have more or less relied on the same architecture but just in larger versions trained on even more data. Now there have been very important innovations and clever changes that have made the scaling up possible but in essence all of the infrastructure and architecture that we've been talking about have been exactly what they've used to build GPT-4 and ChatGPT.

GPT-1 started with just **12 Transformer blocks** each connecting to each other and passing those enriched vectors. GPT-2 increased the dimension size of the word embeddings to **1024** and also quadrupled the number of Transformer blocks. GPT-3 doubled the amount of Transformer blocks and also doubled the model embedding size as well.

GPT-3 was truly transformative in its ability to perform few shot learning and complete tasks at a state-of-the-art level.

## Generative Pre-trained Transformers (GPT): Generational Improvements
The data was also something that changed as we went from the different families of GPT. the original GPT or GPT-1 was trained on something called the **Book Corpus**.
GPT-1 (2018) had 117 million parameters more or less on par with the size of BERT which was also released around that time too.

GPT-2, which came out one year later (2019), was trained on a much larger data set called **WebText**. This text was gathered from the publicly addressable web, and also was the first time that we saw a Transformer model released in different sizes.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/df797747-bde4-432b-8e2d-38d49f557d19)

Starting with the 117 million parameter model with the same number of parameters as the original GPT all the way up to the 1.5 billion extra large version of GPT-2.
GPT-3, which was released in 2020, was pre-trained on an even newer and larger data set **WebText2**, which incorporated 45 terabytes of text.

GPT-3 (2020) started off with **175 billion parameters** and was found to be exceptional at few shot and zero shot learning capabilities. GPT-4, which we only know some rumored information about, was released in 2023.
While OpenAI has not released any of the information to date on how GPT-4 is structured some estimates are that it is a number of smaller **220 billion parameter** models using a mixture of experts approach.

Now we don't know if this is true or not but we will be covering what mixture of experts is in Module 3. 

# GPT Architecture
Let's look even deeper into the GPT architecture. So why do GPTs keep getting bigger every time we see GPT-1 , -2, -3 and -4 increase in size, this is largely because it has more and more layers while the model dimensions do get larger too they don't scale in quite the same way as the number of layers do.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/d521193c-015a-4816-bc7a-8c2dbf805899)

Now the reason that we want more and more layers is because that allows the attention mechanism to focus on more and more aspects. When this works in conjunction with the position-wide feed forward neural network what we enable is conceptually within the model it's able to see more and more about the text.

If you think about the convolutional neural networks, they started off looking at edges and then eventually as we get deeper into the layers they could look at more
composite features, something akin to this is happening in attention.

The early attention is looking at things like word order different parts of speech basic sentence structure, and then as we allow the neural network to reconfigure the vectors into a useful format for the attention block of the next Transformer block to pay attention to we're allowing the model to pay attention to different types of text and speech and so in the middle layers of attention in the middle numbers of blocks we can think of the model as being able to pay attention to things like meaning
relationships between different phrases in the text rather than just within the specific sentences.

And then at the last stages of the attentions in the Transformer blocks you can think of it as looking at the discourse the sentiment and complex long-range dependencies. When we think about how you interact with ChatGPT and other platforms today the context length is increasing more and more and with the number of
layers also needs to increase so that it can pay attention to more of what's going on in the text and understand more richly what is being conveyed.

## Why so many parameters?
And so when we think about why we're getting so many parameters up to potentially a trillion parameters in GPT-4 this comes from the fact that the layers are increasing the model dimensionality is increasing and the number of attention heads as well there are some other factors and the attention heads per layer doesn't play a larger role as it perhaps could but the number of layers and the modal size play a huge role in both the neural network size and the size of the attention matrices.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/98c341f2-8552-435c-a56a-8276c956d547)

## Training GPT
So let's talk about how you might train GPT and focus a little more on that data as we mentioned before GPT-1 was trained on something called the Book Corpus which is comprised of 7000 unpublished books spanning various genres which has about 800 million words covering a wide range of topics and styles. If you download GPT-1 from something like Hugging Face and run it you'll notice that it has this kind of style in its output.

You'll also notice that it tends to output more or less nonsense and that's just because GPT is very small and doesn't have quite the complexity that it's larger newer Generations have.

GPT-2 was trained on something called **WebText**, and this was the first time that we saw a publicly crawled data set used in large language models. This is much larger than Book Corpus and as you can imagine it encompasses a much wider array of what people talk about on the internet. Instead of the 800 million words WebText is comprised of 45 terabytes of data.

This also meant that the team at OpenAI had to do a lot of work in deduplication and filtering out web pages with low quality content. This is still a problem that exists today finding good sources of data and focusing on cleaning that data and making it making sure that it's the highest value data for the model.

For GPT-3 **WebText2** was used, which is even larger and more diverse in the original WebText data set. 

if we look at the performance differences between GPT-2 and GPT-3 we can see that this new data set, even with the same number of parameters, enables GPT-3 to perform much better on the tasks that it's given.
As we move forward into the realm of large language models into the latter half of this decade we'll also see the need for new sources of data whether it come from video transcriptions or other synthetic data sources.

# Comparing LLM Architectures
So now that we've looked at how GPT went from GPT-1 to GPT-3 and all of the different innovations and changes that needed to take place let's take an overarching look of where we've come so far in learning about different Transformer architectures.

## BERT vs. GPT vs. T5 : Which type of LLM is best?
We've seen the different types of architectures that you can create using your Transformer blocks these could be the encoder only models such as BERT decoder only models such as GPT and sequence to sequence Transformers like T5.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/8039b1f7-12fb-4a11-976e-116064d2f99c)

Each of these have different pros and cons and depending on the task that you want to apply your large language model to and the resources that you have at hand, you may need to be strategic about which you which you choose while you may just run and put everything into GPT-4 or -5 and probably get some decent results that also costs a lot more both in compute security and other limitations that you may not be able to afford.

If you're just looking at things like sentiment analysis or if you need to control the data or if you have a very small amount of compute you may be better suited at utilizing something like **BERT or T5** instead. take some time when you can to have a look at the different pros and cons for each of these different Large Language
model frameworks to see which one is best for you.

