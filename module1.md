![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/6f7c5f1c-d044-4168-a2ca-b12a019b2890)The big innovation that unlocked the power of Large Language Model was something called the "attention mechanism". 

**Attention as the word implies, allows a computer, or transformer in this case, to see exactly how one word relates to the others in a certain sequence**. It gives a score of how important each of the word is in a sequence to each other. To you and me this seems like an obvious concept, it's something that we developed early on in life but it's vitally important piece for natural language processing to unlock abilities that it wasn't able to achieve before.

# Attention is (not) all you need
Now while attention was a huge step forward in our abilities to master natural language processing, it was actually only one piece out of a few that were required to 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/767db0da-cbfe-4dbd-922f-2b93be976e4a)

build the tranformers and the models that we see these days.

# The Transformer Block
Transformers like most language models are used to try and predict the next word. The way Transformers do this is different to the way most other language models do it
but they still work on the same underlying principle.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/e2bf364d-ff4f-44f2-ba86-1b440d6ac304)

They will take in a sequence of tokens and then do something with that sequence change the information inside that sequence so that that sequence can be used to predict whatever the next word or token is in the vocabulary.
In a Transformer we take the input tokens and convert them to word embeddings, so we have a vector of different word embedding vectors. We then go through a series of enrichment phases so that those vectors get transformed and build in more and more context and more information for each vector so that when we finally give it to our softmax classification layer or a prediction layer it has a lot of information to work with.
