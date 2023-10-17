# Beyond Text-Based LLMs: Multi-Modality
This is going beyond text data to allow your model to take in different modalities of data and passes them all together to produce useful output. So to me this is one of the most exciting areas of potential for language models and deep learning, in general, is being able to consume all the data that's out there in the world in other formats and do something meaningful with it. You can imagine a lot of applications of this. Of course you can imagine in many domains, you have, you know, visual data like MRI's or stuff like that that your model could look at and you know look at it, in addition to text content, to do something meaningful. 

You can imagine various structured data inside an enterprise sensor data. You know you could imagine a model that takes in all the sensor readings from like, say all the sensors on an airplane, and can do something special with those time series data and video and so on.

There are so many different sources of data that we can feed into these AI applications and one of the really exciting things about the transformer architecture is that it is a pretty general architecture. You've got these input tokens that can represent a wide variety of information and you've got this general mechanism of attention that you can use to look at a bunch of them, so the input doesn't have to represent the text sequence: it could represent an image; it could represent audio, you know, it could be all kinds of other things and it could be, you could imagine learning you know a joint representation across these different kinds of tokens that let your model to do something good you know across both of them. 

# Going beyond uni-modality: LLM-based models that can receive and reason with multimodal info
Some of the most commonly cited examples of multi-modal language models come from OpenAI.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/48fed94a-bc0f-4229-b9b5-4872d1c4386a)

The first that you might have heard about is **Whisper**. It can transcribe speech audio into text. The second is **DALLE**. DALLE can create images from text and for **CLIP**, when it's given text the image descriptions, it can predict the most relevant text description for that image.

# Multi-modality mirrors how we perceive info: More user-friendly, flexible, and capable
Multi-modal models are very useful and in fact, they are much more user-friendly and much more flexible. And they pretty much mirror how humans can perceive information, for instance, you can see this [VideoLlama](https://huggingface.co/spaces/DAMO-NLP-SG/Video-LLaMA) application, where we can ask [VideoLlama](https://arxiv.org/pdf/2306.02858.pd) to describe what it has heard in the video itself and then ask questions to follow up on the video. 

You can also use [MiniGPT-4](https://arxiv.org/pdf/2304.10592.pdf) to explain memes that you might not understand or even ask MiniGPT-4 to come up with some mock-up code for you to build this joke website and insert jokes within the website as well.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/c68239aa-6486-442c-8fa8-c733a797982a)

# Chain-of-Thought MLLMs: We can also supply multi-modal information as “in-context”
Just like large language models, multi-modal language models can also exhibit Chain of Thought reasoning capability. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/e25c66af-6a62-4695-9ada-55671a128029)

We can supply multi-modal information as in-context. For example, on the [left image](https://arxiv.org/pdf/2305.13903.pdf) over here, we can supply a video, a series of video frames and ask that multi-modal language model to explain to us what happen between the frames.

And the [second image](https://arxiv.org/pdf/2302.00923.pdf) that you see on the right, we can provide photos as our in-context; for example, the crackers and soda, and ask the multi-modal application to tell us, you know, which property does these two have in common?

# MLLMs can process multi-modalities simultaneously
Quite impressively, multi-modal language models can also process these multi-modalities simultaneously. So we can pass in an image. We can also pass in a recording at the same time and ask questions to the language model.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/235c8449-2419-4dc6-9f35-5374614fe6ac)

# MLLMs also call tools/models to finish tasks
Multimodal language models can also act as an agent that calls upon other tools or models to finish tasks but a lot of these multi-modal models use Transformer architectures. But how? We have seen so far that Transformer architectures mainly process text.

https://huggingface.co/spaces/microsoft/HuggingGPT

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/816f3f61-ec26-46a9-ba97-6641a5f25166)

# Transformers beyond text
## Transformer: a general sequence processing tool: We can treat many things as a sequence
The Transformer architecture is incredibly versatile. It is a general sequence processing tool and as it turns out, we can treat many things as a sequence, including an image.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/8c90ad71-bc3d-48eb-a468-873fa13f59ce)

This is an image of my cat, an audio file, some music notes, video frames, even a series or a sequence of game actions, and lastly protein as well.

## Cross attention bridges between modalities: Allows different modalities to influence each other
Specifically, the cross-attention mechanism can help bridge between modalities, be it images audio text or neural time series. And in fact, this is what stable diffusion, which is a text-to-image model uses. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/3db9e793-0098-4e56-ac35-53797b6b7773)

So you can see on the right image over here, we can ask [stable diffusion model](https://stability.ai/blog/stable-diffusion-announcement) to generate an image from text and that is using cross attention to bridge between text and images.

# Computer vision
Let's first look at how we can use Transformers for computer vision. How to use Transformers for computer vision has been quite well-researched ever since 2021. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/e7ee82d0-b842-433c-abdd-907b7199c585)

Before then, the de facto architecture for computer vision was **convolutional neural networks**. The models included in this slide represent milestones in the field of computer vision. I won't go over every single model I've outlined here. Rather, I will focus my attention on the **Vision Transformer**, which is the first Transformer used for computer vision that has outperformed convolutional neural networks by almost four times in terms of computational efficiency and accuracy. 

It then spurs on a series of research to apply Transformers to computer vision. We will also come back to zero-shot and few-shot learning in the context of computer vision and Transformers as well. But first, we need to understand how we can represent images as numbers.

# We chop an image up into small pixels: A colored image is made up of Red, Blue, Green (RBG) levels
When we buy a new phone or a new camera, a detail we might care about is the camera resolution, which is the number of pixels in a photo, usually denoted in megapixels. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/b13b3217-dd1f-4fdb-a754-efc7b326e66c)

We can chop an image up into many many small pixels and each pixel will contain information on the primary color composition, namely how much blue, how much green, how much red you are seeing in the image. So you can see in the image in the center over here, for each pixel, we see that there is a HEX code and also the RGB levels, which is what we commonly pass in to neural networks and the pixel values can range anywhere from 0 to 256.

## Colored images are 3-D tensors: Grayscale images are 2-D tensors: all 3 channels have the same value
As it turns out, **colored images can be represented as tensors**. In text processing, we typically represent our text embeddings in the **matrix form, which is a 2D tensor**. For colored images, they are 3D tensors. **The third dimension is the number of channels, representing red, green, and blue**.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/b0017cdd-4477-4849-817a-2d65969cc93f)

**Grayscale images can be 3D tensors too, but since all three channels share the same value, we can represent them as 2D tensors**. This is why we often use grayscale images in large-scale image processing models because they require much less space.

# Initial idea: Turn pixels into a sequence: Use self-attention to predict the next pixel, instead of word token
The initial intuition for pixel processing is to simply turn them into a sequence. We can use the sequence of pixels in both autoregressive context or the masked modeling context like BERT. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/fa0d9d05-9f5b-4534-b1e7-eecc3cd896d9)

But instead of predicting the next word token, **we use self-attention to predict the next pixel**. But there are two limitations:
- first, it's that we lose the vertical spatial relationship between the pixels and this is not hard to see because now when we flatten these pixels into a single sequence from left to right we no longer know if this gray box over a gray pixel over here is directly above the olive green pixel.

# Initial idea: Turn pixels into a sequence: Use self-attention to predict the next pixel, instead of word token
- Secondly, this method of using self-attention incurs really high complexity, O(N^2), because we need to calculate the complexity of each pixel with respect to all other pixels. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/c6f4033f-b97e-4105-8a50-c434149e2677)

Even in a low-resolution image like (256 x 256), we will have over 65k calculations if you take 256^2 across from left to right but we also need to replicate that from top to bottom as well.

So for (256 x 256) image, we would have 10 to the power of 9 calculations for a single attention layer. You can imagine how much more computational resources we need when we have an even higher resolution image than (256 x 256). So this approach is not viable.

# Vision Transformer (ViT): Computes attention on patches of images: image-to-patch embeddings
Here comes **Vision Transformers**. Let's now cover the terminology first.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/35785125-1ed9-4555-b69f-a46a8702225e)

Vision Transformer **represents an input image as a series of image patches**. I hope my cat, Pearl, here doesn't mind I've chopped her up into 16, into 16 different pieces over here. In NLP, we call these patches, the individual patches that you are seeing as **words** or **subword tokens** right.

But for Vision Transformer, it splits an image up into (16 x 16) patches here. But in this image, I'm only chopping my cat up into (4 x 4) pieces. So when you go from left to right, you can imagine that there are 16 patch. And within each patch, there are multiple pixels.

This is why the [Vision Transformer paper](https://arxiv.org/pdf/2010.11929.pdf) is titled "An image is worth 16 times 16 words. So Vision Transformer splits a large image up into a sequence of patches, just like you are seeing here. And **each patch has a dimension of 3, which is number of channels x Pixel number of pixels and number of pixels**.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/ec90bccb-c4f9-4ead-a768-a944cb2dfea3)

After building the image patches, ViT then uses a linear projection layer to match the image patch into a D-dimensional vector. This D dimension of vector outputs are called **patch embeddings**.

Just like similar words should occur in a similar embedding space, similar image patches should also be patched to similar patch embedding space as well. So you can imagine that there is an embedding space over here, where we're mapping different patch embeddings onto that space.

But if you recall from the module, the first module in this course, we learned that Transformers do not have any default ordering mechanism, but we need to enable the model to somehow know or infer the order of or the position of the patches. Therefore, ViT adds positional embeddings. After adding the positional embeddings, the patch embeddings are now complete.

Now here is where we refer to something that we learned about BERT. In BERT, a feature that's introduced to Transformers is the use of this token: the CLS token, which stands for classification token. This token is a special token because it actually doesn't represent an actual token. It begins with a blank slate so the Transformer will be forced to learn to encode a general representation of the entire sequence into that embedding. So ViT also uses the same logic by adding this CLS token, otherwise also known as learnable embedding.

So the output of this learnable embedding would then be used as an input to the classifier so that the classifier can learn to make accurate predictions later. So as you can see over here, we're now passing this entire sequence as an input to a standard Transformer encoder.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/d5d8dad0-20e9-4f52-88ba-25b599cbc395)

Then we pre-train the model with image labels from ImageNet data set, which is what ViT is using. At the very end, the output of the last Transformer block goes through
a classification head, which then gives us an image class prediction. That is how ViT works.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/b6135c60-1ae8-4771-98c5-e98f0cbd9f6d)

# ViT only outperforms ResNets on larger datasets: More computationally efficient than ResNet
We find that ViT only outperforms ResNets, which is another popular convolutional network-based neural network for computer vision on larger data sets. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/52f214d8-8fb7-41ad-bee7-30789cdf9f34)

So you can see that, for ImageNet alone, when the image data set is smaller, ViT performs worse. But on larger data set, ViT outperforms ResNets. Something to note though, is that **ViT is much more computationally efficient to train than ResNet**. In fact, it is four times faster than ResNet to train.

# Many other vision-text models: Not necessarily revolutionary, but an evolution in computer vision research
There are many many other follow-ups to this ViT research using attention mechanism for computer vision. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/8125c34a-9356-4b72-bdfc-3c189e864d45)

One of them is called **[SwinTransformer](https://arxiv.org/abs/2103.14030)** and the next one is called **[MLP-mixer](https://arxiv.org/abs/2105.01601)**. Granted MLP mixer is actually not a Transformer nor CNN, but it inspires many concurrent papers and follow-ups based on the findings from this paper. 

So if you are interested about what MLP mixer is doing, definitely check out their paper. I also wanted to mention that Vision Transformers are an evolution, not necessarily a revolution.
According to this Professor of Computer Science at UMichigan – his name is **Justin Johnson** – he said that we can fundamentally solve the same problems using convolutional neural networks but the main benefit with using attention probably comes down to speed because** matrix multiplication is much more hardware friendly than calculating convolution**. So ViT with the same floating point operations as convolutional networks can train and run much faster.

# Audio
## Audio signals are 2-dim spectrograms: We create embedding vectors for each t-min audio frame
Now let's take a look at audio. Audio can be represented as a function of frequency and time. The same idea applies to audio as well, where we can create embedding vectors for each fixed length of audio frame. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/db8cb4fb-57b2-4642-8ef5-fba8962ca508)

So you can see that as I go from left to right, this tells me the length of the audio and each column will represent the length of each audio frame, so this could be every three seconds, every six seconds, every one minute, and so on.

## Audio is usually much longer than text length: Need to apply convolution layers with large strides to reduce dimensions
**Since now we can simply represent audio as a sequence, therefore an embedding vector, we can also leverage the Transformer architecture**. But what you will find is that audio usually is much longer than text length, so what you see on this slide should look 90 percent or even higher more familiar to you because it's basically the original encoder and decoder architecture.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/eadd5030-8ac2-49ac-9644-8068fa093b8c)

**But it has the addition of convolutional neural network layers after the input layers to reduce the audio input dimensions**. So this is a **[Speech Transformer](https://ieeexplore.ieee.org/document/8462506)**. The authors of this paper also played with using optional modules like adding ResNet or long short-term memory networks (LSTMs) to further process the inputs before passing them to the Transformer encoder block.

# Few multi-modal advances: Also much harder: emotion, acoustics, tone, speed, speaker identification
Compared to computer vision, there are much fewer multimodal advances. For audio, the models displayed on this slide are all Transformers. But for a long time, prior to 2019, audio processing models did not use Transformer architectures.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/768ff822-da13-422d-af2b-af577f69c359)

Most of these models that you see here in fact only focus on either text-to-speech, speech-to-text, or speech-to-speech and this is perhaps not surprising. Because it is hard enough to process audio-only data and extract all the information available, you know, things like emotion, acoustics, tone, speed, how to identify who is speaking, etc.

And the only model that seems to combine multi-modality is [Data2Vec](https://ai.meta.com/blog/the-first-high-performance-self-supervised-algorithm-that-works-for-speech-vision-and-text/) model released by Meta just last year. The other main challenge with producing such models is that it's much harder to procure high-quality multi-modal data, compared to just text data alone or image data alone. So that's what we're going to discuss in the next section: what do training data for multi-modal models actually look like?

# Training Data for MLLMs
## Hand-crafted training data: Text-audio or text-video data is much harder to procure
Compared to text-to-image data and vice versa, text-to-audio or text-to-video data are much harder to collect. In fact, many researchers have to manually curate them from scratch. Here are a few examples. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/c79f293b-78f8-4385-9560-2fd5863aad19)

If you look at the [second image](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data) over here, we see that the annotators have to provide detailed video descriptions frame by frame, denoted by first, next, then, finally, overall. Another group of researchers use the framework F-O, F-A-M-O-S, abbreviated as **FAMOS** to describe the scenes that they see in the picture as well, so they divided up the scene description into either structured and also dense using the same framework.

https://arxiv.org/pdf/2305.13903.pdf
http://maxbain.com/webvid-dataset/

# Instruction-tuned, hand-crafted data

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/14905087-8eed-4c29-b314-8d70f7242f36)

The data can also look like this, where it is organized into a JSON format or a tabular format. The JSON file on the left shows a conversation exchange between a human and also a GPT model, based on a particular image ID.
On the right, we also see similar data but organized into a tabular format. 

# Instruction-tuned, model-generated data: Actually: manually design examples first, then ask model to generate more
You might ask, why can't I ask models to generate data examples for me? 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/50311c9c-2297-4026-b01d-4c2f72b62c83)

The answer is absolutely yes, but the caveat is you need to do some groundwork of providing high-quality examples first. 

You can see in the left image that the researchers did a comprehensive attempt to describe a particular image or scene first, by writing caption, secondly by labeling the objects in the image. And on the right image over here, the researchers also provide several caption examples to a single image that can apply correctly. You can also see another conversation exchange that the annotators have written up. So you can see that this is the level of detail the annotators have to put into, before they can train a good model.

# [LAION-5B](https://laion.ai/blog/laion-5b/): open source image-text data Original data: [Common Crawl](https://commoncrawl.org/); filtered with OpenAI’s CLIP model
The best open-source image-text data set today is probably LAION-5B. Many of the image-text flagship models are trained on proprietary data sets. So this is the first large-scale open-source data set but is released for research purposes only. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/81db76ed-f3b6-4cfa-a152-79db46e67b2e)

It consists of 5.85 billion CLIP-filtered image text pairs; 2.3 billion of those are in English and 2.2 billion of those are in other languages. There is a really important disclaimer though: the images in this data set are mostly copyrighted and LAION doesn't claim any ownership over these images.

Hopefully by now, you see that high-quality data curation especially for multi-modal use cases can be really time-consuming. And it's really non-trivial and yeah that is the key to producing a high-quality model.
The next question on your mind is probably: what if we don't have that much data? or we don't have that much resources to collect good data but still want to leverage these multi-modal models? This is where few-shot learning comes in.

# X-shot learning for MLLMs
Just like language use cases, where few-shot learning has become increasingly popular and desirable, similar research on how to perform few-shot learning for multi-modal language models is also rising on the horizon.

## Computer Vision
We'll look at computer vision first.

### X-shot learning to the rescue?: Gathering multi-modal data is harder than just images or text data
We know that gathering multi-modal data is hard. In this case, using only a few labeled examples is extremely helpful. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/9b0e05c4-07da-4ecd-bf33-d7745e4de0c6)

So in this section, we'll look at one example of zero-shot learning, which is CLIP and then look at one of the most exciting multi-modal few-shot learning models, called Flamingo, released in late 2022.

### Zero-shot: Contrastive Language-Image Pairing: CLIP predicts which image-text pair actually occurs in the training data
CLIP stands for **Constrastive Language Image Pairing**. The core idea of CLIP is that it learns visual representations from the massive corpus of natural language data. CLIP trains an image encoder and a text encoder with a simple contrastive loss, which is, given a collection of image and text, predict which text-image pair actually occurs in the training data. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/ad6731ad-f62e-43df-9c2b-2e3a028a4481)

So the contrastive pre-training involves maximizing the cosine similarity of the encodings on a diagonal of this (N x N) matrix since they are the actual image-text pairs. This is a pretty simple pre-training task that allows CLIP to perform well in a zero-shot setting. At test time, in the second figure, the CLIP model can be seen in action by correctly predicting the dog caption by maximizing the similarity between the word "dog" and the visual information.

# CLIP performs better across settings
What we find is that CLIP can perform better across settings. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/3c59eaa3-b688-46e7-831e-1031689a0805)

You can see that in this image over here, we have ImageNet data set and we also have other maybe more realistic images drawn of a banana in different settings, whether it's a real image, where the bananas can seem to be kind of obscure in images, or whether it's just a sketch, or whether that can be intentional adversarial actions to make it hard, the model harder to identify that is a banana. But CLIP performs well across these non-ImageNet data sets.

But the big limitation is that although we can predict the probability of a caption to see which text is most likely to be associated with an image, it cannot generate text. 

# Few-shot, in-context: Flamingo: Unifies treatment of high-dimensional video and image inputs
So now let's turn to another multi-modal model with few-shot capabilities, called **[Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model)**, released by DeepMind. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/3eea92c0-828d-4798-951f-8e046e26bb5e)

The Flamingo models are a family of visual language models that can take in both input visual data interleaved with text and produce free-form text as the output. 
The second highlight is it uses a **Perceiver Resampler**. This perceiver resampler receive **spatio-temporal** features from the vision encoder and outputs a fixed size set of visual tokens. And then these visual tokens would be used to condition the frozen language model using freshly initialized cross-attention layers that are interleaved between the pre-trained language model layers. So these layers will offer the language model a way to incorporate visual information for the next token prediction task. So let's look at it in more detail. 

# Flamingo bridges vision and language models: Vision encoder similar to CLIP + Chinchilla (Language) accept interleaved inputs
Keep in mind that the goal, the first goal of these authors is to leverage a pre-trained language model so that they don't have to spend more time or compute on training a large language model from scratch. Specifically, they used a model called Chinchilla, which was also introduced by DeepMind. This allows the Flamingo model to have strong generative language abilities and access to a large amount of pre-trained language knowledge. The role of vision model is to extract rich semantic spatial features from the given images and videos.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/effe19f8-374b-4f9b-9c0d-b696abc39d96)

The second goal of Flamingo is to bridge vision and language models harmoniously. So for that, the authors freeze the weights of these models and link them via two learnable architectures.

An important aspect of the Flamingo models is that they can model the likelihood of text_y interleaved with a sequence of preceding images or videos and also preceding text tokens as well.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/b5832c80-335e-4969-b7de-c8945d47c9a6)

So this architecture can enable a wide range of tasks, including open-ended tasks like visual question answering or captioning and also close-ended tasks, like classification.

# Perceiver resampler outputs fixed-sized tokens
Let's take another closer look at the perceiver resampler.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/293afb8e-e45d-4372-8b5e-3c73c750395f)

The perceiver resampler module maps a variable size of spatio visual, spatio-temporal visual features coming out of the vision encoder and then it outputs to a fixed number of output tokens.

The key and value that you see here are simply a concatenation of the spatiotemporal visual features and the queries contain a set of learned latent vectors. On the other end of the resampler is a fixed number of output tokens and in this case, we see five of them.

# Outperforms 6 out of 16 SOTA fine-tuned models: Curated 3 high-quality datasets: LTIP (Long Text-Image Pairs), VTP (Video-Text Pairs), and MultiModal Massive Web (M3W)
Flamingo beats all previous few-shot learning approaches when given as few as 4 examples of tasks. In fact, it even outperforms 6 out of 16 as state-of-the-art fine-tuned models.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/61908955-bbea-45b7-b7ce-a1f310290a70)

A big factor that contributes to this success is that the Flamingo researchers curated three high-quality data sets, reinforcing the message from the previous module that data does make a lot of difference in the model output quality.

# Qualitative inspection on selected samples: Supported input format: (image, text) or (video, text) + visual query
So, let's take a look at some of the selected samples on Flamingo outputs. First you can see that when we are given an input prompt, when we give an input prompt to the
Flamingo model, it can infer what is in the object and also do some reasoning around those objects.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/e27dc2b2-9df5-4284-9608-447040d33048)

Secondly, you can take in image and text and then we can ask a query and then it will conform to the response that it has seen before and generally a response is similar to the previous response format. 

The third one that you see over here is a series of video frames and we can ask Flamingo a question about the video frames. 

# Audio
## Zero-shot: OpenAI’s Whisper: Encoder-decoder transformer: splits input audio into 30-second frames
Now that we have discussed few-shot learning multi-modal models for computer vision, what about audio? Again, the most commonly cited zero-shot example for audio is OpenAI's **Whisper model**. Unsurprisingly, it uses encoder-decoder Transformer architecture.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/89c1a92f-99dc-4ec4-b0e2-b1711b71c939)

It also uses convolutional neural network to reduce the input audio dimensions as well. And in Whisper's case, the input audio is split into 30-second frames. 

# Whisper matches human robustness: Without fine-tuning on benchmark data
![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/5720d60c-1a40-4ff2-9c3b-f657f88524a6)

The researchers compared Whisper to other models that have been fine-tuned on [LibriSpeech](http://www.openslr.org/12), which contains a thousand hours of read English speech. They find that Whisper has much lower average word error rate and can even match human robustness in a zero-shot setting.

# Challenges and alternative architectures
We saw in the previous section about the potential of multi-modal language models. However, we really haven't figured it all out yet. 

##  MLLMs are not immune from LLM limitations: They inherit LLM risks
Multi-modal language models are not immune from LLM limitations. In fact, they inherit a lot of the same risks. First is hallucination. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/d820e06c-31bc-4cc3-ac4c-aea8e8b8e59a)

You can see over here, on the right image, this is output by the Flamingo model, where it completely hallucinates what is in the image. For instance, when asked what
is the message, what is on the phone screen, the Flamingo output says a text message from a friend.

The other LLM limitations like prompt sensitivity or context limit and inference compute costs all apply to few-shot learning cases for multi-modal language models as well. Unsurprisingly, it also inherits the same concerns for bias and also toxicity. 

For instance, this is one of the latest examples that we see on the internet, where an Asian woman asks a model to make a photo more professional but it made her photo more white. And what is interesting is that this model that the company has used, uses fine-tuned Stable Diffusion model and basically uses the best possible open-source text image data set out there, which is **LAION**, so it's very likely that this problem is not unique to this particular company.

And of course, we also have all sorts of copyright issues as well. Just like for LAION, they don't claim any ownership over their copyrighted images. A similar concern or question also arises for Reddit as well. Since we have been using Reddit data set to train a lot of large language models and back in March or April, they also post this question, that they want to get paid for helping to teach big AI systems.

# MLLMs can lack common sense (like LLMs)
![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/04b7cc17-8b63-438f-98a4-33a269725c09)

Multi-modal language models can also lack common sense as well, where we ask models to generate some image based on text, the output can be in completely nonsensical.
And you can also see that we see the same problem with LLMs as well when we ask GPT-3 to complete this prompt, something about cranberry juice but you have a bad cold but now GPT-3 says that you should drink it but you will be dead.

# Attention may not be forever What may remain or rise?
So what's the next thing that we'll build to improve upon our existing models? It's possible that a lot of these challenges will be really hard to resolve and might never go away but they are alternative architectures that we can consider other than the attention mechanism. Attention and particularly self-attention has been a focal point of the field in NLP, of NLP, and even for the field of computer vision. But it looks like there are actually other existing or emerging promising contender architectures. 

I won't cover the nitty-gritty about all of the architectures. My goal here is to bring them into your attention because who knows what might stick?
So first let's talk about what might remain and become an integral part of every application.

# Reinforcement learning with human feedback: Human feedback trains the reward model (LM). KL loss ensures minimal divergence from the original LLM. Proximal Policy Optimization (PPO) updates the LLM.
It is none other than RLHF, which stands for reinforcement learning with human feedback. No matter how good models get, perhaps the best way to build a reliable model in production is to always involve humans in the loop. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/d3438a70-2221-4194-9cfd-dc988605c1e5)

Here, the human feedback is used to train the reward model and the reward model is typically another language model that outputs a label or a ranking or score that is preferred by a human. On the other hand, there is also KL loss to ensure that the fine-tuned model doesn't diverge too much from the original pre-trained model.
So this reward model would encode human preference and assigns a quality label to the model output.

Finally, the **proximal policy optimization**, abbreviated as PPO. The algorithm will update the pre-trained large language model based on the reward signal. This is a really huge topic on its own so I encourage you to read about this topic in your own time.

# Hyena Hierarchy: Convolutional neural networks are making a comeback?
Now let's talk about the possible next "in" architecture. It's called **Hyena Hierarchy**.

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/59cff5d3-c61e-4a47-b5c4-898a3f1da97f)

It uses convolutional neural networks rather than Transformers or attention mechanism. And the researchers found that it makes a pretty good few-shot model for languages and it also matches the performance of Vision Transformers so maybe CNNs will be making a comeback.

# Retentive Networks: A new attention variant: a retention mechanism to connect recurrence and attention, without compromising performance
The second architecture that has gained quite a bit of coverage in NLP newsletters these days is **retentive networks**. 

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/980e4b23-82b0-4f56-a424-9f0c5ba3eee9)

The authors propose a new variant of attention, which is **retention mechanism**. This retention mechanism can connect both recurrence and attention. But what is special about this mechanism is that it is able to achieve higher computational efficiency without compromising model performance. And this is typically a really tough challenge to crack:
**how do you have good model performance without it being too slow and vice versa?**

# Emerging applications: It’s a great time to be alive
It is indeed a very exciting time to be witnessing the emergence of previously unimaginable AI applications. .

## DreamFusion: Generates 3D objects from text caption
First up is **DreamFusion**. It can generate 3D objects from text caption. 

## Make-A-Video: Generates video from text: “Cat watching TV with a remote in hand”
In a similar vein, Meta released this Make-A-Video application where you can generate videos from text.

## PaLM-E-bot: Robotics application: “bring me the rice chips from the drawer”
We also see PaLM-E-bot, where Google has taken PaLM, which is a language model to integrate with robotics applications.

## AlphaCode: generate code
Something that is probably closer to you and me in the industry right now is using large language models to generate code. And here we have AlphaCode to solve
the problem of the minimum number of minutes to make pizzas with n slices.

## Multi-lingual models: Bactrian-X: An instruction-following model

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/72604a76-0431-41ae-b3d4-14cf45558c57)

It is also really cool to see some progress finally from multi-lingual models, especially those that cover lower resource languages like Bactrian-X. 

## Textless NLP: Generate speech from raw audio without text transcription
And it's very exciting to see more and more audio applications, particularly [this](https://ai.meta.com/blog/textless-nlp-generating-expressive-speech-from-raw-audio/) textless NLP application, where you can generate speech from raw audio without any text transcription.
This is a pretty big deal because it can reduce the challenge of not having enough low-resource language data available for training multilingual language models.

# AlphaFold: Uses attention to predict protein folding

![image](https://github.com/vivekprm/LLM-FoundationModels/assets/2403660/b0296448-56df-4f2c-9e66-ef71be13e9b2)

We're also starting to see the Transformer architecture making waves in biological research because remember you can treat protein sequences as a sequence that can be easily passed into a Transformer architecture.

# Gato: a generalist AI agent: Can play Atari, caption images, chat, stack blocks with a real robot arm
Lastly, who knows maybe 10 years from now, we might all have a robot in every household that can do everything for us and with us, including playing games, chatting with us, and stacking blocks because we all need that?

# Module Summary
- MLLMs are gaining traction
- Transformers are general sequence-processing architectures that can accept non-text sequences
- MLLMs inherit limitations from LLMs
- Transformers may not be the last architecture standing
- More exciting and unimaginable MLLM applications are on the horizon
