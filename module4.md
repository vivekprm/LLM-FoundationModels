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
