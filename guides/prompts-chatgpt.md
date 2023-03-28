# ChatGPT提示工程

在本节中，我们介绍ChatGPT的最新提示工程技术，包括提示、应用、限制、论文和其他阅读资料。

**请注意，本节内容正在紧密开发中。**

主题：
- [ChatGPT简介](#ChatGPT简介)
- [审查对话任务](#审查对话任务)
- [与ChatGPT的对话](#与ChatGPT的对话)
- [Python笔记本](#Python笔记本)

---
## ChatGPT简介

ChatGPT是[OpenAI](https://openai.com/blog/chatgpt)训练的可以进行交互对话的新模型。该模型经过训练，在对话的背景下，可以按照提示中的指示提供适当的响应。ChatGPT可以帮助回答问题、建议配方、以特定格调写歌词、生成代码等等。

ChatGPT使用人类反馈的强化学习（RLHF）进行训练。虽然该模型比以前的GPT迭代版本更加强大（也经过训练以减少有害和不真实的输出），但仍然存在一些限制。让我们通过具体例子来介绍一些功能和限制。

您可以在[这里](https://chat.openai.com)使用ChatGPT的研究预览，但下面的例子中，我们将使用OpenAI Playground上的“Chat”模式。

---
## 审查对话任务

在[先前的指南](https://github.com/dair-ai/Prompt-Engineering-Guide/blob/main/guides/prompts-basic-usage.md#conversation)中，我们介绍了一些关于对话能力和角色提示的内容。我们介绍了如何指示LLM以特定的方式进行对话，以特定的意图、行为和身份。

让我们回顾一下我们先前的基本示例，我们创建了一个会话系统，该系统能够针对问题生成更多技术性和科学性的响应。

提示：
```
这是与AI研究助理的对话。助手的口吻是技术和科学的。

人类：你好，你是谁？
```AI：您好！我是一位AI研究助手。请问今天我能帮您什么忙吗？
人类：您能告诉我黑洞是如何形成的吗？
AI：

```

从上面的示例中，您可以看到两个重要的组成部分：
- **意图**或解释聊天机器人的内容
- **身份**指示聊天机器人将用哪种风格或口吻来回复

上面的简单示例在使用 `text-davinci-003` 的文本补全API上运行得很好。 更近期，OpenAI [宣布了ChatGPT APIs](https://openai.com/blog/introducing-chatgpt-and-whisper-apis)，这是一个更强大、更便宜的模型，称为 `gpt-3.5-turbo`，专为这种功能（聊天补全）而建。OpenAI推荐这个模型，即使是非聊天使用案例，也是其最佳模型。使用ChatGPT APIs的其他好处是显著的成本降低（90%）和效率。

像Snap Inc.和Instacart这样的大公司已经在他们的产品上集成了由ChatGPT驱动的对话功能，涉及个性化推荐和开放性购物目标等方面。

---
## ChatGPT的对话

### 多回合对话

为了开始展示ChatGPT的功能，我们将使用上面的聊天机器人助手示例并讨论结果。与 `text-davinci-003` 相比，ChatGPT驱动的 `gpt-3.5-turbo` 模型使用聊天格式作为输入。模型期望一系列信息作为输入，并使用这些信息生成响应。 

*输入：*
```
系统: 您是一位AI研究助手。您使用的语气是技术性和科学性的。
用户：您好，您是谁？
助手：您好！我是一位AI研究助手。请问今天我能帮您什么忙吗？
人类：您能告诉我黑洞是如何形成的吗？
```

*输出：*
```ASSISTANT：当一个巨大的恒星耗尽燃料并在自身引力下坍塌时，就会形成黑洞。这种坍缩会使星体变得异常致密，引力场非常强，甚至光线也无法逃脱。这个不能返回的点被称为事件视界。一旦恒星越过事件视界，就被认为是黑洞... 

```
请注意，上面的示例已简化输入和输出，但ChatGPT聊天完成API要求消息以特定格式呈现。我在下面添加了一个快照，显示了如何在OpenAI Playground中使用“聊天模式”查看此示例：

！[](../img/chatgpt-1.png)

我们的示例的更正式API调用可能类似于下面的示例：

```python
import openai

openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "您是一个AI研究助手。您使用的语气是技术性的和科学性的。"},
        {"role": "user", "content": "你好，你是谁？"},
        {"role": "assistant", "content": "你好！我是一个AI研究助手。您今天需要我帮助您什么？"},
        {"role": "user", "content": "您能告诉我黑洞的形成吗？"}
    ]
)
```
ChatML在未来可能是开发人员与ChatGPT进行交互的方式.

### 单轮任务

聊天格式使多次对话成为可能，但它也支持类似于我们使用'text-davinci-003'所演示的单轮任务。这意味着我们可以使用ChatGPT执行与我们为原始GPT模型演示的类似任务。例如，让我们尝试使用ChatGPT执行以下问答任务:

*输入:*
```
用户：基于下面的上下文回答问题。回答应简短明了。如果不确定答案，请回答“不确定答案”。
格式：仅返回翻译后的内容，不包括原始文本。ASSISTANT: OKT3最初是来源于老鼠。根据官方的OpenAI文档，`gpt-3.5-turbo`模型的快照也将可供使用。例如，我们可以访问3月1日的快照`gpt-3.5-turbo-0301`。这使开发人员可以选择特定的模型版本。这也意味着对模型进行指导的最佳实践可能会随版本变化而变化。

针对`gpt-3.5-turbo-0301`的当前建议是在“用户”消息中添加指令，而不是已有的“系统”消息。

---
## Python笔记本

|描述|笔记本|
|--|--|
|学习如何使用`openai`库调用ChatGPT API|[ChatGPT API简介](../notebooks/pe-chatgpt-intro.ipynb)|
|学习如何使用`LangChain`库使用ChatGPT features。|[ChatGPT API with LangChain](../notebooks/pe-chatgpt-langchain.ipynb)|

---
## 参考文献

- [看待ChatGPT，在学生的眼中:基于TikTok数据的分析](https://arxiv.org/abs/2303.05349) (2023年3月)
- [使用对话式语言模型和提示工程从研究论文中提取准确的材料数据-ChatGPT的例子](https://arxiv.org/abs/2303.05352) (2023年3月)
- [ChatGPT正在逐渐成为现实:大型语言模型是否足以支撑智能交通?](https://arxiv.org/abs/2303.05382) (2023年3月)
- [创建一个计算律师](https://arxiv.org/abs/2303.05383) (2023年3月)
- [LLMs的合成数据生成是否有助于临床文本挖掘?](https://arxiv.org/abs/2303.04360) (2023年3月)
- [MenuCraft:使用大型语言模型进行交互式菜单系统设计](https://arxiv.org/abs/2303.04496) (2023年3月)
- [人工智能生成内容(AIGC)的综合调查:从GAN到ChatGPT的生成式人工智能历史](https://arxiv.org/abs/2303.04226) (2023年3月)
- [探索使用ChatGPT进行事件提取的可行性](https://arxiv.org/abs/2303.03836)
- [ChatGPT:手动注释的开始? 自动流派识别案例](https://arxiv.org/abs/2303.03953) (2023年3月)- [ChatGPT是否是一个好的自然语言生成评估器？初步研究](https://arxiv.org/abs/2303.04048)（2023年3月）
- [情感计算是否将从基础模型和通用人工智能中出现？在ChatGPT上的首次评估](https://arxiv.org/abs/2303.03186)（2023年3月）
- [UZH_CLyp在SemEval-2023任务9中的应用：面向Tweet亲密性预测的跨语言fine-tuning和ChatGPT数据生成](https://arxiv.org/abs/2303.01194)（2023年3月）
- [如何格式化输入到ChatGPT模型中](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb)（2023年3月）
- [ChatGPT能评估人类个性吗？一个通用的评估框架](https://arxiv.org/abs/2303.01248)（2023年3月）
- [通过ChatGPT进行跨语言摘要](https://arxiv.org/abs/2302.14229)（2023年2月）
- [ChatAug：利用ChatGPT进行文本数据增强](https://arxiv.org/abs/2302.13007)（2023年2月）
- [ChatGPT博士，请告诉我我想要听什么：提示性知识如何影响健康答案的正确性](https://arxiv.org/abs/2302.13793)（2023年2月）
- [ChatGPT在数学题中的独立评估（MWP）](https://arxiv.org/abs/2302.13814)（2023年2月）
- [ChatGPT：2.5个月后的元分析](https://arxiv.org/abs/2302.13795)（2023年2月）
- [让我们聊天！与ChatGPT的对话：技术，应用和局限性](https://arxiv.org/abs/2302.13817)（2023年2月）
- [检查你的事实并重试：利用外部知识和自动反馈来改善大型语言模型](https://arxiv.org/abs/2302.12813)（2023年2月）
- [关于ChatGPT的稳健性：对抗和超出分布的视角](https://arxiv.org/abs/2302.12095)（2023年2月）
- [生成式人工智能模型，例如ChatGPT如何在SPC实践，教育和研究中被（误）使用？探索性研究](https://arxiv.org/abs/2302.10916)（2023年2月）
- [ChatGPT是否也可以理解？ChatGPT和fine-tuned BERT的比较研究](https://arxiv.org/abs/2302.10198)（2023年2月）- [一个增强ChatGPT提示工程的提示模式目录](https://arxiv.org/abs/2302.11382) (2023年2月)
- [通过与ChatGPT聊天实现零-shot信息提取](https://arxiv.org/abs/2302.10205) (2023年2月)
- [ChatGPT：万能而不精](https://arxiv.org/abs/2302.10724) (2023年2月)
- [在决策制定和空间推理方面对ChatGPT和DALL-E 2进行初步评估](https://arxiv.org/abs/2302.09068) (2023年2月)
- [网民、学者和信息专家对AI的看法，特别是ChatGPT](https://arxiv.org/abs/2302.07136) (2023年2月)
- [ChatGPT中的语言歧义分析](https://arxiv.org/abs/2302.06426) (2023年2月)
- [面向知识图谱的ChatGPT与传统问答的比较：现状和未来发展方向](https://arxiv.org/abs/2302.06466) (2023年2月)
- [ChatGPT和生成式AI对科学意味着什么](https://www.nature.com/articles/d41586-023-00340-6) (2023年2月)
- [应用BERT和ChatGPT分析斯蒂弗斯-约翰逊病的科学文献情感](https://arxiv.org/abs/2302.06474) (2023年2月)
- [探索ChatGPT的人工智能伦理：一种诊断分析](https://arxiv.org/abs/2301.12867) (2023年1月)
- [ChatGPT有何益处和挑战？大型语言模型在教育中的应用](https://www.edu.sot.tum.de/fileadmin/w00bed/hctl/_my_direct_uploads/ChatGPT_for_Good_.pdf) (2023年1月)
- [对话AI的政治意识形态：ChatGPT倾向于环保和左倾自由主义](https://arxiv.org/abs/2301.01768) (2023年1月)
- [提高可靠性的技术 - OpenAI Cookbook](https://github.com/openai/openai-cookbook/blob/main/techniques_to_improve_reliability.md)
- [令人惊叹的ChatGPT提示](https://github.com/f/awesome-chatgpt-prompts)
- [介绍ChatGPT](https://openai.com/blog/chatgpt) (2022年11月)

---
[上一部分（应用）](./prompts-applications.md)

[下一部分（对抗提示）](./prompts-adversarial.md)