# 提示词（prompt）工程指南（七）：可靠性

我们已经看到了如何使用少样本学习等技术来完成各种任务的精细提示的有效性。 当我们考虑在LLMs之上构建实际应用程序时，必须考虑这些语言模型的可靠性。 本指南着重于展示有效的提示技巧，以提高像GPT-3这样的LLMs的可靠性。 感兴趣的一些话题包括一般性，校准，偏差，社会偏见和实事求是等。

> 完整的中文版本指南和更丰富的参考资料在 Github 和 Gitee 中，自动持续翻译更新：
> 🐙 关于提示词工程（prompt）的指南、论文、讲座、笔记本和资源大全
>
> - <https://github.com/yunwei37/Prompt-Engineering-Guide-zh-CN>
> - <https://gitee.com/yunwei37/Prompt-Engineering-Guide-zh-CN>

主题:

<!-- TOC -->

- [提示词（prompt）工程指南（七）：可靠性](#提示词prompt工程指南七可靠性)
  - [实事求是](#实事求是)
  - [偏差](#偏差)
    - [样本分布](#样本分布)
    - [举例顺序](#举例顺序)
  - [参考文献](#参考文献)

<!-- /TOC -->
---
## 实事求是

LLMs倾向于生成听起来连贯且令人信服但有时是虚构的响应。 改进提示可以帮助改进模型生成更准确/实际的响应，并减少生成不一致和虚构响应的可能性。

一些解决方法可能包括：

- 提供基础事实（例如，相关的文章段落或维基百科条目）作为上下文的一部分，以减少模型产生虚构文本的可能性。
- 配置模型以生成较少多样化的响应，方法是降低概率参数并指示它在不知道答案时承认（例如，“我不知道”）。
- 在提示中提供例子问题和可能知道和不知道的响应的组合。

让我们看一个简单的例子：

*提示：*

```
Q：什么是原子？
A：原子是组成一切的微小粒子。

Q：阿尔文·蒙茨（Alvan Muntz）是谁？
A：？

Q：Kozar-09是什么？
A：？

Q：火星有多少个卫星？
A：有两个，分别是Deimos和Phobos。

Q：Neto Beto Roberto是谁？
```

*输出:*

```
A：？
```

这个名字“Neto Beto Roberto”连我都是凭想象杜撰的，所以模型在这个情况下是正确的。 尝试稍微改变问题，看看能否让其发挥作用。 基于你所学的一切，还有其他方法可以进一步改进它。 

---
## 偏差

LLMs产生的问题世代潜在地可能是有害的，并且可能显示偏见，这可能会恶化模型在下游任务中的表现。其中一些可以通过有效的提示策略来缓解，但可能需要更高级的解决方案，如调节和过滤。

### 样本分布

在进行几次学习时，示例的分布是否会影响模型的性能或以某种方式使模型存在偏差？我们可以在这里进行简单的测试。

*提示：*
```
问题：我刚刚得到了最好的消息！
回答：积极

问：我们刚在工作中得到了加薪！
回答：积极

问：我为今天所取得的成就感到非常自豪。
答：积极

问：我度过了最好的一天！
答：积极

问：我真的很期待周末。
回答：积极

问：我刚刚得到了最好的礼物！
回答：积极

问：我现在非常高兴。
回答：积极

问：我很幸运拥有如此出色的家庭。
回答：积极

问：外面的天气很阴沉。
回答：消极

问：我刚接到了一些可怕的消息。
回答：消极

问：那留下了不好的印象。
答：
```

*输出：*
```
消极
```

在上面的例子中，看起来样本分布并不会使模型存在偏差。这很好。让我们尝试另一个更难分类的例子，看看模型的表现：

*提示：*
```
问：这里的食物很美味！
回答：积极

问：我真的厌倦了这门课程。
回答：消极

问：我简直不敢相信我没通过考试。
回答：消极

问：今天我过得非常愉快！
回答：积极

问：我讨厌这份工作。
回答：消极

问：这里的服务太差了。
回答：消极

问：我对我的生活感到非常沮丧。
回答：消极

问：我永远不能休息。
回答：消极

问：这餐食物真难吃。
回答：消极

问：我受不了我的老板。
回答：消极

问：我有一种感觉。
答：
```

*输出：*
```
消极
```

尽管最后一句话有些主观，但我翻转了分布，改为使用8个积极的例子和2个消极的例子，然后再尝试相同的句子。猜猜模型的回答是什么？它回答“积极”。该模型可能对情感分类有很多知识，因此很难使其显示出此问题的偏见。建议避免偏斜分布，相反，为每个标签提供更平衡的示例数量。对于那些模型没有太多了解的更难的任务，它可能会更加困难。

### 举例顺序

在执行少样本学习时，顺序是否影响模型的性能或以某种方式影响模型？

您可以尝试上面的例子，看看是否可以通过更改顺序使模型偏向某个标签。建议将示例随机排序。例如，避免先出所有正面例子，然后在最后出负面例子。如果标签的分布不均衡，这个问题会进一步扩大。始终确保大量尝试以减少此类偏差。

---

其他即将出现的主题:
- 扰动
- 假相似性
- 领域转移
- 有毒性
- 仇恨言论 / 冒犯内容
- 刻板印象偏见
- 性别偏见
- 即将推出！
- 红队测试

---
## 参考文献

- [宪法AI：AI反馈中的无害性](https://arxiv.org/abs/2212.08073)（2022年12月）
- [重新思考演示的角色：何以使上下文学习起作用？](https://arxiv.org/abs/2202.12837)（2022年10月）
- [提示GPT-3可靠性](https://arxiv.org/abs/2210.09150)（2022年10月）
- [关于使语言模型成为更好的推理者的进展](https://arxiv.org/abs/2206.02336)（2022年6月）
- [ML安全中未解决的问题](https://arxiv.org/abs/2109.13916)（2021年9月）
- [将语言模型红队测试以减少伤害：方法、扩展行为和教训](https://arxiv.org/abs/2209.07858)（2022年8月）。- [StereoSet：在预训练语言模型中测量陈规偏见](https://aclanthology.org/2021.acl-long.416/)（2021年8月）
- [使用前校准：改善语言模型的少量样本性能](https://arxiv.org/abs/2102.09690v2)（2021年2月）
- [提高可靠性的技术 - OpenAI Cookbook](https://github.com/openai/openai-cookbook/blob/main/techniques_to_improve_reliability.md)

---
[上一个部分（对抗性提示）](./prompts-adversarial.md)

[下一个部分（杂项）](./prompts-miscellaneous.md)

> 开源、免费自动持续翻译更新关于 GPT 和 prompt 工程的资料合集并同步国内 Gitee 镜像加速访问：
> 
> 关于提示词工程（prompt）的指南、论文、讲座、笔记本和资源大全（自动持续更新）：
> 
> - https://github.com/yunwei37/Prompt-Engineering-Guide-zh-CN
> - https://gitee.com/yunwei37/Prompt-Engineering-Guide-zh-CN
>
> 关于 GPT-4 语言模型的提示（prompt）、工具和资源的中文精选列表（自动持续更新）
>
> - https://github.com/yunwei37/awesome-gpt4-zh-CN
> - https://gitee.com/yunwei37/awesome-gpt4-zh-CN
>
> 使用 OpenAI API 的例子和中文指南（自动持续翻译更新 OpenAI 官方文档）
>
> - https://github.com/yunwei37/openai-cookbook-zh-cn
> - https://gitee.com/yunwei37/openai-cookbook-zh-cn
> 
> 这个资源库包含了为 Prompt 工程手工整理的资源中文清单，重点是生成性预训练变换器（GPT）、ChatGPT、PaLM 等（自动持续更新）
>
> - https://github.com/yunwei37/Awesome-Prompt-Engineering-ZH-CN
> - https://gitee.com/yunwei37/Awesome-Prompt-Engineering-ZH-CN
